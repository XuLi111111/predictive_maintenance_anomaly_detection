"""Upload endpoint (T1) + sample-CSV asset endpoint (FR-05).

The upload flow:
1. Read the uploaded file into memory (capped by `settings.max_upload_mb`).
2. Detect the CSV separator (SKAB uses ';', some exports use ',').
3. Validate against the SKAB schema via `inference.preprocess.validate_schema`.
4. Persist to `settings.upload_tmp_dir/{file_id}.csv` re-serialised with ';'
   as the canonical separator so the downstream `predict.py` can keep its
   simple `pd.read_csv(..., sep=';')` call.
5. Persist a sidecar JSON with metadata that the predict/report endpoints
   can consume without re-parsing the CSV.
6. Schedule a background task to delete the file after a TTL (NFR-13).

The endpoint returns enough information for the front-end to render an
informative validation row before the user picks a model.
"""
from __future__ import annotations

import io
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
)
from fastapi.responses import FileResponse

from app.core.config import settings
from app.inference.preprocess import SchemaError, validate_schema

router = APIRouter()


# Files older than this are removed on the *next* upload via inline GC.
# We deliberately don't schedule a real timer — running asyncio.sleep
# inside a BackgroundTask blocks Starlette's TestClient (which waits for
# all tasks before returning), and a real cron is overkill for a
# single-user demo. _gc_old_uploads runs at the top of every request.
_RETENTION_SECONDS = 30 * 60


@router.post("/upload")
async def upload_csv(
    file: UploadFile = File(...),
):
    """FR-01/02/03/04/07/21 — validate SKAB schema and stage for prediction.

    Returns ``{file_id, rows, time_range, columns_detected, has_label,
    warnings, model_default}`` so the UI can show schema feedback without
    a second round-trip.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail=(
                "Only .csv files are accepted.\n"
                "How to fix: export your data as CSV from Excel "
                "(File → Save As → CSV UTF-8), Python "
                "(df.to_csv('out.csv', sep=';', index=False)), or your "
                "SCADA system. If the file is already CSV-formatted but "
                "has a different extension, rename it to end with '.csv'."
            ),
        )

    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(
            status_code=400,
            detail=(
                "Uploaded file is empty (0 bytes).\n"
                "How to fix: verify the file actually contains data "
                "before uploading — open it in a text editor and confirm "
                "you can see header + rows."
            ),
        )

    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(raw) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large ({len(raw) / 1_048_576:.1f} MB). "
                f"Limit is {settings.max_upload_mb} MB.\n"
                f"How to fix: trim the file to a shorter time window "
                f"(at 1 Hz, 1 hour = ~360 KB, so 50 MB covers ~140 hours). "
                f"Or split into multiple uploads and analyse separately."
            ),
        )

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"File is not UTF-8 encoded ({exc.reason} at byte "
                f"{exc.start}).\n"
                f"How to fix: re-save the file as UTF-8. In Excel: "
                f"File → Save As → 'CSV UTF-8 (Comma delimited)'. In "
                f"Python: df.to_csv('out.csv', encoding='utf-8'). Avoid "
                f"GBK / Windows-1252 / UTF-16 — these will all fail here."
            ),
        )

    sep = _detect_separator(text)

    try:
        df = pd.read_csv(io.StringIO(text), sep=sep)
    except (pd.errors.ParserError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Could not parse the CSV with separator '{sep}': {exc}.\n"
                f"How to fix: open the file in a text editor and check "
                f"that every row has the same number of columns. Common "
                f"causes: a sensor value contains a literal '{sep}', or "
                f"the file mixes ',' and ';' separators across rows. "
                f"SKAB-format files should use ';' consistently."
            ),
        )

    try:
        info = validate_schema(df)
    except SchemaError as exc:
        raise HTTPException(status_code=400, detail=str(exc.message))

    # Cheap garbage collection of any stale uploads from previous runs.
    _gc_old_uploads(settings.upload_tmp_dir, _RETENTION_SECONDS)

    file_id = uuid.uuid4().hex
    csv_path = settings.upload_tmp_dir / f"{file_id}.csv"
    sidecar_path = settings.upload_tmp_dir / f"{file_id}.json"

    # Re-serialise canonically so predict.py can keep its simple sep=';' read.
    df.to_csv(csv_path, sep=";", index=False)

    sidecar = {
        "filename": file.filename,
        "rows": info.rows,
        "has_label": info.has_label,
        "columns": info.columns_detected,
        "time_range": list(info.time_range),
        "uploaded_at": datetime.utcnow().isoformat() + "Z",
        "original_separator": sep,
    }
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    warnings: list[str] = list(info.warnings)
    if not info.has_label:
        warnings.append(
            "No 'anomaly' column detected — accuracy metrics (Precision / "
            "Recall / F1) will be hidden because there's no ground truth "
            "to compare against. The probability chart and alert flags "
            "are still produced normally. "
            "How to fix (optional): if your dataset has known fault "
            "windows, add an 'anomaly' column with 1 inside faults and 0 "
            "elsewhere."
        )

    return {
        "file_id": file_id,
        "rows": info.rows,
        "time_range": {"start": info.time_range[0], "end": info.time_range[1]},
        "columns_detected": info.columns_detected,
        "has_label": info.has_label,
        "warnings": warnings,
        "model_default": "xgb",
    }


@router.get("/sample-csv")
def sample_csv():
    """FR-05 — stream a small valid SKAB CSV from artifacts/sample.csv.

    The file is bundled with the backend image so users can download it
    without first uploading anything. Generated from valve2/0.csv during
    deployment (see scripts/) — committed under app/backend/artifacts/.
    """
    path = settings.artifact_dir / "sample.csv"
    if not path.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Sample CSV not bundled in {settings.artifact_dir}. "
                   f"Place 'sample.csv' under app/backend/artifacts/.",
        )
    return FileResponse(
        path,
        media_type="text/csv",
        filename="pump_detect_sample.csv",
    )


# ─── Internals ────────────────────────────────────────────────────────

def _detect_separator(text: str) -> str:
    """SKAB CSVs use ';'. Some exports use ','. Pick whichever the
    header line has more of. Falls back to ';' (the project default)."""
    first_line = text.split("\n", 1)[0]
    semi = first_line.count(";")
    comma = first_line.count(",")
    if semi == 0 and comma == 0:
        return ";"
    return ";" if semi >= comma else ","


def _gc_old_uploads(directory: Path, max_age_seconds: int) -> None:
    """Delete any files in `directory` older than the TTL.

    Cheap inline GC: runs on every upload. For our workload (single-user,
    handful of uploads) this is more than enough — no need for a cron task.
    """
    now = time.time()
    try:
        for entry in directory.iterdir():
            try:
                if not entry.is_file():
                    continue
                if now - entry.stat().st_mtime > max_age_seconds:
                    entry.unlink(missing_ok=True)
            except OSError:
                continue
    except FileNotFoundError:
        os.makedirs(directory, exist_ok=True)
