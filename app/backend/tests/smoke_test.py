"""End-to-end smoke test for T2/T3/T4.

Bypasses T1 by manually placing a SKAB CSV + sidecar JSON into the
upload tmp dir, then hits the inference and report code paths directly
(no HTTP layer). Run from `app/backend`:

    python tests/smoke_test.py                          # uses bundled sample.csv
    python tests/smoke_test.py --source path/to/file.csv
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from pathlib import Path

# Make `app.*` imports work when invoked from `app/backend/`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402
from app.inference import predict as inference  # noqa: E402
from app.inference.preprocess import validate_schema  # noqa: E402
from app.inference.registry import MODEL_REGISTRY  # noqa: E402
from app.reports.pdf import build_pdf  # noqa: E402

import pandas as pd  # noqa: E402


DEFAULT_SOURCE = Path(__file__).resolve().parents[1] / "artifacts" / "sample.csv"
THRESHOLD = 0.5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source", type=Path, default=DEFAULT_SOURCE,
        help=f"SKAB-format CSV to test against (default: {DEFAULT_SOURCE.name} "
             f"bundled in app/backend/artifacts/).",
    )
    args = parser.parse_args()
    source_csv: Path = args.source

    if not source_csv.is_file():
        print(f"[FAIL] Source CSV not found: {source_csv}")
        return 1

    file_id = uuid.uuid4().hex
    csv_target = settings.upload_tmp_dir / f"{file_id}.csv"
    sidecar_target = settings.upload_tmp_dir / f"{file_id}.json"

    print(f"→ Staging upload at {csv_target}")
    shutil.copy(source_csv, csv_target)

    df = pd.read_csv(csv_target, sep=";")
    info = validate_schema(df)
    print(f"  Schema OK · rows={info.rows} has_label={info.has_label}")

    sidecar_target.write_text(json.dumps({
        "filename": source_csv.name,
        "rows": info.rows,
        "has_label": info.has_label,
        "columns": info.columns_detected,
        "uploaded_at": "smoke-test",
    }), encoding="utf-8")

    # ── 1. /api/predict (single model) ────────────────────────────
    print("\n[T2] /api/predict on xgb")
    prepared = inference.prepare_input(file_id)
    out = inference.run_model("xgb", prepared, threshold=THRESHOLD)
    assert out.total_windows > 0, "no windows produced"
    assert len(out.probs) == out.total_windows
    if info.has_label:
        assert out.metrics is not None, "metrics should be computed when labels exist"
    print(f"  total={out.total_windows} fault={out.fault_windows} "
          f"peak={out.peak_prob:.3f}")
    if out.metrics:
        print(f"  metrics={out.metrics}")
    assert out.unavailable is False

    # ── 2. /api/predict/compare (all 8 models) ────────────────────
    print("\n[T3] /api/predict/compare across all 8 models")
    for mid in MODEL_REGISTRY:
        r = inference.run_model(mid, prepared, threshold=THRESHOLD)
        flag = "UNAVAILABLE" if r.unavailable else "ok"
        f1 = f"f1={r.metrics['f1']:.3f}" if r.metrics else "no metrics"
        print(f"  {mid:<12} {flag:<11} fault={r.fault_windows:>5} {f1}")
    # At least the transformer should be unavailable (artifact missing).
    transformer_out = inference.run_model("transformer", prepared, THRESHOLD)
    assert transformer_out.unavailable is True, \
        "transformer should be flagged unavailable when .pt is missing"

    # ── 3. /api/report (PDF) ──────────────────────────────────────
    print("\n[T4] /api/report — building PDF for xgb")
    pdf_bytes = build_pdf(prepared, out, threshold=THRESHOLD)
    pdf_path = Path("smoke_report.pdf")
    pdf_path.write_bytes(pdf_bytes)
    print(f"  PDF size: {len(pdf_bytes) / 1024:.1f} KB → {pdf_path.resolve()}")
    assert pdf_bytes.startswith(b"%PDF-"), "output is not a valid PDF"
    assert len(pdf_bytes) < 1_000_000, "PDF should be < 1 MB (FR-17 implied)"

    # Cleanup tmp upload (NFR-13)
    csv_target.unlink(missing_ok=True)
    sidecar_target.unlink(missing_ok=True)

    print("\n[PASS] All smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
