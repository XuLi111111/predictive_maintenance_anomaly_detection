"""Core inference helpers shared by all prediction endpoints.

The route layer (`api/routes/predict.py`, `api/routes/report.py`) only
deals with HTTP framing — actual model invocation and post-processing
live here so they can be unit-tested without spinning up FastAPI.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import HTTPException
from sklearn.metrics import f1_score, precision_score, recall_score

from app.core.config import settings
from app.inference import loader
from app.inference.preprocess import build_labels, build_windows


# ─── Public dataclasses ───────────────────────────────────────────────

@dataclass
class PreparedInput:
    """Output of `prepare_input` — passed to one or many model runs.

    We keep both the raw and scaled feature matrices because not every
    model in the registry was trained on scaled inputs. Tree-based
    ensembles (xgb/rf/et/gb) used raw flattened windows during training;
    feeding them scaled inputs silently breaks their predictions.
    `run_model` picks the right matrix based on `meta.requires_scaling`.
    """

    file_id: str
    df: pd.DataFrame
    windows_raw: np.ndarray     # (N, 20*8) — unscaled flattened windows
    windows_scaled: np.ndarray  # (N, 20*8) — scaler.transform applied
    has_label: bool
    labels: np.ndarray | None
    sidecar: dict


@dataclass
class ModelOutput:
    """Result of one `run_model` call."""

    model_id: str
    probs: list[float]
    anomaly_windows: list[dict]
    peak_idx: int
    peak_prob: float
    total_windows: int
    fault_windows: int
    metrics: dict | None = None
    unavailable: bool = False
    error: str | None = None
    extras: dict = field(default_factory=dict)


# ─── 1. Load + prep (run once per request, even for /compare) ─────────

def prepare_input(file_id: str) -> PreparedInput:
    """Load a previously-uploaded CSV by ``file_id``, validate, build
    windows, and apply the training-fit scaler.

    Raises 404 if the file is not on disk (e.g. expired tmp).
    """
    csv_path = settings.upload_tmp_dir / f"{file_id}.csv"
    sidecar_path = settings.upload_tmp_dir / f"{file_id}.json"

    if not csv_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Uploaded file '{file_id}' not found. "
                   f"It may have been cleaned up — please re-upload.",
        )

    sidecar = _read_sidecar(sidecar_path)
    df = pd.read_csv(csv_path, sep=";")

    has_label = bool(sidecar.get("has_label", settings.label_column in df.columns))

    windows_3d = build_windows(df)
    if windows_3d.size == 0:
        raise HTTPException(
            status_code=400,
            detail="File is too short to produce any sliding windows "
                   f"(need at least {settings.window_size + settings.horizon} rows).",
        )

    # sklearn models expect 2-D (N, F). We flatten once here so all
    # downstream models can share the work. Deep-learning models that
    # need (N, T, C) can reshape using `settings.window_size` and
    # `len(settings.sensor_columns)`.
    n, t, c = windows_3d.shape
    windows_raw = windows_3d.reshape(n, t * c)

    scaler = loader.load_scaler()
    windows_scaled = scaler.transform(windows_raw)

    labels = build_labels(df) if has_label else None

    return PreparedInput(
        file_id=file_id,
        df=df,
        windows_raw=windows_raw,
        windows_scaled=windows_scaled,
        has_label=has_label,
        labels=labels,
        sidecar=sidecar,
    )


# ─── 2. Run a single model ────────────────────────────────────────────

def run_model(
    model_id: str,
    prepared: PreparedInput,
    threshold: float,
) -> ModelOutput:
    """Invoke one model on already-prepared inputs.

    If the model artifact is missing (e.g. transformer pending), returns
    a `ModelOutput` with `unavailable=True` instead of raising — this
    keeps `/compare` resilient when only one model is missing.
    """
    meta = loader.get_model_meta(model_id)

    try:
        model = loader.load_model(model_id)
    except HTTPException as exc:
        if exc.status_code == 503:
            return ModelOutput(
                model_id=model_id,
                probs=[],
                anomaly_windows=[],
                peak_idx=-1,
                peak_prob=0.0,
                total_windows=0,
                fault_windows=0,
                metrics=None,
                unavailable=True,
                error=str(exc.detail),
            )
        raise

    if meta.is_dl:
        # T7 — TransformerFusionLite needs (N, T, F) raw windows scaled
        # by Xu Li's per-feature scaler, NOT the classical per-(t×f)
        # `scaler.pkl`. See loader.load_transformer_scaler for the
        # rationale.
        n = prepared.windows_raw.shape[0]
        t = settings.window_size
        c = len(settings.sensor_columns)
        windows_3d = prepared.windows_raw.reshape(n, t, c)
        tx_scaler = loader.load_transformer_scaler()
        windows_3d = tx_scaler.transform(windows_3d.reshape(-1, c)).reshape(n, t, c)
        probs = _run_torch(model, windows_3d)
    else:
        # Classical models: pick the matrix matching their training regime.
        features = (
            prepared.windows_scaled if meta.requires_scaling
            else prepared.windows_raw
        )
        probs = _run_sklearn(model, features)

    return _summarise(model_id, probs, prepared, threshold)


# ─── 3. Threshold + metrics ───────────────────────────────────────────

def _summarise(
    model_id: str,
    probs: np.ndarray,
    prepared: PreparedInput,
    threshold: float,
) -> ModelOutput:
    """Apply threshold, find anomaly windows, compute metrics if possible."""
    pred = (probs >= threshold).astype(np.int64)

    anomaly_windows = _find_runs(probs, pred)
    peak_idx = int(np.argmax(probs)) if len(probs) > 0 else -1
    peak_prob = float(probs[peak_idx]) if peak_idx >= 0 else 0.0

    metrics: dict | None = None
    if prepared.has_label and prepared.labels is not None and len(prepared.labels) == len(pred):
        metrics = {
            "precision": float(precision_score(prepared.labels, pred, zero_division=0)),
            "recall":    float(recall_score(prepared.labels, pred, zero_division=0)),
            "f1":        float(f1_score(prepared.labels, pred, zero_division=0)),
        }

    return ModelOutput(
        model_id=model_id,
        probs=probs.tolist(),
        anomaly_windows=anomaly_windows,
        peak_idx=peak_idx,
        peak_prob=peak_prob,
        total_windows=int(len(probs)),
        fault_windows=int(pred.sum()),
        metrics=metrics,
    )


def _find_runs(probs: np.ndarray, pred: np.ndarray) -> list[dict]:
    """Group consecutive `pred==1` indices into anomaly windows.

    Each entry: ``{start_idx, end_idx, peak_prob}``. ``end_idx`` is
    inclusive — matches what the front-end expects when shading regions.
    """
    if len(pred) == 0:
        return []
    runs: list[dict] = []
    in_run = False
    start = 0
    for i, v in enumerate(pred):
        if v and not in_run:
            in_run = True
            start = i
        elif not v and in_run:
            in_run = False
            end = i - 1
            runs.append({
                "start_idx": int(start),
                "end_idx":   int(end),
                "peak_prob": float(probs[start : end + 1].max()),
            })
    if in_run:
        end = len(pred) - 1
        runs.append({
            "start_idx": int(start),
            "end_idx":   int(end),
            "peak_prob": float(probs[start : end + 1].max()),
        })
    return runs


# ─── 4. Backend-specific model invocations ────────────────────────────

def _run_sklearn(model: Any, windows_2d: np.ndarray) -> np.ndarray:
    """All sklearn / xgboost models expose `predict_proba`. Some (SVC
    without `probability=True`) do not — fall back to `decision_function`
    pushed through a sigmoid as a best-effort score."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(windows_2d))[:, 1]
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(windows_2d))
        return 1.0 / (1.0 + np.exp(-scores))
    raise HTTPException(
        status_code=500,
        detail=f"Model {type(model).__name__} exposes neither "
               f"predict_proba nor decision_function.",
    )


def _run_torch(model: Any, windows_3d: np.ndarray) -> np.ndarray:
    """Run TransformerFusionLite on already-3D-and-scaled windows.

    Caller is responsible for applying the per-feature transformer scaler
    before calling this — see `run_model` is_dl branch.
    """
    import torch  # type: ignore

    tensor = torch.from_numpy(windows_3d).float()
    with torch.no_grad():
        out = model(tensor)
    # Model emits raw logits (B,). Apply sigmoid to get probabilities.
    out = out.detach().cpu().numpy().ravel()
    return 1.0 / (1.0 + np.exp(-out))


# ─── 5. Sidecar helpers ───────────────────────────────────────────────

def _read_sidecar(path: Path) -> dict:
    """Return the sidecar metadata, or an empty dict if it is missing
    (older uploads, manual file drops during development, etc.)."""
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
