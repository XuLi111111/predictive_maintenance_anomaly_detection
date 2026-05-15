"""Prediction endpoints (T2 + T3).

Both endpoints share the same preparation step (load CSV, build windows,
apply scaler) and only differ in how many models they invoke. The shared
work lives in `app.inference.predict`.
"""
from fastapi import APIRouter

from app.inference import predict as inference
from app.inference.registry import MODEL_REGISTRY
from app.schemas.predict import (
    CompareRequest,
    CompareResponse,
    ModelResult,
    PredictRequest,
    PredictResponse,
)

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """T2 — single-model sliding-window inference.

    Implements FR-09 (use trained scaler), FR-10 (window=20, stride=1),
    FR-11 (single call returns probabilities), FR-14/15/16 (threshold +
    anomaly windows), FR-20 (summary fields), FR-21 (metrics suppressed
    when no ground truth).
    """
    prepared = inference.prepare_input(req.file_id)
    result = inference.run_model(req.model_id, prepared, req.threshold)
    return PredictResponse(**_serialise(result))


@router.post("/predict/compare", response_model=CompareResponse)
def predict_compare(req: CompareRequest) -> CompareResponse:
    """T3 — run all 8 registered models on the same uploaded file (FR-19).

    Models whose artifact is missing (e.g. transformer pending T7) are
    returned with `unavailable=True` instead of failing the whole call.
    """
    prepared = inference.prepare_input(req.file_id)
    results = [
        inference.run_model(mid, prepared, req.threshold)
        for mid in MODEL_REGISTRY
    ]
    return CompareResponse(
        models=[ModelResult(**_serialise(r)) for r in results],
    )


# ─── Internals ────────────────────────────────────────────────────────

def _serialise(out: inference.ModelOutput) -> dict:
    """Map the inference dataclass to the Pydantic response shape."""
    return {
        "model_id":         out.model_id,
        "probs":            out.probs,
        "anomaly_windows":  out.anomaly_windows,
        "peak_idx":         out.peak_idx,
        "peak_prob":        out.peak_prob,
        "total_windows":    out.total_windows,
        "fault_windows":    out.fault_windows,
        "metrics":          out.metrics,
        "unavailable":      out.unavailable,
        "error":            out.error,
    }
