from fastapi import APIRouter, HTTPException

from app.schemas.predict import CompareRequest, PredictRequest

router = APIRouter()


@router.post("/predict")
def predict(req: PredictRequest):
    """FR-09/10/11/14/15/16/20/21 — run sliding-window inference for one model.

    TODO (teammate): load file_id, apply scaler.pkl, build (N,20,8) windows,
    call model.predict_proba, return probs/anomaly_windows/summary.
    Suppress metrics when has_label is False (FR-21).
    """
    raise HTTPException(status_code=501, detail="predict endpoint not implemented")


@router.post("/predict/compare")
def predict_compare(req: CompareRequest):
    """FR-19 — run all 8 models on the same input for the comparison view.

    TODO (teammate): iterate MODEL_REGISTRY, return per-model probs + metrics.
    """
    raise HTTPException(status_code=501, detail="compare endpoint not implemented")
