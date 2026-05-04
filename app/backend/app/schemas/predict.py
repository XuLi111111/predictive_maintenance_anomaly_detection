from pydantic import BaseModel, ConfigDict, Field

# Pydantic v2 reserves the `model_` namespace; we use `model_id` as a domain
# concept (which trained model to invoke), so disable the protection here.
_AllowModelPrefix = ConfigDict(protected_namespaces=())


class PredictRequest(BaseModel):
    model_config = _AllowModelPrefix

    file_id: str
    model_id: str
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    horizon: int | None = None  # FR-13 (Could) — None ⇒ use trained default (10)


class CompareRequest(BaseModel):
    file_id: str
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class AnomalyWindow(BaseModel):
    start_idx: int
    end_idx: int
    peak_prob: float


class PredictResponse(BaseModel):
    model_config = _AllowModelPrefix

    model_id: str
    probs: list[float]
    anomaly_windows: list[AnomalyWindow]
    peak_idx: int
    peak_prob: float
    total_windows: int
    fault_windows: int
    metrics: dict | None = None  # FR-21 — None when has_label is False
