from pydantic import BaseModel, ConfigDict, Field

# Pydantic v2 reserves the `model_` namespace; we use `model_id` as a
# domain concept (which trained model to invoke), so disable the
# protection here.
_AllowModelPrefix = ConfigDict(protected_namespaces=())


# ─── Requests ─────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    model_config = _AllowModelPrefix

    file_id: str
    model_id: str
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    horizon: int | None = None  # FR-13 (Could) — None ⇒ trained default


class CompareRequest(BaseModel):
    file_id: str
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


# ─── Response building blocks ─────────────────────────────────────────

class AnomalyWindow(BaseModel):
    start_idx: int
    end_idx: int
    peak_prob: float


class ModelResult(BaseModel):
    """Per-model output. Used both as the body of `/api/predict` and as
    one row inside `/api/predict/compare`."""

    model_config = _AllowModelPrefix

    model_id: str
    probs: list[float]
    anomaly_windows: list[AnomalyWindow]
    peak_idx: int
    peak_prob: float
    total_windows: int
    fault_windows: int
    metrics: dict | None = None        # FR-21 — None when has_label is False
    unavailable: bool = False          # FR-19 — True when artifact missing
    error: str | None = None           # set together with unavailable=True


# ─── Top-level responses ──────────────────────────────────────────────

class PredictResponse(ModelResult):
    """Single-model endpoint response — same shape as one ModelResult."""


class CompareResponse(BaseModel):
    """Multi-model endpoint response (FR-19)."""

    models: list[ModelResult]
