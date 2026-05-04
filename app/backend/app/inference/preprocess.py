"""Schema validation and sliding-window construction.

Both the training pipeline and the inference API must use the identical
windowing logic to satisfy NFR-05 (reproducibility). The training script is
SKAB/Build_Dataset_SKAB_DLpipeline_By_David.py — once this module is fleshed
out, that script should import from here rather than keep its own copy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.config import settings


class SchemaError(ValueError):
    """Raised when an uploaded CSV does not conform to the SKAB schema."""


def validate_schema(df: pd.DataFrame) -> dict:
    """FR-02/03/04 — validate against settings.sensor_columns. Return metadata.

    TODO (teammate): check timestamp column, all 8 sensor columns present in
    the correct dtype, detect optional 'anomaly' column for FR-21, raise
    SchemaError with a plain-language message listing missing/extra columns.
    """
    raise NotImplementedError


def build_windows(df: pd.DataFrame, window_size: int | None = None) -> np.ndarray:
    """FR-10 — build (N, window_size, 8) sliding windows with stride 1.

    TODO (teammate): mirror build_samples_from_one_file from the training
    script. Drop datetime/anomaly/changepoint, sort by datetime, slide.
    """
    _ = window_size or settings.window_size
    raise NotImplementedError
