"""Schema validation and sliding-window construction.

Both the training pipeline and the inference API must use identical
windowing logic to satisfy NFR-05 (reproducibility). The training script
is `SKAB/Build_Dataset_SKAB_DLpipeline_By_David.py`; the
`build_samples_from_one_file` function there is the source of truth that
this module mirrors.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.core.config import settings


# ─── Errors ────────────────────────────────────────────────────────────

class SchemaError(ValueError):
    """Raised when an uploaded CSV does not conform to the SKAB schema."""

    def __init__(self, message: str, *, missing: list[str] | None = None,
                 extra: list[str] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.missing = missing or []
        self.extra = extra or []


# ─── Schema validation (FR-02 / FR-03 / FR-04 / FR-21) ────────────────

@dataclass
class SchemaInfo:
    """Result of `validate_schema`. Fed to the upload endpoint and the
    sidecar JSON file consumed by predict."""

    rows: int
    has_label: bool
    columns_detected: list[str]
    time_range: tuple[str, str]
    # Non-fatal issues — file accepted but the user should know.
    # Each entry is a plain-language sentence ending with "How to fix: …"
    warnings: list[str]


def validate_schema(df: pd.DataFrame) -> SchemaInfo:
    """Validate that *df* matches the SKAB training schema.

    Rules
    -----
    - Must contain `settings.timestamp_column` (default ``"datetime"``).
    - Must contain all 8 columns in `settings.sensor_columns`, in any
      order, all numeric.
    - May contain `settings.label_column` (``"anomaly"``); presence is
      reported in `SchemaInfo.has_label` for FR-21.
    - May contain extra columns (e.g. ``"changepoint"``); they are
      accepted but ignored downstream.
    - Sampling rate is checked against the 1 Hz training-time cadence
      and produces a *warning* (not a rejection) if the median interval
      is materially different.

    Raises
    ------
    SchemaError
        On fatal schema problems. Every message ends with a "How to fix:"
        sentence so the user knows what to change (FR-03).
    """
    cols = list(df.columns)
    required_sensors = settings.sensor_columns

    # 1. Timestamp column
    ts_col = settings.timestamp_column
    if ts_col not in cols:
        candidates = [c for c in cols
                      if c.lower() in {"timestamp", "time", "date", "datetime"}
                      and c != ts_col]
        rename_hint = (
            f" Looks like your file uses '{candidates[0]}' — "
            f"rename it to '{ts_col}'."
            if candidates else
            f" Add a '{ts_col}' column whose values are timestamps "
            f"like '2020-03-09 15:56:30'."
        )
        raise SchemaError(
            f"Missing required column: '{ts_col}'.\n"
            f"How to fix:{rename_hint}",
            missing=[ts_col],
        )

    # 2. Sensor columns — all 8 must be present (FR-04: do not pad
    #    missing channels with synthetic values; reject the file instead).
    #    We also detect common misspellings (case differences, plural,
    #    underscores) so the user gets actionable rename guidance.
    missing_sensors = [c for c in required_sensors if c not in cols]
    if missing_sensors:
        lower_cols = {c.lower().replace("_", "").replace(" ", ""): c for c in cols}
        rename_pairs: list[str] = []
        for want in missing_sensors:
            key = want.lower().replace("_", "").replace(" ", "")
            actual = lower_cols.get(key)
            if actual and actual != want:
                rename_pairs.append(f"'{actual}' → '{want}'")
        hint_lines = [
            f"How to fix: rename the relevant columns to match the SKAB "
            f"schema exactly (names are case- and space-sensitive).",
        ]
        if rename_pairs:
            hint_lines.append("Suggested renames: " + "; ".join(rename_pairs) + ".")
        hint_lines.append(
            f"All 8 required sensor columns: {', '.join(required_sensors)}."
        )
        raise SchemaError(
            f"Missing {len(missing_sensors)} sensor column(s): "
            f"{', '.join(missing_sensors)}.\n" + "\n".join(hint_lines),
            missing=missing_sensors,
        )

    # 3. Sensor columns must be numeric
    non_numeric = [
        c for c in required_sensors
        if not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric:
        # Spot-check a single offending value so the user can grep their file.
        samples: list[str] = []
        for c in non_numeric[:3]:
            bad_vals = [
                str(v) for v in df[c]
                if not isinstance(v, (int, float)) and not pd.isna(v)
            ][:1]
            if bad_vals:
                samples.append(f"'{c}' contains e.g. {bad_vals[0]!r}")
        sample_hint = "Examples: " + "; ".join(samples) + ". " if samples else ""
        raise SchemaError(
            f"Sensor column(s) contain non-numeric values: "
            f"{', '.join(non_numeric)}.\n"
            f"How to fix: {sample_hint}"
            f"Replace text placeholders (e.g. 'N/A', '--', '#NULL', empty "
            f"strings) with real numbers, or leave the cell completely "
            f"blank if the reading is unavailable. All 8 sensor columns "
            f"must hold floats or integers.",
            extra=non_numeric,
        )

    # 3b. Sensor columns must not contain missing values (NaN). Pandas
    #     happily auto-parses "N/A" / "" / "#NULL" into NaN — the dtype
    #     stays numeric so step 3 doesn't catch it, but the inference
    #     pipeline can't handle gaps. Be explicit so the user can fix
    #     it once rather than getting a cryptic crash later.
    nan_cols = [c for c in required_sensors if df[c].isna().any()]
    if nan_cols:
        nan_counts = {c: int(df[c].isna().sum()) for c in nan_cols}
        counts_str = ", ".join(f"'{c}' ({n} blank)" for c, n in nan_counts.items())
        raise SchemaError(
            f"Sensor column(s) contain missing values: {counts_str}.\n"
            f"How to fix: pandas treats text placeholders like 'N/A', "
            f"'#NULL', or empty cells as missing. Either (a) fill them "
            f"before uploading — interpolate, forward-fill, or replace "
            f"with the last known reading; or (b) drop those rows "
            f"entirely. The model can't infer from gaps; even one "
            f"missing value would corrupt that 20-second window."
        )

    # 4. Parse timestamps so we can report the time range. Failure here
    #    is a schema problem, not a runtime crash.
    try:
        ts = pd.to_datetime(df[ts_col])
    except (ValueError, TypeError) as exc:
        raise SchemaError(
            f"Could not parse '{ts_col}' column as timestamps: {exc}.\n"
            f"How to fix: use ISO-8601 format like '2020-03-09 15:56:30' "
            f"or '2020-03-09T15:56:30'. Avoid: Unix timestamps "
            f"(e.g. 1583768190), Excel serial numbers (e.g. 43899.66), "
            f"and rows where the format changes mid-file."
        ) from exc

    min_rows = settings.window_size + settings.horizon
    if len(df) < min_rows:
        raise SchemaError(
            f"File has only {len(df)} row(s); the model needs at least "
            f"{min_rows} (a {settings.window_size}-sample window plus a "
            f"{settings.horizon}-sample forecast horizon).\n"
            f"How to fix: provide more consecutive sensor readings. At "
            f"the SKAB 1 Hz cadence this is {min_rows} seconds of data; "
            f"recording for ~1 minute is recommended for meaningful results."
        )

    has_label = settings.label_column in cols

    # 5. Sampling-rate sanity check — warn, don't reject. The model was
    #    trained on 1 Hz samples; uneven or wrong-rate data may degrade
    #    predictions.
    warnings = _check_sampling_rate(ts)

    return SchemaInfo(
        rows=len(df),
        has_label=has_label,
        columns_detected=cols,
        time_range=(str(ts.min()), str(ts.max())),
        warnings=warnings,
    )


def _check_sampling_rate(ts: pd.Series) -> list[str]:
    """Inspect the timestamp column for issues that would change the
    *physical-time* span of each 20-sample window (SKAB is 1 Hz, so a
    20-sample window must span 20 seconds). Returns plain-language
    warning strings — never raises.
    """
    warnings: list[str] = []
    try:
        ts_sorted = pd.to_datetime(ts).sort_values().reset_index(drop=True)
        intervals = ts_sorted.diff().dt.total_seconds().dropna()
    except Exception:  # noqa: BLE001
        return warnings
    if intervals.empty:
        return warnings

    median = float(intervals.median())
    std = float(intervals.std()) if len(intervals) > 1 else 0.0
    big_gaps = int((intervals > 2.0).sum())
    duplicates = int((intervals <= 0).sum())

    # Median rate far from 1 Hz → almost certainly wrong rate.
    if abs(median - 1.0) > 0.3:
        rate_hz = 1.0 / median if median > 0 else float("inf")
        warnings.append(
            f"Sampling rate appears to be ~{rate_hz:.2f} Hz (median "
            f"interval {median:.2f}s), but the model was trained on 1 Hz "
            f"(1.0s) samples. A 20-sample window will cover "
            f"{20 * median:.1f}s of real time instead of 20s, which can "
            f"degrade prediction quality. "
            f"How to fix: resample your data to 1 Hz before uploading "
            f"(in pandas: df.resample('1s').mean())."
        )

    # Inconsistent spacing → uneven windows.
    if std > 0.5 and median > 0:
        warnings.append(
            f"Sampling intervals are inconsistent (std {std:.2f}s on a "
            f"median of {median:.2f}s). Each 20-sample window will span "
            f"a different real-time duration, which the model wasn't "
            f"trained for. "
            f"How to fix: resample to a fixed 1 Hz cadence, or trim "
            f"sections where the logger paused."
        )

    if big_gaps > 0:
        warnings.append(
            f"Detected {big_gaps} gap(s) longer than 2 seconds in the "
            f"timestamps. Windows that straddle a gap won't reflect "
            f"continuous pump behaviour. "
            f"How to fix: either interpolate the missing samples, or "
            f"split the file at each gap and upload the segments "
            f"separately."
        )

    if duplicates > 0:
        warnings.append(
            f"Found {duplicates} row(s) with duplicate or out-of-order "
            f"timestamps. Sorting brought them into order but the values "
            f"may be unreliable. "
            f"How to fix: deduplicate by timestamp before uploading "
            f"(in pandas: df.drop_duplicates(subset='datetime'))."
        )

    return warnings


# ─── Sliding window construction (FR-10, NFR-05) ──────────────────────

def build_windows(
    df: pd.DataFrame,
    window_size: int | None = None,
) -> np.ndarray:
    """Build ``(N, window_size, 8)`` sliding windows with stride 1.

    Mirrors `build_samples_from_one_file` in
    `SKAB/Build_Dataset_SKAB_DLpipeline_By_David.py` exactly:

    - Sort rows chronologically by `settings.timestamp_column`.
    - Drop ``datetime``, ``anomaly`` and ``changepoint`` columns; keep
      only the 8 sensor columns in `settings.sensor_columns` order.
    - For ``i`` in ``[window_size - 1, total_len - horizon)``, take
      ``data[i - window_size + 1 : i + 1]`` as one window.

    Parameters
    ----------
    df
        Validated DataFrame (must have already passed `validate_schema`).
    window_size
        Override for `settings.window_size`. Use the default at inference
        time so we match training exactly.
    """
    w = window_size or settings.window_size
    horizon = settings.horizon

    df_sorted = df.sort_values(settings.timestamp_column).reset_index(drop=True)
    # Re-order columns to match the training scaler's feature order. The
    # scaler was fit on the columns in `settings.sensor_columns` order;
    # if we shuffle, every prediction is silently wrong.
    data = df_sorted[settings.sensor_columns].to_numpy(dtype=np.float64)

    total_len = len(df_sorted)
    if total_len < w + horizon:
        # Defensive — validate_schema should have caught this already.
        return np.empty((0, w, len(settings.sensor_columns)), dtype=np.float64)

    starts = np.arange(w - 1, total_len - horizon)
    # (N, window_size, n_features) via fancy indexing
    windows = np.stack([data[i - w + 1 : i + 1] for i in starts], axis=0)
    return windows


def build_labels(
    df: pd.DataFrame,
    window_size: int | None = None,
) -> np.ndarray | None:
    """Build the ``(N,)`` ground-truth labels for each window.

    Returns ``None`` when the DataFrame has no `anomaly` column
    (unlabelled-data mode, FR-21). The label rule matches training:
    ``y[i] = 1`` iff any of the next `horizon` rows after the window end
    has ``anomaly == 1``.
    """
    if settings.label_column not in df.columns:
        return None

    w = window_size or settings.window_size
    horizon = settings.horizon

    df_sorted = df.sort_values(settings.timestamp_column).reset_index(drop=True)
    anomaly = df_sorted[settings.label_column].to_numpy()
    total_len = len(df_sorted)
    if total_len < w + horizon:
        return np.empty((0,), dtype=np.int64)

    labels = np.fromiter(
        (
            1 if np.any(anomaly[i + 1 : i + 1 + horizon] == 1) else 0
            for i in range(w - 1, total_len - horizon)
        ),
        dtype=np.int64,
    )
    return labels
