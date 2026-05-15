"""In-memory ring buffer for the Live Pump page.

The buffer accepts one sensor sample per push (8 channels + timestamp),
and once it has accumulated `window_size` samples it runs the transformer
inference once and broadcasts a tick to every connected WebSocket
subscriber. Single source of truth for live state, lives in the worker
process — sufficient for a single-instance demo deployment.

Concurrency model: a single asyncio.Lock guards the buffer + inference
path so we never double-broadcast or read mid-mutation. Subscribers are
asyncio.Queues with a small max size; if a client falls behind we drop
ticks rather than blocking the producer.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any

import numpy as np

from app.core.config import settings
from app.inference import loader, state_machine
from app.inference.predict import _run_sklearn, _run_torch

log = logging.getLogger(__name__)


# How many ticks each subscriber can buffer before we start dropping.
# Slow clients (e.g. a paused browser tab) won't back-pressure the
# inference loop — they just miss ticks.
_SUBSCRIBER_QUEUE_MAX = 100


# ── Data-quality monitoring constants ─────────────────────────────────
#
# These checks watch the *acquisition pipeline* (PLC / SCADA / simulator),
# not the pump itself. They flag situations the model can't detect:
#   - sensor frozen at one value (cable disconnected / probe failure)
#   - readings outside any physically reasonable range (unit mistake,
#     wiring error, or genuine hardware fault)
#   - the ingest endpoint receiving samples at uneven cadence (collector
#     on the client side is misbehaving)
# Stale-data detection (no samples for several seconds) is handled
# *client-side* — easier, and the server has no way to "push" a stale
# warning anyway when nothing is ticking.

# Per-channel sanity range. Beyond these limits we treat the reading
# as almost-certainly a sensor / unit / wiring problem rather than a
# real anomaly. Tuned to be generous so we don't false-positive on a
# real fault — the goal is "this can't be the pump talking".
_SANITY_RANGES: dict[str, tuple[float, float]] = {
    "Accelerometer1RMS":   (-1.0,  10.0),
    "Accelerometer2RMS":   (-1.0,  10.0),
    "Current":             (-10.0, 100.0),
    "Pressure":            (-10.0, 10.0),
    "Temperature":         (-50.0, 200.0),
    "Thermocouple":        (-50.0, 200.0),
    "Voltage":             (0.0,   500.0),
    "Volume Flow RateRMS": (-10.0, 500.0),
}

# A channel is considered frozen when N consecutive samples have the
# exact same value. 30 = 30 seconds at 1 Hz — way longer than any
# legitimate hold time.
_FROZEN_LOOKBACK = 30

# How many recent inter-sample intervals to keep for cadence stats.
_INTERVAL_LOOKBACK = 20
_UNEVEN_STD_THRESHOLD = 0.5  # seconds


class LiveBuffer:
    """Accumulate samples, run inference on full windows, fan out to WS."""

    def __init__(self, window_size: int, history_cap: int = 600) -> None:
        self.window_size = window_size
        self.samples: deque[dict] = deque(maxlen=history_cap)
        self.subscribers: set[asyncio.Queue] = set()
        self.lock = asyncio.Lock()

        # State machine context — kept across ticks so promotion/demotion
        # hysteresis works correctly (matches the CSV-replay state machine).
        self.recent_probs: deque[float] = deque(maxlen=20)
        self.prev_status: str = "NORMAL"
        self.tick_idx: int = 0

        # Default to the transformer (best F1 = 0.9244). If it's unavailable
        # for any reason we'll fall back to xgb on first inference.
        self.model_id: str = "transformer"

        # Alert threshold used to compute the `in_anomaly_window` flag on
        # each tick. Stats and the chart's reference line both read this.
        # NB: this does NOT replace the 4-tier state machine thresholds —
        # those are domain-tuned tier breakpoints (0.3 / 0.5 / 0.8) and are
        # independent of the user's binary alert threshold.
        self.threshold: float = float(settings.alert_threshold)

        # ── Data-quality monitoring state ───────────────────────────────
        # Last 30 readings of each channel — used to detect "frozen
        # sensor" (all identical).
        self.recent_values: dict[str, deque[float]] = {
            col: deque(maxlen=_FROZEN_LOOKBACK)
            for col in settings.sensor_columns
        }
        # Last 20 inter-arrival intervals (seconds).
        self.recent_intervals: deque[float] = deque(maxlen=_INTERVAL_LOOKBACK)
        self.last_push_time: float | None = None
        # Set of currently-active quality issue keys (e.g. "frozen:Current").
        # Used so we only broadcast a new warning when state CHANGES — no
        # spam.
        self.active_quality: set[str] = set()

    # ── Producer side (called from POST /api/live/ingest) ───────────────

    async def push(self, sample: dict) -> dict | None:
        """Append a sample, run inference if a full window is available,
        and check data-quality. Returns the tick if one was produced,
        else None (still warming up).

        Quality frames are broadcast independently of ticks — they fire
        the moment a transition is detected, even before the buffer is
        full enough to score."""
        async with self.lock:
            now = time.monotonic()
            self._update_quality_metrics(sample, now)
            self.last_push_time = now
            self.samples.append(sample)
            tick = self._maybe_infer()
            quality_frames = self._check_data_quality()

        if tick is not None:
            await self._broadcast(tick)
        for frame in quality_frames:
            await self._broadcast(frame)
        return tick

    def _update_quality_metrics(self, sample: dict, now: float) -> None:
        """Update the per-channel ring + the interval ring. Pure data
        bookkeeping, no warning logic — that's `_check_data_quality`."""
        if self.last_push_time is not None:
            dt = now - self.last_push_time
            if dt > 0:
                self.recent_intervals.append(dt)
        sensors = sample.get("sensors") or {}
        for col in settings.sensor_columns:
            if col in sensors:
                try:
                    self.recent_values[col].append(float(sensors[col]))
                except (TypeError, ValueError):
                    pass

    def _check_data_quality(self) -> list[dict]:
        """Compare the current quality state to what was last broadcast
        and return frames for any new / cleared issues."""
        frames: list[dict] = []
        new_active: set[str] = set()

        # 1. Frozen channels.
        for col in settings.sensor_columns:
            vals = self.recent_values[col]
            if len(vals) >= _FROZEN_LOOKBACK and len(set(vals)) == 1:
                key = f"frozen:{col}"
                new_active.add(key)
                if key not in self.active_quality:
                    frames.append({
                        "type": "quality",
                        "code": "FROZEN_SENSOR",
                        "channel": col,
                        "severity": "warning",
                        "message": (
                            f"Channel '{col}' has reported the exact same "
                            f"value for {_FROZEN_LOOKBACK} consecutive samples. "
                            f"The sensor may be disconnected or stuck."
                        ),
                    })

        # 2. Out-of-range (likely sensor / unit / wiring fault).
        latest = self.samples[-1].get("sensors", {}) if self.samples else {}
        for col, (lo, hi) in _SANITY_RANGES.items():
            if col not in latest:
                continue
            try:
                v = float(latest[col])
            except (TypeError, ValueError):
                continue
            if v < lo or v > hi:
                key = f"range:{col}"
                new_active.add(key)
                if key not in self.active_quality:
                    frames.append({
                        "type": "quality",
                        "code": "OUT_OF_RANGE",
                        "channel": col,
                        "severity": "warning",
                        "message": (
                            f"Channel '{col}' reads {v:.3f}, far outside "
                            f"its expected physical range "
                            f"[{lo:.1f}, {hi:.1f}]. Check the sensor, "
                            f"its unit of measure, or its wiring."
                        ),
                    })

        # 3. Uneven sample cadence.
        if len(self.recent_intervals) >= _INTERVAL_LOOKBACK:
            arr = np.asarray(self.recent_intervals, dtype=float)
            std = float(arr.std())
            if std > _UNEVEN_STD_THRESHOLD:
                key = "uneven:ingest"
                new_active.add(key)
                if key not in self.active_quality:
                    frames.append({
                        "type": "quality",
                        "code": "UNEVEN_SAMPLING",
                        "channel": None,
                        "severity": "warning",
                        "message": (
                            f"Sample arrival cadence is unstable "
                            f"(std {std:.2f}s over the last "
                            f"{_INTERVAL_LOOKBACK} samples). The "
                            f"collector / network may be dropping frames; "
                            f"each 20-sample window will span an uneven "
                            f"real-time duration."
                        ),
                    })

        # Emit "cleared" frames for issues that were active but aren't anymore.
        for key in self.active_quality - new_active:
            kind, _, target = key.partition(":")
            if kind == "frozen":
                frames.append({
                    "type": "quality",
                    "code": "FROZEN_CLEARED",
                    "channel": target,
                    "severity": "info",
                    "message": f"Channel '{target}' is reporting varying values again.",
                })
            elif kind == "range":
                frames.append({
                    "type": "quality",
                    "code": "OUT_OF_RANGE_CLEARED",
                    "channel": target,
                    "severity": "info",
                    "message": f"Channel '{target}' is back within its expected range.",
                })
            elif kind == "uneven":
                frames.append({
                    "type": "quality",
                    "code": "EVEN_SAMPLING_RESTORED",
                    "channel": None,
                    "severity": "info",
                    "message": "Sample cadence is stable again.",
                })

        self.active_quality = new_active
        return frames

    async def set_config(
        self,
        *,
        model_id: str | None = None,
        threshold: float | None = None,
    ) -> dict:
        """Apply runtime configuration changes from the Live page.

        Switching the model resets the recent-prob ring and state-machine
        cursor, because different models have different probability
        distributions and we don't want the *new* model to inherit the
        *old* model's hysteresis state. Threshold changes don't need a
        reset — they only affect downstream display.

        Raises ValueError on invalid inputs so the route can return 400.
        """
        # Imported here to avoid module-import cycles with registry.
        from app.inference.registry import MODEL_REGISTRY

        async with self.lock:
            if model_id is not None:
                if model_id not in MODEL_REGISTRY:
                    raise ValueError(
                        f"Unknown model_id '{model_id}'. "
                        f"Valid: {sorted(MODEL_REGISTRY)}",
                    )
                if model_id != self.model_id:
                    self.model_id = model_id
                    self.recent_probs.clear()
                    self.prev_status = "NORMAL"

            if threshold is not None:
                if not 0.0 < threshold < 1.0:
                    raise ValueError(
                        f"threshold must be in (0, 1), got {threshold}",
                    )
                self.threshold = float(threshold)

            snap = {
                "model_id": self.model_id,
                "threshold": self.threshold,
            }

        # Notify any open dashboards so multiple browsers stay in sync.
        await self._broadcast({
            "type": "config",
            "model_id": snap["model_id"],
            "threshold": snap["threshold"],
        })
        return snap

    # ── Subscriber side (called from WS /api/live/stream) ───────────────

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=_SUBSCRIBER_QUEUE_MAX)
        self.subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self.subscribers.discard(q)

    def snapshot_status(self) -> dict:
        """Read-only view of the buffer for status endpoints / tests."""
        return {
            "samples_in_buffer": len(self.samples),
            "window_size": self.window_size,
            "subscribers": len(self.subscribers),
            "current_status": self.prev_status,
            "ticks_emitted": self.tick_idx,
            "model_id": self.model_id,
            "threshold": self.threshold,
            "active_quality_issues": sorted(self.active_quality),
        }

    # ── Internals ───────────────────────────────────────────────────────

    def _maybe_infer(self) -> dict | None:
        if len(self.samples) < self.window_size:
            return None

        rows = list(self.samples)[-self.window_size:]
        try:
            feats = np.array(
                [[r["sensors"][col] for col in settings.sensor_columns]
                 for r in rows],
                dtype=np.float64,
            )
        except KeyError as exc:
            log.warning("missing sensor column in sample: %s", exc)
            return None

        try:
            prob = self._infer_one(feats)
        except Exception as exc:  # noqa: BLE001
            log.exception("inference failed: %s", exc)
            return None

        self.recent_probs.append(prob)
        new_status = state_machine.next_state(
            self.prev_status, list(self.recent_probs),
        )
        self.prev_status = new_status
        self.tick_idx += 1

        # Attach the sensor readings of the window's most recent sample
        # so the chart tooltip can show "what was the pump doing at this
        # instant?" alongside the prediction.
        latest_sensors = rows[-1].get("sensors") or {}
        return {
            "type": "tick",
            "idx": self.tick_idx,
            "prob": prob,
            "status": new_status,
            "timestamp": rows[-1].get("timestamp"),
            "in_anomaly_window": prob >= self.threshold,
            "sensors": {
                col: float(latest_sensors.get(col, 0.0))
                for col in settings.sensor_columns
            },
        }

    def _infer_one(self, feats_2d: np.ndarray) -> float:
        """Run inference on a single (window_size, 8) feature matrix.

        Picks the model by `self.model_id`. Falls back to xgb if the
        transformer artifact is unavailable.
        """
        try:
            model = loader.load_model(self.model_id)
            meta = loader.get_model_meta(self.model_id)
        except Exception:  # noqa: BLE001 -- including HTTPException 503
            if self.model_id != "xgb":
                log.warning("%s unavailable; falling back to xgb", self.model_id)
                self.model_id = "xgb"
                model = loader.load_model("xgb")
                meta = loader.get_model_meta("xgb")
            else:
                raise

        if meta.is_dl:
            tx_scaler = loader.load_transformer_scaler()
            c = len(settings.sensor_columns)
            scaled = tx_scaler.transform(feats_2d.reshape(-1, c)).reshape(
                1, self.window_size, c,
            )
            probs = _run_torch(model, scaled)
        else:
            flat = feats_2d.reshape(1, -1)
            if meta.requires_scaling:
                scaler = loader.load_scaler()
                flat = scaler.transform(flat)
            probs = _run_sklearn(model, flat)

        return float(probs[0])

    async def _broadcast(self, tick: dict) -> None:
        for q in list(self.subscribers):
            try:
                q.put_nowait(tick)
            except asyncio.QueueFull:
                # Slow subscriber — skip rather than block the producer.
                log.debug("dropped tick for slow subscriber")


# Module-level singleton consumed by the route layer.
live_buffer = LiveBuffer(window_size=settings.window_size)


__all__ = ["LiveBuffer", "live_buffer"]
