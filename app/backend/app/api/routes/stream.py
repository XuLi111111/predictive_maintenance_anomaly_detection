"""WebSocket replay endpoint for the Live Monitor (T8).

`WS /api/stream` accepts an uploaded `file_id` plus a model id and replays
the resulting per-window probabilities back to the client at a configurable
speed. The client controls pause/resume/speed/stop mid-stream; the server
computes the four-level alert state via `inference.state_machine`.

The architecture is deliberately stateless: all heavy work
(`prepare_input` + `run_model`) happens once up-front, and the loop just
streams pre-computed probabilities. That keeps CPU low even at 100x speed
and lets us swap the data source for live MQTT later without touching
this loop.
"""
from __future__ import annotations

import asyncio
import json
from collections import deque
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool

from app.core.config import settings
from app.inference import predict as inference
from app.inference.registry import MODEL_REGISTRY
from app.inference.state_machine import States, next_state

router = APIRouter()


# Streamer state lives entirely on the WebSocket — no globals.
class _Stream:
    __slots__ = ("paused", "stopped", "speed")

    def __init__(self, speed: float) -> None:
        self.paused = False
        self.stopped = False
        self.speed = max(0.1, min(speed, 1000.0))


@router.websocket("/stream")
async def stream(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        await _run(websocket)
    except WebSocketDisconnect:
        # Client closed the tab — nothing to do.
        return
    except Exception as exc:  # noqa: BLE001
        # Last-resort guard so we don't leak stack traces.
        await _safe_send(websocket, {
            "type": "error",
            "code": "INTERNAL_ERROR",
            "message": str(exc),
        })
    finally:
        await _safe_close(websocket)


# ─── Main loop ────────────────────────────────────────────────────────

async def _run(websocket: WebSocket) -> None:
    # 1. Wait for the start frame.
    cfg = await _recv_json(websocket, expect_type="start")
    if cfg is None:
        return

    file_id = cfg.get("file_id")
    model_id = cfg.get("model_id", "xgb")
    threshold = float(cfg.get("threshold", settings.alert_threshold))
    speed = float(cfg.get("speed", 1.0))

    if not file_id or model_id not in MODEL_REGISTRY:
        await websocket.send_json({
            "type": "error",
            "code": "BAD_REQUEST",
            "message": "Missing file_id or unknown model_id.",
        })
        return

    # 2. Prepare inputs + run the model once. Both are CPU-bound — push
    #    them off the event loop so other connections aren't blocked.
    try:
        prepared = await run_in_threadpool(inference.prepare_input, file_id)
        output = await run_in_threadpool(
            inference.run_model, model_id, prepared, threshold,
        )
    except Exception as exc:  # noqa: BLE001
        await websocket.send_json({
            "type": "error",
            "code": "PREPARE_FAILED",
            "message": str(exc),
        })
        return

    if output.unavailable:
        await websocket.send_json({
            "type": "error",
            "code": "MODEL_UNAVAILABLE",
            "message": output.error or f"Model '{model_id}' artifact missing.",
        })
        return

    # 3. Replay loop. The inner control coroutine reads frames non-blockingly.
    state = _Stream(speed)
    asyncio.create_task(_control_loop(websocket, state))

    cur_state = "NORMAL"
    history: deque[float] = deque(maxlen=10)
    base_ts = _base_timestamp(prepared)

    # Build a (N_rows, 8) numpy view of the *sorted* sensor matrix so each
    # tick can carry the 8 channel readings at that window's last sample.
    # Sorting + column selection must mirror build_windows() exactly so
    # idx → row mapping stays consistent (preprocess.py is the source of
    # truth for the windowing convention).
    df_sorted = (
        prepared.df.sort_values(settings.timestamp_column).reset_index(drop=True)
    )
    sensor_cols = settings.sensor_columns
    sensor_matrix = df_sorted[sensor_cols].to_numpy(dtype=float)
    window_size = settings.window_size

    for idx, prob in enumerate(output.probs):
        if state.stopped:
            break
        # Honour pause without busy-waiting.
        while state.paused and not state.stopped:
            await asyncio.sleep(0.05)
        if state.stopped:
            break

        history.append(float(prob))
        cur_state = next_state(cur_state, list(history))

        # window k spans rows [k, k + window_size). Its last sample is at
        # row index `window_size - 1 + k` in the sorted dataframe.
        last_row = window_size - 1 + idx
        window_sensors = {
            col: float(sensor_matrix[last_row, ci])
            for ci, col in enumerate(sensor_cols)
        }

        await websocket.send_json({
            "type": "tick",
            "idx": idx,
            "prob": float(prob),
            "status": cur_state,
            "timestamp": _ts_iso(base_ts, idx),
            "in_anomaly_window": cur_state in ("WARNING", "ALERT"),
            "sensors": window_sensors,
        })

        await asyncio.sleep(1.0 / state.speed)

    # 4. End frame with summary (matches batch ModelResult fields).
    await websocket.send_json({
        "type": "end",
        "summary": {
            "total_windows": output.total_windows,
            "fault_windows": output.fault_windows,
            "peak_idx": output.peak_idx,
            "peak_prob": output.peak_prob,
            "metrics": output.metrics,
        },
    })


# ─── Control side-channel ─────────────────────────────────────────────

async def _control_loop(websocket: WebSocket, state: _Stream) -> None:
    """Read pause/resume/set_speed/stop frames concurrently with the tick loop."""
    try:
        while not state.stopped:
            msg = await websocket.receive_text()
            try:
                payload = json.loads(msg)
            except json.JSONDecodeError:
                continue
            mtype = payload.get("type")
            if mtype == "pause":
                state.paused = True
            elif mtype == "resume":
                state.paused = False
            elif mtype == "set_speed":
                try:
                    state.speed = max(0.1, min(float(payload.get("speed", 1.0)), 1000.0))
                except (TypeError, ValueError):
                    pass
            elif mtype == "stop":
                state.stopped = True
                return
    except WebSocketDisconnect:
        state.stopped = True


# ─── Helpers ──────────────────────────────────────────────────────────

async def _recv_json(websocket: WebSocket, *, expect_type: str) -> dict[str, Any] | None:
    """Receive the first frame and validate its type."""
    try:
        raw = await websocket.receive_text()
    except WebSocketDisconnect:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        await websocket.send_json({
            "type": "error",
            "code": "BAD_JSON",
            "message": "First frame must be valid JSON.",
        })
        return None
    if payload.get("type") != expect_type:
        await websocket.send_json({
            "type": "error",
            "code": "BAD_HANDSHAKE",
            "message": f"Expected first frame type='{expect_type}'.",
        })
        return None
    return payload


def _base_timestamp(prepared: inference.PreparedInput) -> datetime:
    """Use the upload's first timestamp if parseable, else 'now'."""
    try:
        ts_col = settings.timestamp_column
        first = prepared.df[ts_col].iloc[0]
        return datetime.fromisoformat(str(first))
    except Exception:  # noqa: BLE001
        return datetime.now()


def _ts_iso(base: datetime, idx: int) -> str:
    """Each window step is 1 second of pump data (SKAB sampling rate)."""
    return (base + timedelta(seconds=idx)).isoformat()


async def _safe_send(websocket: WebSocket, payload: dict[str, Any]) -> None:
    try:
        await websocket.send_json(payload)
    except Exception:  # noqa: BLE001
        pass


async def _safe_close(websocket: WebSocket) -> None:
    try:
        await websocket.close()
    except Exception:  # noqa: BLE001
        pass


__all__ = ["router"]
