"""Live ingestion endpoints for the always-on pump monitoring page.

POST /api/live/ingest
    A pump simulator (or, in production, the SCADA bridge / Edge agent)
    pushes one sensor sample per call. The server appends it to the live
    buffer and — once enough samples are present — runs inference and
    broadcasts a tick to every connected WebSocket client.

WS /api/live/stream
    Browsers subscribe here. Each tick produced by the buffer is fanned
    out to every active connection. There's no per-client state; if a
    client disconnects we simply drop its queue.

GET /api/live/status
    Diagnostic snapshot — how many samples are buffered, how many
    subscribers, what state the machine is in. Useful when the page shows
    "waiting for data" and the operator wants to know if the simulator
    is even running.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import settings
from app.inference.live_buffer import live_buffer

log = logging.getLogger(__name__)
router = APIRouter()


class Sample(BaseModel):
    """One pump reading.

    `sensors` must contain all 8 SKAB channels by name. Extra keys are
    ignored, missing keys cause the inference step to skip this window
    (but the sample is still buffered for diagnostics).
    """

    timestamp: str = Field(..., description="ISO timestamp of this reading")
    sensors: Annotated[
        dict[str, float],
        Field(..., description="Sensor channel → reading, e.g. {'Current': 0.221}"),
    ]


@router.post("/live/ingest")
async def ingest(sample: Sample) -> dict:
    tick = await live_buffer.push(sample.model_dump())
    return {
        "ok": True,
        "tick_emitted": tick is not None,
        "samples_in_buffer": len(live_buffer.samples),
    }


@router.get("/live/status")
def status() -> dict:
    snap = live_buffer.snapshot_status()
    snap["expected_columns"] = settings.sensor_columns
    return snap


class ConfigPatch(BaseModel):
    """Runtime config patch. Fields are optional — only the ones the user
    actually touched on the Live page get sent."""
    # Allow `model_id` despite Pydantic's reserved `model_` namespace
    # (same workaround as ReportRequest / PredictRequest).
    model_config = ConfigDict(protected_namespaces=())

    model_id: str | None = Field(default=None)
    threshold: float | None = Field(default=None)


@router.post("/live/config")
async def update_config(patch: ConfigPatch) -> dict:
    try:
        snap = await live_buffer.set_config(
            model_id=patch.model_id,
            threshold=patch.threshold,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    log.info("live config updated: %s", snap)
    return {"ok": True, **snap}


@router.websocket("/live/stream")
async def stream(ws: WebSocket) -> None:
    """Push ticks to the browser; clean up the subscription when the
    client disconnects.

    We race two tasks:
      * push_loop — pumps the subscriber's queue out to the socket
      * recv_loop — reads (and discards) inbound frames purely to detect
        the disconnect.

    Whichever finishes first cancels the other and we unsubscribe. This
    prevents the classic leak where ``await q.get()`` would hang forever
    after a client closed its tab.
    """
    await ws.accept()
    q = live_buffer.subscribe()
    log.info("live WS connected; subscribers=%d", len(live_buffer.subscribers))

    async def push_loop() -> None:
        # Immediate hello so the client knows the socket is up.
        await ws.send_json({
            "type": "hello",
            "samples_in_buffer": len(live_buffer.samples),
            "window_size": live_buffer.window_size,
            "current_status": live_buffer.prev_status,
        })
        while True:
            tick = await q.get()
            await ws.send_json(tick)

    async def recv_loop() -> None:
        # No protocol messages expected; receive_text just blocks until
        # the client disconnects, at which point starlette raises.
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            return

    push_task = asyncio.create_task(push_loop())
    recv_task = asyncio.create_task(recv_loop())

    try:
        _done, pending = await asyncio.wait(
            {push_task, recv_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()
        # Drain cancellations so the event loop doesn't warn about
        # un-awaited tasks.
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        for t in _done:
            try:
                exc = t.exception()
            except asyncio.CancelledError:
                continue
            if exc is not None and not isinstance(exc, WebSocketDisconnect):
                log.warning("live WS task ended with exception: %r", exc)
    finally:
        live_buffer.unsubscribe(q)
        log.info("live WS closed; subscribers=%d", len(live_buffer.subscribers))
