"""End-to-end test for the WebSocket stream endpoint (T8).

Uses FastAPI's ``TestClient.websocket_connect`` so no real network or
browser is needed. We stage an upload by reusing the smoke_test helper
(direct CSV write into the upload_tmp_dir) so this test does not depend
on T1's HTTP route — that lets us catch regressions in T8 even if T1
breaks for unrelated reasons.
"""
from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path

import pytest

# Make `app.*` importable when this file is executed by pytest from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from app.core.config import settings
from app.inference.preprocess import validate_schema
from app.main import app

import pandas as pd


SAMPLE_CSV = Path(__file__).resolve().parents[1] / "artifacts" / "sample.csv"


@pytest.fixture(scope="module")
def staged_file_id() -> str:
    """Place a CSV + sidecar directly into upload_tmp_dir, return its file_id.

    Bypasses T1 so this test stays focused on T8 behaviour.
    """
    if not SAMPLE_CSV.is_file():
        pytest.skip(f"sample.csv missing from {SAMPLE_CSV}; cannot run stream test.")

    file_id = uuid.uuid4().hex
    csv_path = settings.upload_tmp_dir / f"{file_id}.csv"
    sidecar_path = settings.upload_tmp_dir / f"{file_id}.json"

    shutil.copy(SAMPLE_CSV, csv_path)

    df = pd.read_csv(csv_path, sep=";")
    info = validate_schema(df)
    sidecar_path.write_text(json.dumps({
        "filename": SAMPLE_CSV.name,
        "rows": info.rows,
        "has_label": info.has_label,
        "columns": info.columns_detected,
        "uploaded_at": "test",
    }), encoding="utf-8")

    yield file_id

    csv_path.unlink(missing_ok=True)
    sidecar_path.unlink(missing_ok=True)


def test_stream_happy_path(staged_file_id: str):
    """Connect, send start, drain ticks, expect a final end frame."""
    client = TestClient(app)
    with client.websocket_connect("/api/stream") as ws:
        ws.send_json({
            "type": "start",
            "file_id": staged_file_id,
            "model_id": "xgb",
            "threshold": 0.5,
            "speed": 1000.0,   # fast — don't actually sleep 21 seconds
        })

        ticks = 0
        end = None
        for _ in range(2000):
            msg = ws.receive_json()
            if msg["type"] == "tick":
                assert "idx" in msg and "prob" in msg and "status" in msg
                assert msg["status"] in ("NORMAL", "WATCH", "WARNING", "ALERT")
                # Sensors block carries the 8 SKAB channels at the
                # window's last sample (added so the chart tooltip can
                # show readings on hover).
                assert "sensors" in msg
                assert set(msg["sensors"].keys()) >= {
                    "Accelerometer1RMS", "Accelerometer2RMS", "Current",
                    "Pressure", "Temperature", "Thermocouple",
                    "Voltage", "Volume Flow RateRMS",
                }
                ticks += 1
            elif msg["type"] == "end":
                end = msg
                break
            elif msg["type"] == "error":
                pytest.fail(f"unexpected error frame: {msg}")

        assert end is not None, "stream never sent an 'end' frame"
        assert ticks > 0, "stream sent no 'tick' frames"
        assert end["summary"]["total_windows"] == ticks


def test_stream_unknown_model(staged_file_id: str):
    client = TestClient(app)
    with client.websocket_connect("/api/stream") as ws:
        ws.send_json({
            "type": "start",
            "file_id": staged_file_id,
            "model_id": "not-a-model",
            "threshold": 0.5,
            "speed": 1000.0,
        })
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert msg["code"] == "BAD_REQUEST"


def test_stream_unknown_file_id():
    client = TestClient(app)
    with client.websocket_connect("/api/stream") as ws:
        ws.send_json({
            "type": "start",
            "file_id": "deadbeef" * 4,
            "model_id": "xgb",
            "threshold": 0.5,
            "speed": 1000.0,
        })
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert msg["code"] == "PREPARE_FAILED"


def test_stream_bad_handshake_first_frame():
    client = TestClient(app)
    with client.websocket_connect("/api/stream") as ws:
        ws.send_json({"type": "tick", "idx": 0})  # wrong type
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert msg["code"] == "BAD_HANDSHAKE"
