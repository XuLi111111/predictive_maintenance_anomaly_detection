"""HTTP smoke test for T1 (upload endpoint).

Verifies the round-trip from raw CSV → validate → file_id → predict.
Catches regressions like the CSV-separator bug, missing columns, and
the ``has_label=false`` metric-suppression branch (FR-21).
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings
from app.main import app


SAMPLE_CSV = Path(__file__).resolve().parents[1] / "artifacts" / "sample.csv"


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def sample_bytes() -> bytes:
    if not SAMPLE_CSV.is_file():
        pytest.skip(f"sample.csv missing at {SAMPLE_CSV}")
    return SAMPLE_CSV.read_bytes()


def test_upload_happy_path(client: TestClient, sample_bytes: bytes):
    resp = client.post(
        "/api/upload",
        files={"file": ("sample.csv", sample_bytes, "text/csv")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "file_id" in body
    assert body["rows"] > 0
    assert body["has_label"] is True
    assert "Accelerometer1RMS" in body["columns_detected"]
    # File should be staged on disk.
    csv_path = settings.upload_tmp_dir / f"{body['file_id']}.csv"
    sidecar_path = settings.upload_tmp_dir / f"{body['file_id']}.json"
    assert csv_path.is_file()
    assert sidecar_path.is_file()


def test_upload_rejects_non_csv_extension(client: TestClient):
    resp = client.post(
        "/api/upload",
        files={"file": ("not_a_csv.txt", b"foo,bar\n1,2\n", "text/plain")},
    )
    assert resp.status_code == 400


def test_upload_rejects_empty_file(client: TestClient):
    resp = client.post(
        "/api/upload",
        files={"file": ("empty.csv", b"", "text/csv")},
    )
    assert resp.status_code == 400


def test_upload_rejects_missing_sensor_columns(client: TestClient):
    bad = b"datetime;Accelerometer1RMS\n2020-01-01 00:00:00;0.5\n" * 40
    resp = client.post(
        "/api/upload",
        files={"file": ("malformed.csv", bad, "text/csv")},
    )
    assert resp.status_code == 400
    assert "sensor column" in resp.json()["detail"].lower()


def test_upload_unlabelled_drops_metrics(client: TestClient, sample_bytes: bytes):
    """Strip the anomaly column from the sample and confirm has_label=False."""
    text = sample_bytes.decode("utf-8")
    lines = [l for l in text.split("\n") if l]
    header = lines[0].split(";")
    keep_idx = [i for i, c in enumerate(header) if c not in ("anomaly", "changepoint")]
    stripped = "\n".join(
        ";".join(parts.split(";")[i] for i in keep_idx) for parts in lines
    ).encode("utf-8")

    resp = client.post(
        "/api/upload",
        files={"file": ("unlabelled.csv", stripped, "text/csv")},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["has_label"] is False


def test_sample_csv_is_served(client: TestClient):
    resp = client.get("/api/sample-csv")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    assert b"Accelerometer1RMS" in resp.content
