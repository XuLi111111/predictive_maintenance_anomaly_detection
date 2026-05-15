"""Plumbing test for the transformer inference path (T7).

Does NOT verify Xu Li's actual model performance — just verifies that:
  1. The transformer scaler exists and has the right shape (8 features).
  2. A `model_transformer.pt` placed in artifacts/ is wired through the
     `is_dl=True` branch in `predict.run_model`.
  3. The branch applies per-feature scaling and produces (N,) probs.

We synthesise a fresh, untrained `TransformerFusionLiteModel` on the fly,
save it, and run inference. The probabilities are nonsense (random init)
but their shape and dtype prove the wiring is correct.
"""
from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings
from app.inference import loader, predict as inference
from app.inference.preprocess import validate_schema
from app.inference.registry import MODEL_REGISTRY


SAMPLE_CSV = Path(__file__).resolve().parents[1] / "artifacts" / "sample.csv"
TX_SCALER = Path(__file__).resolve().parents[1] / "artifacts" / "transformer_scaler.pkl"
TX_MODEL = Path(__file__).resolve().parents[1] / "artifacts" / "model_transformer.pt"


@pytest.fixture(scope="module")
def staged_file_id():
    if not SAMPLE_CSV.is_file():
        pytest.skip(f"sample.csv missing at {SAMPLE_CSV}")
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


def test_transformer_scaler_has_8_features():
    """The per-feature scaler must have shape (8,) — one per SKAB sensor."""
    if not TX_SCALER.is_file():
        pytest.skip(f"transformer_scaler.pkl missing at {TX_SCALER}; "
                    f"run scripts/extract_transformer_scaler.py first.")
    sc = joblib.load(TX_SCALER)
    assert sc.mean_.shape == (8,)
    assert sc.scale_.shape == (8,)


def test_transformer_endpoint_unavailable_when_pt_missing(staged_file_id):
    """Until Xu's .pt arrives, the transformer entry must report unavailable."""
    if TX_MODEL.is_file():
        pytest.skip("Real transformer artifact is present; this test only "
                    "covers the missing-artifact path.")
    prepared = inference.prepare_input(staged_file_id)
    out = inference.run_model("transformer", prepared, threshold=0.5)
    assert out.unavailable is True
    assert out.probs == []


def test_transformer_inference_with_dummy_model(staged_file_id, tmp_path, monkeypatch):
    """Wire a freshly-initialised TransformerFusionLite into the pipeline
    and check that inference returns one probability per window.

    We swap in a temp artifact file, clear the loader cache, and restore
    the original on teardown.
    """
    if not TX_SCALER.is_file():
        pytest.skip("transformer_scaler.pkl missing; run extract script.")

    import torch
    from app.inference.transformer_model import TransformerFusionLiteModel

    # Build an untrained model with the median sweep config from Xu's grid.
    torch.manual_seed(123)
    model = TransformerFusionLiteModel(
        input_dim=8, d_model=48, nhead=4, ff_dim=96, dropout=0.1, num_layers=2,
    )
    model.eval()

    target = settings.artifact_dir / "model_transformer.pt"
    backup = None
    if target.is_file():
        backup = target.with_suffix(".pt.bak")
        shutil.copy(target, backup)
    try:
        torch.save(model, target)
        # Clear loader cache so the new file is picked up.
        loader._model_cache.pop("transformer", None)

        prepared = inference.prepare_input(staged_file_id)
        out = inference.run_model("transformer", prepared, threshold=0.5)

        assert out.unavailable is False
        assert out.error is None
        assert out.total_windows > 0
        assert len(out.probs) == out.total_windows
        assert all(0.0 <= p <= 1.0 for p in out.probs)
    finally:
        target.unlink(missing_ok=True)
        if backup is not None:
            shutil.move(backup, target)
        loader._model_cache.pop("transformer", None)
