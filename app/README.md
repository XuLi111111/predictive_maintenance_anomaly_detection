# pump.detect — Application

CITS5206 Group 14 capstone. Three-tier deployment of a pump anomaly
early-warning system on top of David's SKAB training pipeline.

```
React frontend  ←→  FastAPI backend  ←→  Trained model artifacts
```

This `app/` directory is the deployable web application. The training
pipeline lives in the sibling `src/`, `data/`, and `SKAB/` directories
at the repository root.

> See `CONTRIBUTING.md` for the per-module status table and contributor
> conventions.

---

## Quick reference — what's where

```
app/
├── backend/             FastAPI + inference + PDF reports
│   ├── app/             source code
│   ├── artifacts/       trained model files (NOT in git)
│   ├── tests/           21 pytest cases
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/            React + Vite + TypeScript
│   ├── src/             pages, components, hooks, api, styles
│   ├── Dockerfile
│   ├── nginx.conf
│   └── package.json
├── scripts/
│   └── pump_simulator.py   Streams realistic SKAB-style data into /api/live/ingest
├── docker-compose.yml
├── README.md            (this file)
└── CONTRIBUTING.md      module status, conventions, layout
```

---

## What the app does

Two complementary workflows on the same underlying inference pipeline:

### 1. Upload → inline replay → PDF report  (batch / analyst use case)
- User uploads a SKAB-format CSV at `/upload`
- Schema is validated server-side with plain-language "How to fix" errors
- User picks 1 of 8 pre-trained models + an alert threshold
- "▶ Start replay" streams the file window-by-window in the **same page**
  (no jump): chart + status banner + 4-stat panel + Pause / Stop /
  speed control + Brush slider to scrub back through history
- When the replay finishes, "↓ Download PDF report" produces a 5-page
  report (Executive Summary with risk badge, anomaly timeline, 8-model
  comparison, confusion matrix + ROC/PR when labelled, sensor deep-dive
  with z-score narrative, methodology + provenance appendix)

### 2. Live monitor  (real-time / operator use case)
- `/live` is always-on. Subscribes to a WebSocket and renders ticks as
  they arrive.
- Data comes from `app/scripts/pump_simulator.py` (or in production, a
  SCADA bridge that POSTs to `/api/live/ingest`).
- Top control bar: status badge + model dropdown + threshold slider +
  Pause / Clear. Model and threshold changes apply runtime — no
  restart.
- Data-quality banner surfaces FROZEN_SENSOR / OUT_OF_RANGE /
  UNEVEN_SAMPLING from the server, plus a client-side STALE_DATA
  watchdog when no ticks arrive for 5 s.

Both workflows use the same per-window stateless inference code. The
"swap source from CSV to PLC and nothing else changes" property of the
pipeline is real, not a slogan.

---

## Run with Docker (the recommended way)

Prerequisites:
- Docker Desktop 4.x or compatible engine
- Trained model artifacts in `app/backend/artifacts/` (see below)

```bash
cd app
docker compose up --build
```

- **UI**: http://localhost:8080
- **API health**: http://localhost:8080/api/health
- **Swagger**: http://localhost:8080/api/docs  (auto-generated from FastAPI)

To exercise the Live page, in a separate terminal:

```bash
# From the repo root, with the project venv activated
python app/scripts/pump_simulator.py
```

The simulator pushes one sample per second to `/api/live/ingest` with
random anomaly bursts (60–180 s apart, 15–40 samples long, random
intensity). Pass `--no-burst` to stream only normal data, `--seed 42` to
get a deterministic run.

---

## Run locally without Docker

### Backend
```bash
cd app/backend
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
# Copy trained artifacts into app/backend/artifacts/  (see below)

uvicorn app.main:app --reload --port 8000
```

- API: http://localhost:8000
- Swagger: http://localhost:8000/docs

### Frontend
```bash
cd app/frontend
npm install
npm run dev
```

Vite serves the UI at http://localhost:5173 and proxies `/api/*` and
WebSockets to `http://localhost:8000`.

### Tests
```bash
cd app/backend
pytest tests/                    # 21 pass + 1 skip
```

The skipped test (`test_transformer_inference.py`) exercises a code
path that writes to `artifacts/` — fine on bare-metal, fails inside
Docker where the directory is mounted read-only.

---

## Model artifacts (not in git)

Drop these into `app/backend/artifacts/`:

```
scaler.pkl                       (per-(timestep × feature) for classical models)
transformer_scaler.pkl           (per-feature for transformer only)
sample.csv                       (50-row SKAB slice, served by /api/sample-csv)
model_lr.pkl       model_rf.pkl  model_svm.pkl   model_et.pkl
model_gb.pkl       model_knn.pkl model_xgb.pkl
model_transformer.pt             (TransformerFusionLite — F1 = 0.9244)
transformer_threshold.json       {"threshold": 0.59, "source_model": "..."}
```

`.gitignore` excludes `*.pkl`, `*.pt`, `*.npz`, `*.joblib` so they can
never be committed by accident.

The 7 sklearn artifacts are produced by
`SKAB/SKAB_ClassicalML_Baseline_By_David.py`. The transformer
artifacts come from `src/deep_learning/SKAB_TransformerFusionLite_TrainingSearch_ByXuLi.py`
followed by `src/deep_learning/convert_xu_transformer.py`.

---

## Useful API endpoints (quick reference)

| Method | Path | Purpose |
|--------|------|---------|
| GET    | `/api/health`              | liveness check |
| GET    | `/api/sample-csv`          | download SKAB sample |
| GET    | `/api/models`              | 8 models metadata for the picker |
| POST   | `/api/upload`              | validate + persist a CSV; returns `file_id` + warnings |
| POST   | `/api/predict`             | single-model prediction on a `file_id` |
| POST   | `/api/predict/compare`     | all 8 models on the same `file_id` |
| POST   | `/api/report`              | render the 5-page PDF |
| WS     | `/api/stream`              | CSV-replay with pause / stop / speed control |
| POST   | `/api/live/ingest`         | push one sensor sample (called by simulator / PLC bridge) |
| POST   | `/api/live/config`         | runtime-switch model and / or threshold |
| GET    | `/api/live/status`         | snapshot: buffer size, active model, threshold, quality issues |
| WS     | `/api/live/stream`         | subscribe to live ticks + config + quality frames |

Auto-generated full reference at `/api/docs` when the backend is running.

---

## Architecture references

- **SRS**: `/System Requirments .pdf` at repo root (FR-01 – FR-21,
  NFR-01 – NFR-14)
- **Static prototype** (historical visual reference): `/static landing page.html`
- **Trained model baseline**: `SKAB/SKAB_ClassicalML_Baseline_By_David.py`
- **Transformer training**: `src/deep_learning/SKAB_TransformerFusionLite_TrainingSearch_ByXuLi.py`
- **Data preprocessing pipeline**: top-level `SKAB_Preprocessing_README.md`
