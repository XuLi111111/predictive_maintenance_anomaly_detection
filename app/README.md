# pump.detect — Application Skeleton

CITS5206 Group 14 capstone. Three-tier scaffold (NFR-09): React frontend → FastAPI → model layer.

## Layout

```
backend/    FastAPI + inference layer (route stubs, registry, config)
frontend/   React + Vite + TS (Landing page implemented; Upload/Results stubs)
docker-compose.yml
```

## What is implemented

- Backend: project structure, settings, model registry with 8 entries, route
  files raising `501` so the API surface is visible in OpenAPI but inference
  logic is left to the teammate.
- Frontend: Vite + React + TS, design tokens carried 1:1 from
  `page/static landing page.html`, Landing page fully ported, Upload/Results
  routes mounted as placeholders.
- Tab navigation (Home / Upload / Results) per FR-06.
- Theme toggle with localStorage persistence (NFR-03).
- Docker Compose: nginx serves the SPA and reverse-proxies `/api/*` to uvicorn.

## What is NOT implemented (left for teammates)

- All `501` endpoints in `backend/app/api/routes/*` — see TODO comments.
- `backend/app/inference/preprocess.py` — `validate_schema`, `build_windows`.
- Loader for `model_transformer.pt` (artifact pending).
- Frontend Upload page (drop zone, model grid, perf panel, run button).
- Frontend Results page (probability chart, comparison grid, PDF export).
- ReportLab PDF templates.

## Run locally

Backend:
```bash
cd backend
pip install -r requirements.txt
# Copy the 7 sklearn artifacts + scaler.pkl into backend/artifacts/
uvicorn app.main:app --reload --port 8000
```

Frontend:
```bash
cd frontend
npm install
npm run dev
```
Vite proxies `/api/*` to `http://localhost:8000`.

## Run with Docker

```bash
docker compose up --build
# UI:  http://localhost:8080
# API: http://localhost:8080/api/health
```

## Model artifacts

Drop these into `backend/artifacts/` (the directory is volume-mounted read-only
by docker-compose):

```
scaler.pkl
model_lr.pkl  model_rf.pkl  model_svm.pkl  model_et.pkl
model_gb.pkl  model_knn.pkl model_xgb.pkl
model_transformer.pt   ← pending
```

The 7 sklearn artifacts already exist at
`data/processed/dataset2/skab_classical_models/`.
```
