# pump.detect — Application Skeleton

CITS5206 Group 14 capstone. Three-tier scaffold (NFR-09):
**React frontend → FastAPI → model layer.**

This `app/` directory is the deployable web application. The training
pipeline lives in the sibling `src/`, `data/`, and `models/` directories at
the repository root.

---

## Layout

```
app/
├── backend/                FastAPI + inference layer
│   ├── app/                source code
│   ├── artifacts/          trained model files (NOT in git — see section 5)
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/               React + Vite + TS
│   ├── src/                source code
│   ├── Dockerfile
│   ├── nginx.conf
│   └── package.json
├── docker-compose.yml
├── README.md               (this file)
└── CONTRIBUTING.md         task assignments T1~T7, branch workflow
```

---

## What is implemented

- **Backend** — project structure, settings module, model registry with 8
  entries, route files raising `501` so the API surface is visible in
  OpenAPI docs but inference logic is left to teammates (T1~T4, T7).
- **Frontend** — Vite + React + TypeScript, design tokens carried 1:1 from
  the original prototype (`/static landing page.html` at repo root).
  The **Landing page is fully implemented**; Upload and Results pages are
  mounted as placeholders pending T5 / T6.
- **Tab navigation** (Home / Upload / Results) per FR-06.
- **Theme toggle** with `localStorage` persistence (NFR-03).
- **Docker Compose** — nginx serves the SPA and reverse-proxies `/api/*`
  to uvicorn (NFR-10).

---

## What is NOT implemented (left for teammates)

See `CONTRIBUTING.md` section 4 for the full task table.

- All `501` endpoints in `backend/app/api/routes/` — `upload.py` (T1),
  `predict.py` (T2/T3), `report.py` (T4).
- `backend/app/inference/preprocess.py` — `validate_schema`,
  `build_windows` (T1).
- Loader for `model_transformer.pt` (artifact pending — T7).
- Frontend `Upload.tsx` — drop zone, model grid, perf panel, run button (T5).
- Frontend `Results.tsx` — probability chart, comparison grid,
  PDF export (T6).
- ReportLab PDF templates (T4).

---

## Run locally

### Backend
```bash
cd app/backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
# Copy the 7 sklearn artifacts + scaler.pkl into app/backend/artifacts/
uvicorn app.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

### Frontend
```bash
cd app/frontend
npm install
npm run dev
```

Frontend: http://localhost:5173 — Vite proxies `/api/*` to
`http://localhost:8000`.

### Full stack with Docker
```bash
cd app
docker compose up --build
# UI:  http://localhost:8080
# API: http://localhost:8080/api/health
```

---

## Model artifacts

Drop these into `app/backend/artifacts/` (volume-mounted read-only by
`docker-compose.yml`):

```
scaler.pkl
model_lr.pkl   model_rf.pkl   model_svm.pkl   model_et.pkl
model_gb.pkl   model_knn.pkl  model_xgb.pkl
model_transformer.pt          ← pending T7
```

The 7 sklearn artifacts already exist on David's local machine at:
```
data/processed/dataset2/skab_classical_models/
```

They are distributed via OneDrive (not in git — `.gitignore` excludes
`*.pkl`, `*.pt`, `*.npz`, `*.joblib`).

---

## For teammates implementing T5 / T6

The original interactive prototype is at the **repository root**:

```
/static landing page.html
```

Open it in a browser to see the target visual / interaction for the
Upload and Results pages. See `CONTRIBUTING.md` section 2 for the
HTML-to-React translation workflow. `app/frontend/src/pages/Landing.tsx`
is the reference implementation showing how to port the same prototype
to React.

---

## Architecture references

- **SRS**: `page/System Requirments .docx` (FR-01 ~ FR-21, NFR-01 ~ NFR-14)
- **Client meeting**: `page/Group14_Meeting_Minutes_WEEK9.docx`
- **Static prototype**: `/static landing page.html` (repo root)
- **Trained model baseline**: `SKAB/SKAB_ClassicalML_Baseline_By_David.py`
- **Data preprocessing pipeline**: `SKAB_Preprocessing_README.md` (repo root)
