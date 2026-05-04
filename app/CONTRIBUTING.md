# Contributing — pump.detect

Welcome. This document tells you what to build, where to put it, and how to
ship it without stepping on anyone's toes.

---

## 1. Repository layout

```
capstone/
├── backend/                  FastAPI + inference layer
│   ├── app/
│   │   ├── main.py           Entry point — DO NOT modify casually
│   │   ├── core/config.py    Shared settings (sensor cols, threshold, paths)
│   │   ├── api/routes/       ← your work goes here
│   │   │   ├── upload.py     stub (501)
│   │   │   ├── models.py     done
│   │   │   ├── predict.py    stub (501)
│   │   │   └── report.py     stub (501)
│   │   ├── inference/
│   │   │   ├── registry.py   8 models metadata — done
│   │   │   └── preprocess.py ← schema validation + sliding window
│   │   └── schemas/          Pydantic request/response models
│   ├── artifacts/            ← put .pkl / .pt files here (NOT in git)
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/                 React + Vite + TypeScript
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Landing.tsx   done
│   │   │   ├── Upload.tsx    ← stub, needs full implementation
│   │   │   └── Results.tsx   ← stub, needs full implementation
│   │   ├── components/       Nav, ThemeToggle, Footer (shared — don't break)
│   │   ├── styles/           Design tokens + global CSS
│   │   └── api/              ← create client.ts here for backend calls
│   ├── package.json
│   └── Dockerfile
│
├── docker-compose.yml
└── README.md
```

---

## 2. Run it locally

### Backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Copy the trained artifacts into backend/artifacts/ — see section 5
uvicorn app.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

### Frontend
```bash
cd frontend
npm install
npm run dev
```
Frontend: http://localhost:5173 (proxies `/api/*` to the backend on 8000)

### Full stack (Docker)
```bash
docker compose up --build
# UI: http://localhost:8080
```

---

## 3. Task assignments

| Task | Owner | Files | Acceptance |
|------|-------|-------|------------|
| **T1** Upload API + schema validation | — | `backend/app/api/routes/upload.py`, `backend/app/inference/preprocess.py` | Reject malformed CSV with plain-language error (FR-02/03/04). Return `{file_id, rows, time_range, has_label}`. |
| **T2** Predict API (single model) | — | `backend/app/api/routes/predict.py` | Apply `scaler.pkl`, build (N,20,8) windows, return probs + summary. Suppress metrics when `has_label=false` (FR-21). |
| **T3** Compare API (8 models) | — | same file as T2 | Run all 8 models on the same input. Reuse T2's window builder. |
| **T4** PDF report | — | `backend/app/api/routes/report.py`, `backend/app/reports/` | ReportLab; include detected columns, preprocessing steps, plain-language chart explanation (FR-17/18). |
| **T5** Upload page UI | — | `frontend/src/pages/Upload.tsx`, new components under `frontend/src/components/` | Drop zone, schema validation feedback, model picker (8 cards), perf panel, run button (FR-01/05/06/08/11). |
| **T6** Results page UI | — | `frontend/src/pages/Results.tsx` | Probability line chart with Recharts, alert-band shading (threshold 0.5), summary panel, 8-model comparison grid, PDF download button (FR-14/15/19/20). |
| **T7** Transformer artifact + loader | — | `backend/app/inference/loader.py` (new), `backend/artifacts/model_transformer.pt` | Load PyTorch `.pt` in `eval()` mode, deterministic seed (NFR-05). Update `registry.py` metrics. |

Pick one, claim it in the group chat, then start.

---

## 4. Branch & PR workflow

```bash
# 1. Always start from the latest main
git checkout main
git pull

# 2. Branch naming: feat/<short-task-id>
git checkout -b feat/upload-api          # T1
git checkout -b feat/upload-page         # T5
git checkout -b feat/results-page        # T6
# ... etc.

# 3. Commit small, push often
git add <files>
git commit -m "feat(upload): validate SKAB schema (FR-02)"
git push -u origin feat/upload-api

# 4. Open a PR on GitHub targeting `main`
#    Title format:  [T1] Upload API + schema validation
#    Body: link the FRs you implemented + a screenshot if it's UI
```

**Rules:**
- Never push directly to `main`.
- Don't modify files outside your task scope without asking. Especially:
  `backend/app/main.py`, `backend/app/core/config.py`, `frontend/src/components/Nav.tsx`,
  `frontend/src/styles/*` — these are shared.
- If you need a new shared utility, ping the group first.
- Run `npm run build` (frontend) or hit `/docs` (backend) before pushing — make
  sure your branch at least starts.

---

## 5. Model artifacts (NOT in git)

The 7 sklearn artifacts already exist locally at:
```
data/processed/dataset2/skab_classical_models/
  scaler.pkl
  model_lr.pkl  model_rf.pkl  model_svm.pkl
  model_et.pkl  model_gb.pkl  model_knn.pkl  model_xgb.pkl
```

**Setup steps for each developer:**
1. Get the 7 `.pkl` files from David (shared via OneDrive / drive).
2. Copy them into `backend/artifacts/`.
3. The 8th file `model_transformer.pt` is pending — T7 will deliver it.

`.gitignore` already excludes `*.pkl` `*.pt` `*.npz` so they will never be
accidentally committed.

---

## 6. Conventions

**Python (backend)**
- Type hints everywhere. Pydantic for all request/response shapes.
- Use `app.core.config.settings` — never hard-code paths or thresholds.
- New endpoints go under `app/api/routes/<name>.py` and are registered in
  `app/main.py`.

**TypeScript (frontend)**
- Strict mode is on. No `any`.
- Reuse the design tokens in `src/styles/tokens.css` (CSS variables) — do not
  hard-code colors.
- API calls go through a single client in `src/api/client.ts` (T5 owner
  creates this file).
- New components: `src/components/<Name>.tsx`. Page-level layout in
  `src/pages/`.

**Commits**
- Conventional commits: `feat: ...`, `fix: ...`, `docs: ...`, `refactor: ...`.
- Reference the FR id in the body when relevant: `Implements FR-02, FR-03.`

---

## 7. Questions / blockers

Drop them in the group chat with the task ID prefix, e.g. `[T2] How should we
handle horizon != 10?`. If it's an architecture question, raise it before
writing code, not after.
