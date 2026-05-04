# Contributing — pump.detect Web App

Welcome. This document tells you what to build, where to put it, and how to
ship it without stepping on anyone's toes.

This file covers **only the web application** under `app/`. The training
pipeline (`src/`, `data/`, `models/`, `results/`) is governed by separate
conventions in the top-level `README.md`.

---

## 1. Repository layout (web app only)

```
app/
├── backend/                       FastAPI + inference layer
│   ├── app/
│   │   ├── main.py                Entry point — DO NOT modify casually
│   │   ├── core/config.py         Shared settings (sensor cols, threshold, paths)
│   │   ├── api/routes/            ← your work goes here
│   │   │   ├── upload.py          stub (501) — T1
│   │   │   ├── models.py          done
│   │   │   ├── predict.py         stub (501) — T2 / T3
│   │   │   └── report.py          stub (501) — T4
│   │   ├── inference/
│   │   │   ├── registry.py        8 models metadata — done
│   │   │   └── preprocess.py      ← schema validation + sliding window (T1)
│   │   └── schemas/               Pydantic request/response models
│   ├── artifacts/                 ← put .pkl / .pt files here (NOT in git)
│   ├── .dockerignore
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend/                      React + Vite + TypeScript
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Landing.tsx        done — reference implementation
│   │   │   ├── Upload.tsx         ← stub, T5
│   │   │   └── Results.tsx        ← stub, T6
│   │   ├── components/            Nav, ThemeToggle, Footer (shared — don't break)
│   │   ├── styles/                Design tokens + global CSS
│   │   └── api/                   ← create client.ts here for backend calls
│   ├── .dockerignore
│   ├── Dockerfile
│   ├── nginx.conf
│   ├── package.json
│   └── package-lock.json
│
├── docker-compose.yml
├── README.md
└── CONTRIBUTING.md                (this file)
```

---

## 2. Visual reference for Upload / Results pages (READ THIS BEFORE T5 / T6)

The original interactive prototype — already approved by the client — is at
the **repository root**:

```
/static landing page.html
```

It is a single self-contained HTML file with three sections (`#landing`,
`#uploadPage`, `#resultsPage`) and inline CSS / JS. To use it as a spec:

1. Open it in a browser by double-clicking the file
2. Click "Get started" to navigate to the upload screen
3. Click "Preview Results →" (bottom-right debug button) to see the results screen
4. Use these screens as the **visual + interaction reference** for T5 / T6

### How to translate it to React (T5 / T6 workflow)

`app/frontend/src/pages/Landing.tsx` is the **already-completed example** of
how the same HTML pattern was ported. Open both files side by side and you
will see the conversion is mostly mechanical:

| Original HTML            | React/JSX equivalent              |
|--------------------------|-----------------------------------|
| `class="..."`            | `className="..."`                 |
| `onclick="fn()"`         | `onClick={fn}`                    |
| `<input ... onchange>`   | `<input ... onChange>`            |
| inline `<style>` block   | already split into `tokens.css` + `app.css` |
| `<script>` functions     | React hooks (`useState`, `useEffect`) |

**Steps for T5 (Upload page):**
1. Find `<div class="upload-page" id="uploadPage">` block in the HTML
2. Translate the structure into JSX in `app/frontend/src/pages/Upload.tsx`
3. Copy the relevant CSS rules (`.drop-zone`, `.model-card`, `.perf-panel`,
   `.schema-row`, etc.) from the HTML's `<style>` block into
   `app/frontend/src/styles/app.css`
4. Convert the `<script>` functions (`handleFile`, `selectModel`,
   `updatePerformancePanel`) to React hooks
5. Replace mock behaviour with real API calls via `src/api/client.ts`

**Steps for T6 (Results page):**
1. Find `<div class="results-page" id="resultsPage">` block in the HTML
2. Same translation process
3. Replace the static SVG charts with [Recharts](https://recharts.org/)
   `<LineChart>` + `<ReferenceArea>` for the alert band

The hard work (design decisions, colours, layout, fonts, copywriting) is
already done — you are doing structural translation, not new design.

---

## 3. Run it locally

### Backend
```bash
cd app/backend
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Copy the trained artifacts into app/backend/artifacts/ — see section 6
uvicorn app.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

### Frontend
```bash
cd app/frontend
npm install
npm run dev
```
Frontend: http://localhost:5173 (proxies `/api/*` to the backend on 8000)

### Full stack (Docker)
```bash
cd app
docker compose up --build
# UI:  http://localhost:8080
# API: http://localhost:8080/api/health
```

---

## 4. Task assignments

Pick one, claim it in the group chat (reply `T1` / `T2` / ...), then start.

| Task | Owner | Files | Acceptance criteria |
|------|-------|-------|---------------------|
| **T1** Upload API + schema validation | — | `app/backend/app/api/routes/upload.py`, `app/backend/app/inference/preprocess.py` | Reject malformed CSV with plain-language error (FR-02/03/04). Return `{file_id, rows, time_range, columns_detected, has_label, warnings}`. Implement `/api/sample-csv` (FR-05). |
| **T2** Predict API (single model) | — | `app/backend/app/api/routes/predict.py` | Apply `scaler.pkl`, build (N,20,8) windows, return probs + anomaly windows + summary. Suppress `metrics` when `has_label=false` (FR-21). Threshold from `settings.alert_threshold` (FR-16). |
| **T3** Compare API (8 models) | — | same file as T2 | Run all 8 models on the same input. Reuse T2's preprocessing — do not re-run scaler/windowing per model. (FR-19) |
| **T4** PDF report | — | `app/backend/app/api/routes/report.py`, new `app/backend/app/reports/pdf.py` | ReportLab; include detected columns, preprocessing steps applied, embedded probability chart (matplotlib → PNG), and a plain-language explanation for non-technical readers (FR-17/18). |
| **T5** Upload page UI | — | `app/frontend/src/pages/Upload.tsx`, new components under `app/frontend/src/components/`, new `app/frontend/src/api/client.ts` | Drop zone, schema validation feedback, model picker (8 cards), perf panel (live metrics for selected model), threshold slider, run button (FR-01/05/06/08/11). See section 2. |
| **T6** Results page UI | — | `app/frontend/src/pages/Results.tsx` | Probability LineChart with shaded alert bands (Recharts `<ReferenceArea>`), 4-stat summary panel, 8-model comparison grid, PDF download button (FR-14/15/19/20). See section 2. |
| **T7** Transformer artifact + loader | — | new `app/backend/app/inference/loader.py`, `app/backend/artifacts/model_transformer.pt` | Load PyTorch `.pt` in `eval()` mode, set `torch.manual_seed(42)` for NFR-05. Update transformer entry in `registry.py` with real metrics. Wire branching in `predict.py` based on `is_dl` flag. |

### Dependency graph

```
T1 ──► T2 ──► T3
       │
       └────► T4
       │
       └────► T6 (frontend can mock first)
T5 (independent of backend, can mock first)
T7 (independent, but T2 must check is_dl flag)
```

**Recommended kickoff order:** T1 + T2 + T5 in parallel (frontend mocks
backend responses initially), then T3 / T6 / T4, finally T7.

---

## 5. Branch & PR workflow

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
git add <specific files>
git commit -m "feat(upload): validate SKAB schema (FR-02)"
git push -u origin feat/upload-api

# 4. Open a PR on GitHub targeting `main`
#    Title format:  [T1] Upload API + schema validation
#    Body must include:
#      - Closes #<issue-number>
#      - Which FRs you implemented
#      - Screenshot if UI work
#      - Confirmation that `cd app && docker compose up --build` still works
```

### Hard rules

- **Never push directly to `main`.** Always go through PR.
- **Don't modify files outside your task scope** without asking. Especially:
  - `app/backend/app/main.py`
  - `app/backend/app/core/config.py`
  - `app/frontend/src/components/Nav.tsx`
  - `app/frontend/src/components/ThemeToggle.tsx`
  - `app/frontend/src/styles/tokens.css`
  - `app/docker-compose.yml`
  - `app/backend/app/inference/registry.py` (only T7 may modify the transformer row)
- If you need a new shared utility, ping the group **before** writing it.
- Run `npm run build` (frontend) or hit `http://localhost:8000/docs`
  (backend) before pushing — make sure your branch at least starts.
- Hook failures: fix the underlying issue, do not bypass with `--no-verify`.

---

## 6. Model artifacts (NOT in git)

The 7 sklearn artifacts already exist in David's local copy at:
```
data/processed/dataset2/skab_classical_models/
  scaler.pkl
  model_lr.pkl  model_rf.pkl  model_svm.pkl
  model_et.pkl  model_gb.pkl  model_knn.pkl  model_xgb.pkl
```

### Setup steps for each developer

1. Get the 7 `.pkl` files from David (shared via OneDrive).
2. Copy them into `app/backend/artifacts/`.
3. The 8th file `model_transformer.pt` is pending — T7 will deliver it.

`.gitignore` already excludes `*.pkl` `*.pt` `*.npz` so they will never be
accidentally committed.

---

## 7. Conventions

### Python (backend)
- Type hints everywhere. Pydantic v2 for all request / response shapes.
- Use `app.core.config.settings` — never hard-code paths or thresholds.
- New endpoints go under `app/backend/app/api/routes/<name>.py` and are
  registered in `app/backend/app/main.py`.
- Pydantic models with a `model_id` field must include
  `model_config = ConfigDict(protected_namespaces=())` to silence the v2
  warning. See `schemas/predict.py` for the pattern.

### TypeScript (frontend)
- Strict mode is on. No `any`.
- Reuse the design tokens in `app/frontend/src/styles/tokens.css`
  (CSS variables) — do not hard-code colours.
- API calls go through a single client in `app/frontend/src/api/client.ts`
  (T5 owner creates this file).
- New components: `app/frontend/src/components/<Name>.tsx`. Page-level
  layout in `app/frontend/src/pages/`.
- Routing uses `react-router-dom` v6 — see `App.tsx`.

### Commits
- Conventional commits: `feat: ...`, `fix: ...`, `docs: ...`,
  `refactor: ...`, `chore: ...`, `test: ...`.
- Reference the FR id in the body when relevant: `Implements FR-02, FR-03.`
- Reference the GitHub issue: `Closes #<n>` (closes on merge) or
  `Refs #<n>` (just links).

---

## 8. Questions / blockers

Drop them in the group chat with the task ID prefix:
- `[T2] How should we handle horizon != 10?`
- `[T5] Need a shared <Spinner /> — OK to add?`

If it's an architecture question, raise it **before** writing code, not
after. Better to lose 10 minutes asking than 2 days rewriting.
