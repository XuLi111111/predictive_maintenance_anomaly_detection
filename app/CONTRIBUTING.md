# Contributing — pump.detect Web App

Welcome. This document tells you where the code lives, what the current
status of each module is, and how to run / extend the app without
stepping on anyone's toes.

This file covers **only the web application** under `app/`. The training
pipeline (`src/`, `data/`, `models/`, `results/`) is governed by separate
conventions in the top-level `README.md`.

> **Status as of 2026-05-13 (demo day)**: every original task T1–T9 has
> shipped, plus a second wave of UX / live-mode / report-quality work.
> See section 4 for the up-to-date status of each module.

---

## 1. Repository layout (web app only)

```
app/
├── backend/                          FastAPI + inference layer
│   ├── app/
│   │   ├── main.py                   Entry point — registers routers + CORS
│   │   ├── core/config.py            Shared settings (sensor cols, threshold, paths)
│   │   ├── api/routes/
│   │   │   ├── upload.py             POST /api/upload + GET /api/sample-csv
│   │   │   ├── models.py             GET  /api/models
│   │   │   ├── predict.py            POST /api/predict + /api/predict/compare
│   │   │   ├── report.py             POST /api/report (5-page PDF)
│   │   │   ├── stream.py             WS   /api/stream  (CSV replay)
│   │   │   └── live.py               POST /api/live/ingest + /config,
│   │   │                             GET  /api/live/status,
│   │   │                             WS   /api/live/stream
│   │   ├── inference/
│   │   │   ├── registry.py           8 models metadata
│   │   │   ├── preprocess.py         Schema validation, NaN/range/sampling checks,
│   │   │   │                         "How to fix" error messages, sliding window
│   │   │   ├── loader.py             Per-process model + scaler caches
│   │   │   ├── predict.py            prepare_input + run_model (shared)
│   │   │   ├── state_machine.py      4-tier alert state (NORMAL/WATCH/WARNING/ALERT)
│   │   │   ├── transformer_model.py  TransformerFusionLite definition
│   │   │   └── live_buffer.py        In-memory ring buffer + pub/sub +
│   │   │                             data-quality monitoring
│   │   ├── reports/
│   │   │   └── pdf.py                5-page report (Executive Summary, Detection
│   │   │                             Detail, Model Comparison + Confusion + ROC/PR,
│   │   │                             Sensor Deep-Dive, Methodology + Appendix)
│   │   └── schemas/                  Pydantic request/response models
│   ├── artifacts/                    Trained model files (NOT in git)
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend/                         React + Vite + TypeScript
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Landing.tsx           "/" — hero + mockup + benefits
│   │   │   ├── Upload.tsx            "/upload" — Steps 1-4 (inline live replay)
│   │   │   ├── Results.tsx           "/results" — batch results + PDF download
│   │   │   └── Live.tsx              "/live" — always-on monitor
│   │   ├── components/
│   │   │   ├── Nav.tsx               Tab nav (Home / Upload / Results / Live)
│   │   │   ├── ThemeToggle.tsx       Light / dark toggle
│   │   │   ├── Footer.tsx
│   │   │   ├── LiveChart.tsx         Probability chart + Brush slider + sensor tooltip
│   │   │   ├── StatusBanner.tsx      4-tier coloured banner
│   │   │   ├── SpeedControl.tsx      0.5× / 1× / 10× / 100× + pause / stop
│   │   │   ├── LiveControlBar.tsx    Status badge + model selector + threshold
│   │   │   └── LiveStatusBadge.tsx   🔴 LIVE / WAITING / DISCONNECTED
│   │   ├── hooks/
│   │   │   └── useAlertAudio.ts      Shared audio chime + alert loop
│   │   ├── api/
│   │   │   ├── client.ts             REST helpers (upload, predict, report, models)
│   │   │   ├── streamClient.ts       WS wrapper for CSV replay
│   │   │   └── liveClient.ts         WS wrapper for live stream + config + quality
│   │   └── styles/
│   │       ├── tokens.css            Design tokens (light + dark themes)
│   │       └── app.css               Global CSS
│   ├── Dockerfile
│   ├── nginx.conf
│   └── package.json
│
├── scripts/
│   └── pump_simulator.py             Local pump simulator (stdlib only)
│
├── docker-compose.yml
├── README.md
└── CONTRIBUTING.md                   (this file)
```

---

## 2. Visual prototype reference (historical)

The original interactive prototype that the client approved is at the
**repository root**: `/static landing page.html`. Each page in
`src/pages/` was ported from there; visual decisions (colours, layout,
fonts) descended from this file plus the **2026-05-12 design refresh**
(violet primary, refreshed shadows, light/dark themes).

The Upload, Results, and Live pages have evolved well beyond the
prototype — this file is kept only for traceability, not as a working
spec.

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

## 4. Module status (as of demo day 2026-05-13)

All original tasks T1–T9 are complete. The second post-T9 wave covers UX
polish, live-mode hardening, and report depth — also done.

### Backend

| Module | Status | Notes |
|--------|--------|-------|
| `upload.py` (T1) | ✅ done | Schema validation + 30-min TTL GC + "How to fix" errors. |
| `models.py` | ✅ done | Static registry → 8 cards on frontend. |
| `predict.py` (T2/T3) | ✅ done | `prepare_input` shared across endpoints; tree models on raw windows, linear / kernel on scaled, transformer on its own per-feature scaler. |
| `report.py` (T4) | ✅ done | Reruns all 8 models so the PDF comparison page is real. |
| `reports/pdf.py` | ✅ enhanced | 5-page report: Executive Summary (risk badge LOW/MEDIUM/HIGH + action sentence) · Detection Detail (chart + anomaly timeline + plain English) · Model Comparison (8-model table + agreement statement + confusion matrix + ROC/PR with best-F1 callout when labeled) · Sensor Deep-Dive (per-channel z-score + 8-panel sparklines) · Methodology + Appendix. |
| `stream.py` (T8) | ✅ done | CSV-replay WS now also emits `sensors` in each tick for chart tooltip. |
| `live.py` | ✅ new | Always-on real-time endpoint: POST ingest / POST config / GET status / WS stream. |
| `inference/live_buffer.py` | ✅ new | Single ring buffer + pub/sub + data-quality monitoring (frozen / out-of-range / uneven cadence). |
| `inference/preprocess.py` | ✅ enhanced | Sampling-rate check (median / std / gaps / duplicates), NaN check, all errors end with "How to fix" guidance. |
| `inference/state_machine.py` | ✅ done | Four-tier 0.30 / 0.50 / 0.80 with hysteresis. |
| `inference/transformer_model.py` (T7) | ✅ done | TransformerFusionLite F1 = 0.9244 on SKAB test set. |

### Frontend

| Module | Status | Notes |
|--------|--------|-------|
| `pages/Landing.tsx` | ✅ done | SVG mockup colours use CSS vars (theme-safe). |
| `pages/Upload.tsx` (T5) | ✅ enhanced | One-stop page: expected format at top → upload → model grid → threshold (with quick picks) → CTAs → **inline Step 4 live timeline** (was a separate `/monitor` route — now merged). |
| `pages/Results.tsx` (T6) | ✅ done | Main probability chart + 4-stat summary + 8-model comparison + PDF download. |
| `pages/Live.tsx` | ✅ new | Always-on monitoring with control bar (model dropdown + threshold slider + Pause/Clear), data-quality alerts banner, client-side stale-data watchdog. |
| `components/LiveChart.tsx` | ✅ enhanced | Brush slider for scrolling history + custom tooltip with 8 sensor readings + colour-coded line. |
| `components/LiveControlBar.tsx` | ✅ new | Status badge + model dropdown + threshold slider + Pause/Clear actions. |
| `components/LiveStatusBadge.tsx` | ✅ new | 4-state pill (connecting / waiting / live / disconnected) with pulse on `live`. |
| `hooks/useAlertAudio.ts` | ✅ new | Shared audio cue hook (chime + repeating alert loop). Used by both Upload inline replay and Live page. |
| `api/client.ts` | ✅ done | REST helpers + `downloadReport()`. |
| `api/streamClient.ts` | ✅ done | WS wrapper for `/api/stream` (CSV replay). |
| `api/liveClient.ts` | ✅ new | WS wrapper for `/api/live/stream` + auto-reconnect + typed frames (`hello` / `tick` / `config` / `quality`). |
| `styles/{tokens.css,app.css}` | ✅ refreshed | Violet primary, dark mode with violet-400, bolder shadow scale, full radius scale. |

### Other

| File | Status | Notes |
|------|--------|-------|
| `scripts/pump_simulator.py` | ✅ new | Realistic dynamics (jitter + bell-curve burst + random timing + `--seed`). Uses stdlib `urllib.request` — no extra pip deps. |
| `docker-compose.yml` | ✅ done | api + web (nginx) services; web has `depends_on.condition: service_healthy`. |
| `tests/` | ✅ 21 passing | Schema, state machine, stream (with `sensors` assertion), upload smoke. |

### Known low-priority leftovers (post-demo)

- Sensor unit sanity (e.g. °F vs °C). Requires a baseline-stats artifact.
- `SRS PDF` says FR-21 priority `Should`; client confirmed implementation already meets `Must`.

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
data/processed/skab/skab_classical_models/
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
