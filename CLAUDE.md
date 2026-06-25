# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A university project (Sistemas Lineales y Señales) building a Brain-Computer Interface that
classifies motor imagery from EEG. The academic point is to make **LTI system theory explicit
in the code**: discrete convolution, FIR filters, linear spatial filtering (CSP), and frequency
response — not to hide it behind library calls. EEGNet (deep learning) is secondary; it's
included because its conv layers mimic the hand-built FIR/CSP pipeline.

No real hardware yet — everything runs on public EEG datasets (MOABB). The architecture is
meant to leave room for a future Ultracortex Mark IV headset via LSL.

The project has three stages: (1) backend LTI pipeline + offline classification + live-stream
simulation — done; (2) didactic React frontend — in progress; (3) interoperability layer
(control games via LSL, Arduino via Serial, etc.) — not started.

Communication/comments in this repo are in **Spanish** — keep new docstrings/comments/commit
messages consistent with that unless told otherwise.

## Repo layout

```
backend/   Python DSP/ML pipeline + FastAPI server (Stage 1 + API for Stage 2)
frontend/  React/TS SPA — didactic UI (Stage 2)
configs/   Dataset YAML configs (default.yaml, physionet.yaml, liu2024.yaml)
docs/      frontend-design.md (frontend architecture decisions), glosario.md (served via
           /api/glossary), presentacion.md
```

**Note on `configs/`:** these YAMLs live at the **repo root**, not inside `backend/`.
`backend/src/bci/server/app.py` resolves them explicitly via `BACKEND_ROOT.parent / cfg_path`.
However `backend/src/bci/config.py`'s `load_config()` default (`DEFAULT_CONFIG_PATH`) still
points at `backend/configs/default.yaml`, which no longer exists — so scripts relying on the
*default* (no `--config` passed) will fail until that's reconciled. Always pass
`--config ../configs/default.yaml` (or fix `config.py`) when running backend scripts standalone.

## Backend (`backend/`)

### Setup & commands

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # torch installed separately, see comment in the file

python scripts/download_data.py --save                       # fetch/cache a dataset
python scripts/setup_data.py --config ../configs/default.yaml  # bulk-download ALL subjects (new machine)
python scripts/run_offline.py [--config configs/physionet.yaml]  # full offline pipeline
python scripts/evaluate_all.py [--config ...] [--subjects 1 2 3] # within-subject eval, all subjects
python scripts/run_live_sim.py [--realtime]                    # causal streaming simulation
python scripts/train_model.py / train_eegnet.py                # persist a trained model (.pkl+.json)
python scripts/train_all_regimes.py --config ../configs/default.yaml  # train/persist the 4 regimes per subject
python scripts/precompute_payloads.py --config ../configs/default.yaml  # precompute offline viz JSON (portability)
python scripts/probe_dataset.py --dataset BNCI2014_004         # empirical viability probe of a MOABB dataset
python scripts/run_server.py [--reload]                        # FastAPI on :8000
python scripts/demo_dsp.py && python scripts/demo_csp.py       # didactic figures
```

Tests (pytest, configured via `pyproject.toml` with `testpaths = ["tests"]`):

```bash
cd backend
python -m pytest -q                       # all tests
python -m pytest tests/test_dsp.py -v     # single file
python -m pytest tests/test_csp.py::test_name -v   # single test
```

There is no linter/formatter configured for the backend (no ruff/black/flake8 config) —
match the surrounding style.

### Pipeline architecture

The core abstraction is `MotorImageryPipeline` (`backend/src/bci/pipeline/offline.py`), a
fit/predict chain: **FIR (fixed) → CSP → log-variance → LDA**. It's built from `fs` (the
dataset's *real* sample rate) so the same pipeline/config works across datasets with different
sampling rates (BCI IV 2a=250Hz, PhysionetMI=160Hz, Liu2024=500Hz) — see "fs dinámico" in the
README.

Data flow through the modules:

1. **`datasets/moabb_loader.py`** — loads raw (unfiltered) epochs via MOABB+MNE into an
   `EpochedData` dataclass (`X`: trials×channels×samples, `y`, `metadata` with
   subject/session/run, `ch_names`, `sfreq`). Deliberately does **no** frequency filtering here
   — that's the point of doing it by hand downstream.
2. **`dsp/fir_filters.py`** — designs a bandpass FIR (µ/β, 8–30Hz) by windowed-sinc
   (ideal lowpass difference → truncate → Hamming window → normalize gain). Odd `num_taps`
   required for linear phase.
3. **`dsp/convolution.py`** — hand-rolled discrete convolution in three forms: `convolve_mac`
   (literal double loop, for teaching), `mac_terms` (exposes individual MAC products for the
   frontend's convolution visualization), `convolve` (vectorized, used in production via
   `apply_filter`). Intentionally **not** using `scipy.signal.lfilter` (that solves a difference
   equation / can be IIR — this project wants explicit convolution).
4. **`spatial/csp.py`** — Common Spatial Patterns as a linear spatial filter, `Z = W·X`, solved
   via the Koles (1990) generalized-eigenvalue method (whiten the composite covariance, then
   jointly diagonalize per-class covariances). Binary only (2 classes); `n_components` must be
   even (taken in pairs from the extremes of the eigenvalue spectrum). Optional shrinkage
   regularization toward a scaled identity, useful with few trials per class.
5. **`features/log_variance.py`** — log-variance per CSP component is the feature vector.
6. **`models/lda.py`** — linear classifier (closed-form, decision boundary = hyperplane).

**Data leakage discipline**: CSP and LDA must be fit only on the training partition within
each split; the FIR has fixed coefficients so applying it doesn't leak. `evaluate_kfold`
(stratified CV) and `evaluate_by_session` (train on session `0train`, test on `1test` — the
honest inter-session estimate of live performance) are both in `pipeline/offline.py`.

**Causality matters for streaming**: offline filtering uses `mode='same'` (uses future samples,
fine for batch). Live streaming cannot do that — `streaming/simulator.py`'s `CausalFIR`
maintains a state buffer of the last `M-1` samples between chunks so filtering by chunks gives
the exact same result as filtering all at once, fully causal (`mode='valid'`). This introduces
a real group delay of `(M-1)/2` samples that's unavoidable live (compensated for offline by
`'same'`). `StreamSimulator.stream()` runs a sliding window over a continuous signal, classifies
each window via `pipeline.classify_window()`, and emits `{t, pred, probs, power, feat, disc}` —
`feat` is the log-variance vector (CSP stage) and `disc` is the signed projection onto the LDA
discriminant axis (CSP stage vs. LDA stage, kept separate for the frontend's pipeline-stage
visualization).

**Train/demo split discipline** (`pipeline/training.py`): models are trained *before* streaming
and persisted (`.pkl` + a `ModelCard` `.json` "ficha"). If the dataset has ≥2 sessions, train on
the first and reserve the second as the held-out "demo" stream (honest inter-session estimate);
otherwise reserve a stratified 30% split. The live demo only ever replays the held-out trials —
the model never saw them during training. EEGNet is trained differently (see
`train_eegnet_subject` docstring): it's a visualization mirror, not a live classifier — trained
on *all* trials for cleaner learned filters, with separate honest accuracy numbers
(inter-session + within-subject k-fold) reported alongside.

**`models/eegnet.py`**: the deep-learning bridge. Its Conv2D temporal layer ≈ a learned FIR
filter bank; its DepthwiseConv2D spatial layer ≈ a learned CSP; SeparableConv2D+pooling ≈
band-power extraction. Used only to visualize what a network *discovers* on its own
(`/api/eegnet` exposes its learned filters' frequency response + spatial weights for
side-by-side comparison with the hand-built FIR/CSP). **As of 2026-06, EEGNet IS also used
for live inference** in the Clasificación page (decision reverted): `EEGNetStreamSimulator`
(`streaming/simulator.py`) streams it window-by-window, just without the CSP/LDA stage panels.

### Server (`backend/src/bci/server/app.py`)

FastAPI app exposing the Stage 1 pipeline to the frontend. Key things to know before touching it:

- `REGISTRY` maps dataset id → `{label, config, subjects, fs, accuracy, sessions}`; this is the
  single source of truth for which datasets the API serves. There is **no `role` field** (removed
  jun 2026): a dataset's use is derived from `sessions` via `_is_live(meta)` (`sessions >= 2` ⇒
  suitable for the live demo's honest inter-session estimate). ALL datasets appear in Results
  (population benchmark); the `live` ones (≥2 sessions) **also** appear in the live demo. The
  `/api/datasets`, `/api/results*`, `/api/train_config` responses expose `sessions` + a derived
  `live` boolean (frontend `lib/datasets.ts` `isLive`/`LIVE_DATASET_LIST` mirror this).
- Three in-memory caches: `_data_cache` (loaded `EpochedData` per dataset+subject),
  `_model_cache` (trained model+card per dataset+subject+method), `_csp_cache`/`_raw_cache`
  for derived/expensive responses. Cache keys are tuples; invalidation is process-lifetime only
  (restart the server to pick up retrained models).
- `_ensure_model()` loads a persisted model from disk if present; otherwise trains and saves one
  on the fly as a fallback so the demo never hard-fails — but the intended workflow is to train
  via `scripts/train_model.py`/`train_eegnet.py` ahead of time.
- **Precomputed viz payloads (portability):** the offline visualization endpoints (`/api/info`,
  `/api/positions`, `/api/csp`, `/api/csp_signal`, `/api/lda`, `/api/eegnet`) are **disk-first** —
  they serve a precomputed `viz_{dataset}_s{subject}_{method}_{kind}.json` (under `paths.processed`)
  if present, else compute on the fly from raw data (current fallback). The build logic lives **once**
  in `server/payloads.py` (`build_*` functions), used by both the server and
  `scripts/precompute_payloads.py`; precomputed and on-the-fly responses are byte-identical (locked by
  `tests/test_payloads.py`). After this, the web renders all offline pages **without raw data** — raw
  data is only needed for live `/ws/stream` of the demo dataset. Re-run `precompute_payloads.py` after
  retraining a model.
- `/api/glossary` parses `docs/glosario.md` directly (categories = `## N. Title`, terms =
  `### Title`) — that file is the single source of truth for glossary content, not duplicated
  in the frontend.
- `/ws/stream` replays only the held-out ("demo") trials in a loop, applying causal FIR +
  `StreamSimulator` and pushing JSON frames at `step_s` cadence.

### Datasets

Each dataset has its own YAML in `configs/` (root-level, see note above) with sections
`dataset`, `epoching`, `classification_window`, `fir_filter`, `csp`, `classifier`, `streaming`,
`paths`. The pipeline auto-detects each dataset's real `fs` from the loaded epochs rather than
hardcoding it, so the same YAML-driven pipeline serves BCI IV 2a (250Hz), PhysionetMI (160Hz),
and Liu2024 (500Hz).

## Frontend (`frontend/`)

### Setup & commands

```bash
cd frontend
npm install
npm run dev      # http://localhost:5173, proxies /api and /ws to FastAPI on :8000 (vite.config.ts)
npm run build    # tsc -b (typecheck) && vite build
npm run lint      # eslint .
npm run preview
```

There's no frontend test runner configured — verification is via `npm run build` (typecheck)
and manual exercise of the dev server.

Stack: React 19 + TypeScript + Vite, Tailwind v4, react-router-dom, **Zustand** (global state),
**uPlot** (live signal charts, imperative `setData` for ~10Hz updates without re-render),
Recharts (analytical charts), three.js + @react-three/fiber + drei (3D brain), Framer Motion,
lucide-react. Full rationale for each library choice is in `docs/frontend-design.md`.

### Architecture

- **`store/useStore.ts`** (Zustand) is the single global state: selected dataset/subject/channel,
  play/pause/loop, sidebar, system status, latency, a rolling log buffer, and the UI accent
  color. The sidebar (`components/layout/Sidebar.tsx`) is the "master controller" that drives
  this store; pages read from it but Play/Pause/Loop only affect the current page's widgets,
  not the whole app.
- **`api/client.ts`** — thin REST (`getJSON`) + WebSocket (`openStream`) wrapper. In dev, Vite
  proxies `/api` and `/ws` to the FastAPI server (see `vite.config.ts`); no base URL config
  needed.
- **Routing** (`App.tsx`): all routes nest under `AppLayout` (top nav + collapsible sidebar).
  Pages: `Home`, `Dashboard`, `SignalLab` (`/lab`), `SpatialCSP` (`/csp`, "El Modelo"),
  `Brain3DPage` (`/brain`), `LiveStream` (`/live`, "Clasificación"), `Results`, `Glossary`.
- **Two worlds** (`lib/nav.ts`): every page is tagged `'offline'` (amber, pre-computed before
  streaming: El Modelo/CSP, Results) or `'online'` (green, real-time: Laboratorio, Clasificación,
  Cerebro 3D) or `'general'`. `components/WorldBadge.tsx` + `PageShell`'s `world` prop surface
  this distinction in the UI; keep new pages tagged correctly when adding them to `NAV_GROUPS`.
- **Grid/widget system** (`components/GridBoard.tsx`): a **homegrown** drag/resize/snap grid
  (12 columns, dotted background, per-page layout persisted to `localStorage`), used uniformly
  across Dashboard, Laboratorio, Clasificación, El Modelo, and Resultados. `react-grid-layout`
  was evaluated and **dropped** — its v2.x didn't hook drag/resize correctly under React 19 — so
  don't reach for it; extend `GridBoard` instead. Each panel is a `GridWidget`
  (`{ i, title, accent, w, h, minW, minH, actions?, el }`); adding/removing widgets reconciles
  the layout via first-fit placement for new ones while preserving existing positions.
  `components/charts/FillChart.tsx` measures available height via `ResizeObserver` and feeds it
  to `UPlotChart`, which resizes via uPlot's `setSize` (no instance recreation) to stay smooth
  on live data.
- **Live pipeline visualization** (`components/charts/PipelineStages.tsx`): renders the
  filtered-signal → CSP-projection → LDA-decision chain as three explicit stages so the FIR/CSP/
  LDA boundary is visually obvious, not implied. The CSP and LDA panels are imperative SVG
  (background "training cloud" drawn once, live point moved via refs/`useImperativeHandle`) for
  the same reason uPlot uses `setData` — avoiding React re-render at streaming frequency.
- **EEGNet** lives as a tab inside `SpatialCSP`/"El Modelo" (`components/EEGNetModel.tsx`,
  comparing learned vs. hand-built filters) **and** as one of the 4 selectable regimes in the
  Clasificación live page (`LiveStream.tsx`: CSP+LDA/EEGNet × within/cross). The earlier "no
  EEGNet live" scope decision was **reverted (2026-06)**: the live selector passes `method=` to
  `/ws/stream`; EEGNet streams via `EEGNetStreamSimulator` (no CSP/LDA stage panels for it).

## Cross-cutting things to know

- The backend has no DB — everything is either loaded fresh from MOABB/MNE, computed on the
  fly, or persisted as flat `.pkl`/`.json` model artifacts under each dataset's `paths.processed`
  directory.
- CORS is wide open (`allow_origins=['*']`) in `server/app.py` — fine for this local/academic
  setup, but don't assume that pattern generalizes if this is ever deployed.
- When adding a new dataset, you need: a new YAML in `configs/`, an entry in `REGISTRY`
  (`server/app.py`), and the dataset must already be loadable via `moabb.datasets.<Name>` (or via
  a custom downloader under `datasets/downloaders/` for non-MOABB sources, per the
  diversity-of-datasets goal stated in `Instrucciones.txt`).
