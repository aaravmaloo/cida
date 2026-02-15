# CIDA (AI Detection + Humanization Platform)

Production-oriented monorepo for:

- AI-generated text detection
- AI text humanization
- Admin analytics and report generation
- Detector training + calibration + ONNX export

---

## 1. Architecture

### Applications

- `apps/web`: Next.js frontend (Vercel)
- `services/api`: FastAPI backend (Railway/Render)
- `services/worker`: ARQ background worker for report jobs
- `services/trainer`: detector training pipeline
- `packages/shared-schemas`: shared schema artifacts

### Runtime flow

1. User submits text in web UI.
2. Web calls API `/v1/analyze`.
3. API runs detector (ONNX + tokenizer + calibration).
4. API stores events in Postgres and caches repeat requests in Redis.
5. Report requests are enqueued to worker.
6. Worker renders JSON/PDF and stores/serves report files.

---

## 2. What Was Upgraded

### Detector model pipeline

- Upgraded trainer defaults to `microsoft/deberta-v3-large`.
- Added weighted focal BCE support for harder decision boundaries.
- Added temperature scaling + validation threshold optimization.
- Added expanded metrics: accuracy/f1/precision/recall/roc_auc/brier.
- Added runtime bundle export with ONNX + tokenizer + calibration manifest.

### Detector inference quality

- API now loads tokenizer from local artifacts first (runtime bundle path).
- API uses calibrated `optimal_threshold` from calibration artifact.
- Clear model/fallback logging at startup.
- Fallback heuristic retained only when model artifacts are missing.

### Humanizer quality

- Improved fallback rewrite engine (style-aware rewriting, sentence splitting, term preservation).
- Still supports transformer path when full ML stack is available.

---

## 3. Local Development

### Prerequisites

- Node 20+ / npm
- Python 3.11
- Docker (optional, for local stack)

### Install frontend deps

```bash
npm install
```

### Install API deps

```bash
cd services/api
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

### Install trainer deps

```bash
cd services/trainer
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

### Run local infra + services

```bash
docker compose -f infra/docker/docker-compose.yml up --build
```

or run API/web separately:

```bash
cd services/api
uvicorn app.main:app --reload --port 8000
```

```bash
npm run dev:web
```

---

## 4. Detector Training (Recommended Flow)

From `services/trainer`:

```bash
python -m src --csv ../../train_data/balanced_ai_human_prompts.csv --output-dir ../artifacts/latest --quantize
```

This runs:

1. `src.train`
2. `src.calibration`
3. `src.evaluate`
4. `src.export_onnx`
5. `src.model_card`

### Main artifacts

- `artifacts/latest/model/`
- `artifacts/latest/model.onnx`
- `artifacts/latest/calibration.json`
- `artifacts/latest/eval_metrics.json`
- `artifacts/latest/runtime_bundle/`
- `artifacts/latest/model_card.json`

---

## 5. Deploying Detector Artifacts to API

API quality depends on these files being available at runtime:

- `DETECTOR_ONNX_PATH` -> ONNX model
- tokenizer directory (typically sibling `model/` folder in runtime bundle)
- `CALIBRATION_PATH` -> calibration JSON with `optimal_threshold`

### Useful API env vars

- `DETECTOR_ONNX_PATH`
- `DETECTOR_TOKENIZER_PATH` (optional explicit tokenizer dir)
- `CALIBRATION_PATH`
- `DETECTOR_MODEL_NAME` (fallback tokenizer source)
- `DETECTOR_ALLOW_REMOTE_DOWNLOAD` (`false` by default)

If artifacts are missing, API falls back to heuristic scoring.

---

## 6. Deployment (Railway + Vercel)

### API service

- Dockerfile: `infra/docker/Dockerfile.api`
- Start command: leave empty (Dockerfile CMD), or:
  - `sh -c "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"`

### Worker service

- Dockerfile: `infra/docker/Dockerfile.worker`
- Start command: leave empty (Dockerfile CMD)

### Frontend (Vercel)

- `NEXT_PUBLIC_API_BASE_URL` must be the API public URL and include protocol.
- Browser calls are proxied through same-origin rewrites (`/v1/*`) to reduce CORS issues.

---

## 7. Required Environment Variables

### API

- `DATABASE_URL`
- `REDIS_URL`
- `JWT_SECRET`
- `ADMIN_PASSKEY_HASH`
- `CORS_ALLOWED_ORIGINS` (example: `https://cida-web.vercel.app`)
- `DETECTOR_ONNX_PATH`
- `CALIBRATION_PATH`

### Worker

- `DATABASE_URL`
- `REDIS_URL`
- Optional storage vars (`R2_*`, `REPORT_LOCAL_DIR`, etc.)

### Web

- `NEXT_PUBLIC_API_BASE_URL`

---

## 8. Operational Checks

### Health

```bash
curl -i https://<api-domain>/healthz
curl -i https://<api-domain>/readyz
```

### CORS preflight

```bash
curl -i -X OPTIONS "https://<api-domain>/v1/analyze" \
  -H "Origin: https://cida-web.vercel.app" \
  -H "Access-Control-Request-Method: POST"
```

### Common issue signatures

- `502 Application failed to respond` -> service not reachable, wrong start command/port/domain.
- `UndefinedTableError: relation ... does not exist` -> DB schema missing.
- `tokenizer_unavailable` / `onnx_missing_or_ort_unavailable` logs -> model artifacts not mounted.

---

## 9. Notes on "SOTA"

This repo now has a stronger, modern baseline pipeline, but true SOTA claims require:

- larger and fresher benchmark datasets
- strict benchmark protocol
- repeated runs + statistical reporting
- external leaderboard comparison

Use this stack as a production-ready high-quality baseline and iterate with better data.
