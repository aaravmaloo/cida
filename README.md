# CIDA

CIDA is a production-oriented AI text platform with two core capabilities:

- AI-generated text detection
- AI text humanization

It also includes admin analytics, async report generation, and a training pipeline for exporting detector artifacts.

## What This System Does

- Accepts text or file input from the web app.
- Scores AI-likelihood with a calibrated detector.
- Rewrites text with configurable humanization style/strength.
- Stores events for analytics and auditing.
- Generates downloadable JSON/PDF reports asynchronously.

## How It Works

1. A user submits content from `apps/web`.
2. Frontend calls `/v1/*` endpoints (rewritten to backend in Next.js config).
3. `services/api` applies:
   - rate limiting (Redis sliding window),
   - Turnstile verification,
   - detector inference or heuristic fallback,
   - persistence of analysis/humanization/report metadata in Postgres.
4. Report requests enqueue an ARQ job in Redis.
5. `services/worker` consumes the job, renders report artifacts, stores them (local or R2), and updates job status.

## AI Models and Training Techniques

### Detector Model

- Primary backbone: `microsoft/deberta-v3-large` fine-tuned for binary classification (`0=human`, `1=ai`).
- Alternate path: custom from-scratch BERT classifier with configurable layers/hidden size/heads/FFN, with minimum parameter enforcement.
- Sequence handling: head-tail tokenization at max length 512 to preserve both opening and ending context in long text.

### Training Objective and Optimization

- Custom `WeightedFocalBCETrainer` is used:
  - weighted BCE with `pos_weight = negatives / positives` for class imbalance,
  - focal modulation (`focal_gamma`, default `1.5`) to focus on hard examples.
- Early stopping callback is enabled.
- Precision/accelerator policy is adaptive:
  - TPU bf16 and GPU bf16/fp16/tf32 are auto-controlled by hardware capability and flags.

### Calibration and Thresholding

- Raw logits are calibrated with temperature scaling by minimizing NLL on validation logits.
- Expected Calibration Error (ECE) is computed with reliability bins (default 15 bins).
- Decision threshold is tuned by scanning `0.05..0.95` and selecting the best F1 (accuracy tie-break).
- `calibration.json` stores:
  - `temperature`
  - `ece`
  - `optimal_threshold`
  - reliability breakdown

### Evaluation

- Metrics reported at both `0.5` and `optimal_threshold`:
  - accuracy, F1, precision, recall, ROC-AUC, Brier score.
- Outputs are generated from saved test logits (`test_logits.npz`) after calibration.

### Serving and Inference

- Model is exported to ONNX (`opset 17`) with dynamic axes.
- Optional dynamic quantization to int8 is supported through ONNX Runtime quantization.
- Runtime bundle contains ONNX graph, tokenizer files, calibration, metrics, and manifest.
- API inference flow:
  - tokenize input,
  - run ONNX logits,
  - apply calibrated sigmoid (`logit / temperature`),
  - classify with tuned threshold.

### Fallback Heuristic (When Artifacts Are Missing)

- If ONNX Runtime or tokenizer artifacts are unavailable, API falls back to metric-based heuristic scoring.
- Heuristic uses engineered text signals:
  - readability grades,
  - complexity,
  - burstiness,
  - lexical diversity,
  - average word length.
- Confidence is capped in fallback mode to avoid overconfident outputs.

### Humanizer Model Path

- Primary optional model path: `google/flan-t5-base` via `text2text-generation`.
- If transformer pipeline is unavailable, system uses deterministic rule-based rewriting:
  - style-specific substitutions (`natural`, `concise`, `formal`),
  - sentence splitting for long clauses,
  - preserved-term protection/restoration,
  - quality flags (minimal change, over-compression, readability shift).

## Why It Works

- Calibration-aware inference: detector applies temperature scaling and tuned threshold from `calibration.json`, improving decision reliability over raw logits.
- Fast serving path: ONNX Runtime + tokenizer loading prioritizes local artifacts for predictable runtime.
- Resilience: if ONNX/tokenizer artifacts are unavailable, API falls back to heuristic scoring instead of hard failing.
- Abuse controls: per-IP rate limits + Turnstile checks reduce endpoint abuse.
- Async job architecture: heavy report generation is offloaded from request/response path to worker queue.
- Observability hooks: health/readiness endpoints and Prometheus instrumentation support operations.

## Underlying Tech

- Frontend: Next.js 15, React 19, TypeScript, Tailwind (`apps/web`)
- API: FastAPI, SQLAlchemy (async), Alembic, Redis, ONNX Runtime (`services/api`)
- Worker: Python + ARQ + ReportLab + optional S3-compatible storage (R2) (`services/worker`)
- Trainer: PyTorch/Transformers pipeline for training, calibration, evaluation, ONNX export (`services/trainer`)
- Shared schemas: OpenAPI/shared package (`packages/shared-schemas`)

## Repository Layout

- `apps/web`: user/admin UI
- `services/api`: primary backend
- `services/worker`: async report processor
- `services/trainer`: detector training/export pipeline
- `infra/docker`: local and deploy Dockerfiles
- `infra/railway`: minimal Railway service mapping
- `train_data`: local training data inputs

## Runtime Services and Data Stores

Primary runtime services:

- `api` (FastAPI)
- `worker` (ARQ worker)

Primary data stores:

- PostgreSQL (system of record for events/jobs/admin logs)
- Redis (cache + queue + rate-limit state)

This is your "2 databases" setup on Railway: one relational data store (Postgres) and one in-memory data store (Redis).

## Where It Is Running (The Current Deployment)

Based on current project docs/config:

- Backend runtime: Railway
  - Service `api` from `infra/docker/Dockerfile.api`
  - Service `worker` from `infra/docker/Dockerfile.worker`
  - Managed `postgres` and `redis` Railway services
- Frontend runtime: Vercel (configured to proxy `/v1/*` to Railway API)

## Environment Variables

### API (`services/api`)

Required:

- `DATABASE_URL`
- `REDIS_URL`
- `JWT_SECRET`
- `ADMIN_USER`
- `ADMIN_PASS`
- `CORS_ALLOWED_ORIGINS`

Detector/runtime:

- `DETECTOR_ONNX_PATH`
- `CALIBRATION_PATH`
- `DETECTOR_MODEL_NAME` (fallback tokenizer source)
- `DETECTOR_TOKENIZER_PATH` (optional explicit tokenizer directory)
- `DETECTOR_ALLOW_REMOTE_DOWNLOAD` (optional; defaults to safer local-only behavior)

Optional:

- `TURNSTILE_SECRET`
- `SENTRY_DSN`
- `R2_ENDPOINT`, `R2_BUCKET`, `R2_REGION`, `R2_ACCESS_KEY`, `R2_SECRET_KEY`
- `REPORT_LOCAL_DIR`
- `PUBLIC_BASE_URL`

### Worker (`services/worker`)

Required:

- `DATABASE_URL`
- `REDIS_URL`

Optional storage/output:

- `REPORT_LOCAL_DIR`
- `R2_ENDPOINT`, `R2_BUCKET`, `R2_REGION`, `R2_ACCESS_KEY`, `R2_SECRET_KEY`
- `PUBLIC_BASE_URL`

### Web (`apps/web`)

Required:

- `NEXT_PUBLIC_API_BASE_URL` (Railway API public URL, including `https://`)

## Local Setup and Run

### Option A: Full stack via Docker (recommended)

From repository root:

```bash
docker compose -f infra/docker/docker-compose.yml up --build
```

Endpoints:

- Web: `http://localhost:3000`
- API: `http://localhost:8000`
- Postgres: `localhost:5432`
- Redis: `localhost:6379`

### Option B: Run services manually

### 1) Install web dependencies

```bash
npm install
```

### 2) Run API

```bash
cd services/api
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 3) Run worker

```bash
cd services/worker
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
arq app.worker.WorkerSettingsConfig
```

### 4) Run web

From repo root:

```bash
npm run dev:web
```

## Model Training and Artifact Handoff

Train and export from `services/trainer`:

```bash
python -m src --csv ../../train_data/balanced_ai_human_prompts.csv --output-dir ../artifacts/latest --quantize
```

Expected outputs include:

- `services/trainer/artifacts/latest/model.onnx`
- `services/trainer/artifacts/latest/calibration.json`
- `services/trainer/artifacts/latest/runtime_bundle/`

Deploy these artifacts with API and point env vars accordingly (`DETECTOR_ONNX_PATH`, `CALIBRATION_PATH`, tokenizer path).

### Railway-first (no local hosting required)

The API Docker image bakes detector artifacts from `services/artifacts/artifacts_latest` into `/app/runtime_bundle` at build time.
If `calibration.json` is missing, a safe default is generated during build. If tokenizer files are missing, build attempts to fetch and save the tokenizer for `microsoft/deberta-v3-large`.

## Health and Ops Checks

```bash
curl -i https://<api-domain>/healthz
curl -i https://<api-domain>/readyz
```

Useful signals:

- `healthz` failing: service is down or unreachable.
- `readyz` failing: dependency readiness issue (typically DB connectivity).
- detector warnings about missing ONNX/tokenizer: running fallback mode.

## Security, Ethics, and Moral Commitments

This system can influence trust, reputation, and moderation decisions. It must be used with clear guardrails:

- No single-score absolutism: detector output is probabilistic, not proof of authorship.
- Human oversight required: never apply punitive or high-stakes decisions solely from model output.
- Transparency: disclose when text has been machine-transformed or machine-scored.
- Privacy by design: avoid storing raw sensitive user text unless strictly necessary and consented.
- Fairness and bias checks: evaluate performance across writing styles, dialects, and non-native English usage.
- Abuse prevention: maintain anti-spam controls, authentication boundaries, and auditable admin actions.
- Responsible deployment: version models, track calibration quality, and roll back quickly if drift/degradation appears.
- Academic/work integrity: do not present this as a cheating tool; position it for editing assistance and risk analysis.

## License and Contribution

Add or update your project license and contribution policy before external distribution.
