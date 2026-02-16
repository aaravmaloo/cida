# CIDA

CIDA is an AI text platform with two core capabilities:

- AI-generated text detection
- AI text humanization

It also includes admin analytics, async report generation, rate limiting, and Turnstile verification.

## Model Stack (Hugging Face)

CIDA now uses open-source Hugging Face models directly at runtime.

### Detector

- Model: `shahxeebhassan/bert_base_ai_content_detector`
- URL: https://huggingface.co/shahxeebhassan/bert_base_ai_content_detector
- Task: text classification (human vs AI)
- Base: `bert-base-uncased`
- License: MIT
- Label mapping in source dataset (`shahxeebhassan/human_vs_ai_sentences`):
  - `0` = human-written
  - `1` = AI-generated

### Humanizer

- Model: `Eemansleepdeprived/Humaneyes`
- URL: https://huggingface.co/Eemansleepdeprived/Humaneyes
- Task: text2text generation / paraphrasing
- Base: `tuner007/pegasus_paraphrase`
- License: MIT

## What the System Does

1. User submits text or a file from `apps/web`.
2. Frontend calls `/v1/*` endpoints.
3. `services/api`:
   - validates Turnstile,
   - enforces Redis sliding-window rate limits,
   - runs detector inference,
   - runs humanizer rewriting,
   - stores events in Postgres.
4. Report jobs are queued in Redis.
5. `services/worker` renders and stores JSON/PDF reports.

If Hugging Face model loading fails, API uses deterministic fallback logic (heuristic detector + rule-based rewriting) so the service still responds.

## Repository Layout

- `apps/web`: Next.js frontend
- `services/api`: FastAPI backend (detector + humanizer)
- `services/worker`: async report worker
- `packages/shared-schemas`: shared OpenAPI/schema package
- `infra/docker`: Dockerfiles + compose
- `infra/railway`: Railway service mapping

## Runtime Services

- API: FastAPI (`services/api`)
- Worker: ARQ (`services/worker`)
- Database: PostgreSQL
- Queue/cache/rate-limit state: Redis

## Environment Variables

### API (`services/api`)

Required:

- `DATABASE_URL`
- `REDIS_URL`
- `JWT_SECRET`
- `ADMIN_USER`
- `ADMIN_PASS`
- `CORS_ALLOWED_ORIGINS`

Model/runtime:

- `MODEL_VERSION`
- `DETECTOR_MODEL_NAME` (default `shahxeebhassan/bert_base_ai_content_detector`)
- `DETECTOR_ALLOW_REMOTE_DOWNLOAD` (`true` to pull from HF when not cached)
- `DETECTOR_AI_LABEL` (default `1`)
- `DETECTOR_MAX_LENGTH`
- `DETECTOR_EAGER_LOAD` (default `false`; lazy-load detector to reduce startup RAM)
- `RELEASE_DETECTOR_FOR_HUMANIZER` (default `true`; frees detector RAM before humanizer load)
- `HUMANIZER_MODEL_NAME` (default `Eemansleepdeprived/Humaneyes`)
- `HUMANIZER_USE_REMOTE_API` (default `true`; calls Hugging Face Inference API)
- `HUMANIZER_ALLOW_REMOTE_DOWNLOAD`
- `HUMANIZER_REQUIRE_MODEL` (default `true`; if model load/inference fails, request fails instead of fallback)
- `HUMANIZER_API_TIMEOUT_SECONDS` (default `60`)
- `HUMANIZER_API_URL` (optional direct endpoint URL; overrides router path)
- `HF_TOKEN` (required when `HUMANIZER_USE_REMOTE_API=true`)
- `HF_ROUTER_BASE_URL` (default `https://router.huggingface.co`)
- `HUMANIZER_MAX_INPUT_TOKENS` (upper bound; runtime also clamps to model context window)
- `HUMANIZER_MAX_NEW_TOKENS`

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

Optional:

- `REPORT_LOCAL_DIR`
- `R2_ENDPOINT`, `R2_BUCKET`, `R2_REGION`, `R2_ACCESS_KEY`, `R2_SECRET_KEY`
- `PUBLIC_BASE_URL`

### Web (`apps/web`)

Required:

- `NEXT_PUBLIC_API_BASE_URL`

## Local Run

### Option A: Docker Compose

From repo root:

```bash
docker compose -f infra/docker/docker-compose.yml up --build
```

Services:

- Web: `http://localhost:3000`
- API: `http://localhost:8000`
- Postgres: `localhost:5432`
- Redis: `localhost:6379`

### Option B: Manual

Install web deps from repo root:

```bash
npm install
```

Run API:

```bash
cd services/api
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Run worker:

```bash
cd services/worker
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
arq app.worker.WorkerSettingsConfig
```

Run web:

```bash
npm run dev:web
```

## Deployment Notes

- API Docker image no longer expects ONNX artifacts.
- Models are loaded using `transformers` + `torch` from Hugging Face model IDs.
- First startup may download models if not cached.

## Health Checks

```bash
curl -i https://<api-domain>/healthz
curl -i https://<api-domain>/readyz
```

## Credits

- Detector model by **shahxeebhassan**:
  - https://huggingface.co/shahxeebhassan/bert_base_ai_content_detector
- Humanizer model by **Eemansleepdeprived**:
  - https://huggingface.co/Eemansleepdeprived/Humaneyes
- Detector dataset by **shahxeebhassan**:
  - https://huggingface.co/datasets/shahxeebhassan/human_vs_ai_sentences

## Security and Responsible Use

- Detector output is probabilistic, not proof of authorship.
- Keep human review in high-stakes decisions.
- Disclose machine-generated scoring and rewriting to end users.
- Avoid storing sensitive text unless needed and consented.

## License and Contribution

Add or update your project license/contribution policy before external distribution.
