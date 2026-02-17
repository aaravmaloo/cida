# CIDA

CIDA is an AI text platform with a core capability focused on:

- AI-generated text detection

It also includes admin analytics, async report generation, rate limiting, and Turnstile verification.

## Model Stack (HF Space)

CIDA now runs detector scoring through a Hugging Face Space endpoint.

### Detector

- Model: `desklib/ai-text-detector-v1.01`
- Provider: Hugging Face Space (`aaravmaloo/ai-content-detector`)
- Task: AI-likelihood scoring (returns `ai_probability` in `[0, 1]`)
- Runtime: API calls Space Gradio API (default `/gradio_api/call/detect_ai_content`) with `{"data": ["<text>"]}` payload; if unavailable, service falls back to local heuristic scoring.

## What the System Does

1. User submits text or a file from `apps/web`.
2. Frontend calls `/v1/*` endpoints.
3. `services/api`:
   - validates Turnstile,
   - enforces Redis sliding-window rate limits,
   - runs detector inference,
   - stores events in Postgres.
4. Report jobs are queued in Redis.
5. `services/worker` renders and stores JSON/PDF reports.

If Space inference is unavailable or response parsing fails, API uses deterministic fallback logic (heuristic detector) so the service still responds.

## Repository Layout

- `apps/web`: Next.js frontend
- `services/api`: FastAPI backend (detector)
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

- `HF_SPACE_PREDICT_URL` (default `https://aaravmaloo-ai-content-detector.hf.space/gradio_api/call/detect_ai_content`)
- `HF_SPACE_API_TOKEN` (optional)
- `HF_SPACE_TIMEOUT_SECONDS` (default `20`)
- `HF_SPACE_MAX_INPUT_CHARS` (default `12000`)
- `HF_SPACE_MODEL_VERSION` (default `desklib/ai-text-detector-v1.01`)

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
- Detector scoring is done via HF Space API calls.
- If your Space is private, configure `HF_SPACE_API_TOKEN` in deployment secrets.

## Health Checks

```bash
curl -i https://<api-domain>/healthz
curl -i https://<api-domain>/readyz
```

## Credits

- Detector inference provider: **Hugging Face Space**
- Detector model: **Desklib `desklib/ai-text-detector-v1.01`**

## Security and Responsible Use

- Detector output is probabilistic, not proof of authorship.
- Keep human review in high-stakes decisions.
- Disclose machine-generated scoring to end users.
- Avoid storing sensitive text unless needed and consented.

## License and Contribution

Add or update your project license/contribution policy before external distribution.
