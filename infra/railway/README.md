# Railway backend deployment

Deploy `services/api` and `services/worker` as separate Railway services.

## API service

- Root directory: `.`
- Dockerfile: `infra/docker/Dockerfile.api`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Optional build arg for full ML humanizer backend: `INSTALL_TORCH_CPU=1` (default is `0` for smaller images)

## Worker service

- Root directory: `.`
- Dockerfile: `infra/docker/Dockerfile.worker`
- Start command: `arq app.worker.WorkerSettingsConfig`

## Required variables

- `DATABASE_URL`
- `REDIS_URL`
- `JWT_SECRET`
- `ADMIN_PASSKEY_HASH`
- `CORS_ALLOWED_ORIGINS`
- `NEXT_PUBLIC_API_BASE_URL` (in Vercel)
- Optional R2: `R2_ENDPOINT`, `R2_BUCKET`, `R2_ACCESS_KEY`, `R2_SECRET_KEY`

