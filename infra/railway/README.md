# Railway backend deployment

Deploy `services/api` and `services/worker` as separate Railway services.

## API service

- Root directory: `.`
- Dockerfile: `infra/docker/Dockerfile.api`
- Start command: leave empty to use Dockerfile `CMD` (recommended).
- If you override it, use: `sh -c "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"`
- Optional build arg for full ML humanizer backend: `INSTALL_TORCH_CPU=1` (default is `0` for smaller images)

## Worker service

- Root directory: `.`
- Dockerfile: `infra/docker/Dockerfile.worker`
- Start command: leave empty to use Dockerfile `CMD` (recommended).

## Required variables

- `DATABASE_URL`
- `REDIS_URL`
- `JWT_SECRET`
- `ADMIN_PASSKEY_HASH`
- `CORS_ALLOWED_ORIGINS` (for Vercel: `https://cida-web.vercel.app`, no trailing slash)
- Optional: `CORS_ALLOW_ORIGIN_REGEX` (e.g. `^https://cida-web\\.vercel\\.app$`)
- `NEXT_PUBLIC_API_BASE_URL` (in Vercel)
- `DETECTOR_ONNX_PATH` (or keep default)
- `CALIBRATION_PATH` (or keep default)
- Optional: `DETECTOR_TOKENIZER_PATH`
- Optional: `DETECTOR_ALLOW_REMOTE_DOWNLOAD=true`
- Optional R2: `R2_ENDPOINT`, `R2_BUCKET`, `R2_ACCESS_KEY`, `R2_SECRET_KEY`

