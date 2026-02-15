# Deployment Guide

## 1. Prerequisites

1. Create accounts and projects on Railway and Vercel.
2. Connect this repository to both platforms.
3. Make sure the repository root is used as the project root.

## 2. Deploy Backend on Railway

1. Create Railway services:
   - `postgres`
   - `redis`
   - `api`
   - `worker`
2. Configure `api` service:
   - Dockerfile: `infra/docker/Dockerfile.api`
   - Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
3. Configure `worker` service:
   - Dockerfile: `infra/docker/Dockerfile.worker`
   - Start command: `arq app.worker.WorkerSettingsConfig`
4. Add required environment variables to both backend services:
   - `DATABASE_URL`
   - `REDIS_URL`
   - `JWT_SECRET`
   - `ADMIN_PASSKEY_HASH`
   - `CORS_ALLOWED_ORIGINS`
5. Optional object storage variables (if using R2):
   - `R2_ENDPOINT`
   - `R2_BUCKET`
   - `R2_ACCESS_KEY`
   - `R2_SECRET_KEY`
6. Deploy and copy the API public URL (for example, `https://your-api.up.railway.app`).

## 3. Deploy Frontend on Vercel

1. Create a Vercel project connected to this repository.
2. Ensure Vercel uses this config at `apps/web/vercel.json`:
   - `framework`: `nextjs`
   - `installCommand`: `npm install`
   - `buildCommand`: `npm run build --workspace @cida/web`
3. Set environment variable in Vercel:
   - `NEXT_PUBLIC_API_BASE_URL=https://<your-railway-api-url>`
4. Deploy the project.

## 4. Post-Deployment Checks

1. Open the frontend URL and verify pages load.
2. Test API health and core endpoints from the deployed app.
3. Confirm CORS is correctly set for the Vercel domain in `CORS_ALLOWED_ORIGINS`.
