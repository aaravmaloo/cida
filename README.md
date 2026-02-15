# CIDA v2 Monorepo

Production-grade AI content detection and writing-assist platform.

## Stack

- Backend: FastAPI + PostgreSQL + Redis + ARQ (Railway)
- Frontend: Next.js 15 + TypeScript + Tailwind + Recharts (Vercel)
- Training: PyTorch + HuggingFace Transformers + ONNX Runtime
- Reports: JSON + PDF export

## Monorepo Layout

- `apps/web` - Vercel frontend
- `services/api` - FastAPI backend
- `services/worker` - background jobs
- `services/trainer` - model training + calibration + ONNX export
- `packages/shared-schemas` - shared API types
- `infra` - Docker/Railway/Vercel configs

## Quick Start

1. Backend dependencies:
   - `python -m venv .venv`
   - `. .venv/Scripts/Activate.ps1`
   - `pip install -r services/api/requirements.txt`
2. Frontend dependencies:
   - `npm install`
3. Local stack:
   - `docker compose -f infra/docker/docker-compose.yml up -d`
4. Run backend:
   - `uvicorn app.main:app --reload --port 8000` from `services/api`
5. Run frontend:
   - `npm run dev:web`

## Deployment

- Backend: Railway project with services `api`, `worker`, `postgres`, `redis`
- Frontend: Vercel project from `apps/web`

## Security Defaults

- No raw text retention by default
- Public API with strict per-IP rate limits
- Admin-only passkey auth for dashboard endpoints

