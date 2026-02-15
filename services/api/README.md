# API Service

## Run locally

1. Create venv and install dependencies:
   - `python -m venv .venv`
   - `. .venv/Scripts/Activate.ps1`
   - `pip install -r requirements.txt`
   - For tests only: `pip install -r requirements-dev.txt`
   - Optional full humanizer model backend: `pip install -r requirements-torch-cpu.txt`
2. Copy `.env.example` to `.env` and set secrets.
3. Run migrations:
   - `alembic upgrade head`
4. Start server:
   - `uvicorn app.main:app --reload --port 8000`

## Core endpoints

- `POST /v1/analyze`
- `POST /v1/humanize`
- `POST /v1/reports`
- `GET /v1/reports/{report_id}`
- `POST /v1/admin/login`
- `GET /v1/analytics/summary` (admin token required)
