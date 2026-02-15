from __future__ import annotations

import uuid
from pathlib import Path
from urllib.parse import urlparse

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.rate_limit import enforce_sliding_window
from app.core.redis import get_redis
from app.db.session import get_db
from app.models.entities import AnalysisEvent, ReportJob
from app.schemas.report import ReportCreateResponse, ReportStatusResponse
from app.services.turnstile import verify_turnstile
from app.utils.http import client_ip

router = APIRouter()


class CreateReportRequest(BaseModel):
    analysis_id: str


@router.post("/reports", response_model=ReportCreateResponse)
async def create_report(
    request: Request,
    body: CreateReportRequest,
    db: AsyncSession = Depends(get_db),
):
    redis = await get_redis()
    ip = client_ip(request)
    rl = await enforce_sliding_window(redis, key=f"rl:report:{ip}", limit=10, window_seconds=60)
    if not rl.allowed:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Report rate limit exceeded")

    turnstile_token = request.headers.get("x-turnstile-token")
    if not await verify_turnstile(turnstile_token, ip):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Turnstile validation failed")

    analysis = await db.scalar(select(AnalysisEvent).where(AnalysisEvent.analysis_id == body.analysis_id))
    if analysis is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")

    report_id = uuid.uuid4().hex
    job = ReportJob(report_id=report_id, analysis_id=body.analysis_id, status="queued", format="pdf_json")
    db.add(job)
    await db.commit()

    settings = get_settings()
    parsed = urlparse(settings.redis_url)
    redis_settings = RedisSettings(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        database=int((parsed.path or "/0").strip("/")),
        password=parsed.password,
        ssl=parsed.scheme == "rediss",
    )
    pool = await create_pool(redis_settings)
    await pool.enqueue_job("generate_report", report_id=report_id, analysis_id=body.analysis_id)
    await pool.aclose()

    return ReportCreateResponse(report_id=report_id, status="queued")


@router.get("/reports/{report_id}", response_model=ReportStatusResponse)
async def report_status(report_id: str, db: AsyncSession = Depends(get_db)):
    job = await db.scalar(select(ReportJob).where(ReportJob.report_id == report_id))
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")

    return ReportStatusResponse(
        report_id=job.report_id,
        status=job.status,
        json_url=job.json_url,
        pdf_url=job.pdf_url,
        error_message=job.error_message,
        updated_at=job.updated_at,
    )


@router.get("/reports/files/{report_id}/{filename}")
async def local_report_file(report_id: str, filename: str):
    settings = get_settings()
    file_path = Path(settings.report_local_dir).resolve() / report_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report artifact not found")
    return FileResponse(str(file_path))

