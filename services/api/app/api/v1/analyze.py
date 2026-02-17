from __future__ import annotations

import json
import time
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.rate_limit import enforce_sliding_window
from app.core.redis import get_redis
from app.db.session import get_db
from app.models.entities import AnalysisEvent
from app.schemas.analyze import AnalyzeResponse
from app.services.inference import detector_service
from app.services.turnstile import verify_turnstile
from app.utils.files import extract_text_from_upload
from app.utils.hashing import sha256_text
from app.utils.http import client_ip
from app.utils.request_body import read_json_body
from app.utils.text import normalize_text

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_content(
    request: Request,
    db: AsyncSession = Depends(get_db),
    file: UploadFile | None = File(default=None),
    text_form: str | None = Form(default=None, alias="text"),
    source_form: str | None = Form(default="paste", alias="source"),
):
    settings = get_settings()
    redis = await get_redis()
    ip = client_ip(request)

    rl = await enforce_sliding_window(redis, key=f"rl:analyze:{ip}", limit=20, window_seconds=60)
    if not rl.allowed:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Analyze rate limit exceeded")

    turnstile_token = request.headers.get("x-turnstile-token")
    if not await verify_turnstile(turnstile_token, ip):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Turnstile validation failed")

    start = time.perf_counter()
    text: str | None = None
    source = "paste"

    content_type = (request.headers.get("content-type") or "").lower()
    if "application/json" in content_type:
        payload = await read_json_body(request)
        text = payload.get("text")
        source = payload.get("source", "paste")
    elif "multipart/form-data" in content_type:
        if file is not None:
            text = await extract_text_from_upload(file, settings.max_upload_bytes)
            source = "upload"
        elif text_form:
            text = text_form
            source = source_form or "paste"
    else:
        text = text_form

    if not text:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="text or file is required")

    normalized = normalize_text(text)
    text_hash = sha256_text(normalized)
    cache_key = f"analysis:{text_hash}:{settings.hf_model}"

    cached = await redis.get(cache_key)
    if cached:
        payload = json.loads(cached)
        payload["analysis_id"] = uuid.uuid4().hex
        payload["latency_ms"] = round((time.perf_counter() - start) * 1000, 3)
    else:
        payload = detector_service.analyze(normalized)
        payload["analysis_id"] = uuid.uuid4().hex
        payload["latency_ms"] = round((time.perf_counter() - start) * 1000, 3)
        await redis.setex(cache_key, settings.cache_ttl_seconds, json.dumps(payload, ensure_ascii=True))

    event = AnalysisEvent(
        analysis_id=payload["analysis_id"],
        text_hash=text_hash,
        source=source,
        ai_probability=payload["ai_probability"],
        human_score=payload["human_score"],
        confidence_band=payload["confidence_band"],
        readability_grade=payload["readability"]["flesch_kincaid_grade"],
        complexity_score=payload["complexity"],
        burstiness_score=payload["burstiness"],
        vocab_diversity_score=payload["vocab_diversity"],
        word_count=payload["word_count"],
        estimated_read_time=payload["estimated_read_time"],
        latency_ms=payload["latency_ms"],
        metadata_json={"source_ip": ip, "cached": bool(cached)},
    )
    db.add(event)
    await db.commit()

    return AnalyzeResponse(**payload)

