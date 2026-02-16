from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.rate_limit import enforce_sliding_window
from app.core.redis import get_redis
from app.db.session import get_db
from app.models.entities import HumanizeEvent
from app.schemas.humanize import HumanizeRequest, HumanizeResponse
from app.services.humanizer import humanizer_service
from app.services.inference import detector_service
from app.services.turnstile import verify_turnstile
from app.utils.hashing import sha256_text
from app.utils.http import client_ip

router = APIRouter()


@router.post("/humanize", response_model=HumanizeResponse)
async def humanize_content(
    request: Request,
    body: HumanizeRequest,
    db: AsyncSession = Depends(get_db),
):
    settings = get_settings()
    redis = await get_redis()
    ip = client_ip(request)
    rl = await enforce_sliding_window(redis, key=f"rl:humanize:{ip}", limit=6, window_seconds=60)
    if not rl.allowed:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Humanize rate limit exceeded")

    turnstile_token = request.headers.get("x-turnstile-token")
    if not await verify_turnstile(turnstile_token, ip):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Turnstile validation failed")

    if settings.release_detector_for_humanizer:
        detector_service.release_model()

    start = time.perf_counter()
    try:
        result = humanizer_service.rewrite(
            text=body.text,
            style=body.style,
            strength=body.strength,
            preserve_terms=body.preserve_terms,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    latency_ms = round((time.perf_counter() - start) * 1000, 3)
    humanize_id = uuid.uuid4().hex

    event = HumanizeEvent(
        humanize_id=humanize_id,
        source_hash=sha256_text(body.text),
        style=body.style,
        strength=body.strength,
        input_word_count=result["input_word_count"],
        output_word_count=result["output_word_count"],
        readability_delta=result["readability_delta"],
        latency_ms=latency_ms,
        metadata_json={
            "quality_flags": result["quality_flags"],
            "source_ip": ip,
            "humanizer_mode": result["humanizer_mode"],
            "humanizer_model": result["humanizer_model"],
        },
    )
    db.add(event)
    await db.commit()

    return HumanizeResponse(
        humanize_id=humanize_id,
        rewritten_text=result["rewritten_text"],
        diff_stats=result["diff_stats"],
        readability_delta=result["readability_delta"],
        quality_flags=result["quality_flags"],
        humanizer_mode=result["humanizer_mode"],
        humanizer_model=result["humanizer_model"],
        latency_ms=latency_ms,
    )

