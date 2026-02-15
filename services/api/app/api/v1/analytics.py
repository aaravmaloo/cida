from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import require_admin
from app.db.session import get_db
from app.models.entities import AnalysisEvent, HumanizeEvent
from app.schemas.analytics import AnalyticsSummaryResponse

router = APIRouter()


@router.get("/analytics/summary", response_model=AnalyticsSummaryResponse)
async def analytics_summary(
    _admin=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    total_analyses = int((await db.scalar(select(func.count()).select_from(AnalysisEvent))) or 0)
    total_humanizations = int((await db.scalar(select(func.count()).select_from(HumanizeEvent))) or 0)
    avg_ai_probability = float((await db.scalar(select(func.avg(AnalysisEvent.ai_probability)))) or 0.0)

    latencies = (await db.scalars(select(AnalysisEvent.latency_ms))).all()
    p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0

    rows = (
        await db.execute(
            select(AnalysisEvent.confidence_band, func.count()).group_by(AnalysisEvent.confidence_band)
        )
    ).all()
    dist = {row[0]: int(row[1]) for row in rows}
    for key in ("low", "medium", "high"):
        dist.setdefault(key, 0)

    return AnalyticsSummaryResponse(
        total_analyses=total_analyses,
        total_humanizations=total_humanizations,
        avg_ai_probability=round(avg_ai_probability, 6),
        p95_latency_ms=round(p95_latency, 3),
        confidence_distribution=dist,
        abuse_block_count=0,
    )

