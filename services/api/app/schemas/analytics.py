from pydantic import BaseModel


class AnalyticsSummaryResponse(BaseModel):
    total_analyses: int
    total_humanizations: int
    avg_ai_probability: float
    p95_latency_ms: float
    confidence_distribution: dict[str, int]
    abuse_block_count: int

