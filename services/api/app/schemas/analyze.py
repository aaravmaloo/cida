from typing import Literal

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    text: str | None = None
    source: str = Field(default="paste", pattern="^(paste|upload)$")
    return_report: bool = False


class ReadabilityMetrics(BaseModel):
    flesch_reading_ease: float
    flesch_kincaid_grade: float
    smog_index: float


class AnalyzeResponse(BaseModel):
    analysis_id: str
    ai_probability: float
    human_score: float
    predicted_label: Literal["AI", "Human"]
    confidence_band: str
    readability: ReadabilityMetrics
    complexity: float
    burstiness: float
    vocab_diversity: float
    word_count: int
    estimated_read_time: float
    model_version: str
    latency_ms: float

