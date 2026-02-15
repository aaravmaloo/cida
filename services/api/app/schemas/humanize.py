from pydantic import BaseModel, Field


class HumanizeRequest(BaseModel):
    text: str
    style: str = Field(default="natural", pattern="^(natural|formal|concise)$")
    strength: int = Field(default=2, ge=1, le=3)
    preserve_terms: list[str] = Field(default_factory=list)


class DiffStats(BaseModel):
    changed_tokens: int
    change_ratio: float


class HumanizeResponse(BaseModel):
    humanize_id: str
    rewritten_text: str
    diff_stats: DiffStats
    readability_delta: float
    quality_flags: list[str]
    latency_ms: float

