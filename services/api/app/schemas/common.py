from datetime import datetime

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    detail: str
    trace_id: str


class MetadataEnvelope(BaseModel):
    model_version: str
    generated_at: datetime


class HealthResponse(BaseModel):
    status: str = "ok"

