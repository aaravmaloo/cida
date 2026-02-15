from datetime import datetime

from pydantic import BaseModel


class ReportCreateResponse(BaseModel):
    report_id: str
    status: str


class ReportStatusResponse(BaseModel):
    report_id: str
    status: str
    json_url: str | None = None
    pdf_url: str | None = None
    error_message: str | None = None
    updated_at: datetime | None = None

