from __future__ import annotations

import json
from pathlib import Path

import boto3
from botocore.client import Config
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sqlalchemy import DateTime, Float, Integer, String, Text, func, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import WorkerSettings


class Base(DeclarativeBase):
    pass


class AnalysisEvent(Base):
    __tablename__ = "analysis_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    analysis_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    ai_probability: Mapped[float] = mapped_column(Float)
    human_score: Mapped[float] = mapped_column(Float)
    confidence_band: Mapped[str] = mapped_column(String(16))
    readability_grade: Mapped[float] = mapped_column(Float)
    complexity_score: Mapped[float] = mapped_column(Float)
    burstiness_score: Mapped[float] = mapped_column(Float)
    vocab_diversity_score: Mapped[float] = mapped_column(Float)
    word_count: Mapped[int] = mapped_column(Integer)
    estimated_read_time: Mapped[float] = mapped_column(Float)
    latency_ms: Mapped[float] = mapped_column(Float)
    metadata_json: Mapped[dict] = mapped_column(JSONB, default=dict)


class ReportJob(Base):
    __tablename__ = "report_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    report_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    analysis_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(16), default="queued")
    format: Mapped[str] = mapped_column(String(16), default="pdf_json")
    json_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    pdf_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


settings = WorkerSettings()
engine = create_async_engine(settings.database_url, future=True, pool_pre_ping=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Storage:
    def __init__(self) -> None:
        self.local_root = Path(settings.report_local_dir).resolve()
        self.local_root.mkdir(parents=True, exist_ok=True)

    @property
    def s3_enabled(self) -> bool:
        return all(
            [
                settings.r2_endpoint,
                settings.r2_bucket,
                settings.r2_access_key,
                settings.r2_secret_key,
            ]
        )

    def _client(self):
        return boto3.client(
            "s3",
            endpoint_url=settings.r2_endpoint,
            region_name=settings.r2_region,
            aws_access_key_id=settings.r2_access_key,
            aws_secret_access_key=settings.r2_secret_key,
            config=Config(signature_version="s3v4"),
        )

    def upload(self, report_id: str, filename: str, data_path: Path) -> str:
        if not self.s3_enabled:
            target = self.local_root / report_id / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data_path.read_bytes())
            return f"{settings.public_base_url}/v1/reports/files/{report_id}/{filename}"

        key = f"reports/{report_id}/{filename}"
        client = self._client()
        client.upload_file(str(data_path), settings.r2_bucket, key)
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.r2_bucket, "Key": key},
            ExpiresIn=3600,
        )


storage = Storage()


def render_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def render_pdf(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    y = height - 50

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(40, y, "CIDA Detection Report")
    y -= 30
    pdf.setFont("Helvetica", 10)

    for key, value in payload.items():
        pdf.drawString(40, y, f"{key}: {value}")
        y -= 14
        if y < 80:
            pdf.showPage()
            y = height - 50

    pdf.save()


async def generate_report(ctx, report_id: str, analysis_id: str) -> dict:
    async with SessionLocal() as db:
        report = await db.scalar(select(ReportJob).where(ReportJob.report_id == report_id))
        if report is None:
            return {"status": "missing_report"}

        report.status = "processing"
        await db.commit()

        analysis = await db.scalar(select(AnalysisEvent).where(AnalysisEvent.analysis_id == analysis_id))
        if analysis is None:
            report.status = "failed"
            report.error_message = "Analysis missing"
            await db.commit()
            return {"status": "failed", "reason": "analysis_missing"}

        payload = {
            "analysis_id": analysis.analysis_id,
            "ai_probability": analysis.ai_probability,
            "human_score": analysis.human_score,
            "confidence_band": analysis.confidence_band,
            "readability_grade": analysis.readability_grade,
            "complexity_score": analysis.complexity_score,
            "burstiness_score": analysis.burstiness_score,
            "vocab_diversity_score": analysis.vocab_diversity_score,
            "word_count": analysis.word_count,
            "estimated_read_time": analysis.estimated_read_time,
            "latency_ms": analysis.latency_ms,
        }

        tmp_root = Path("./tmp_reports") / report_id
        tmp_root.mkdir(parents=True, exist_ok=True)

        json_path = tmp_root / "report.json"
        pdf_path = tmp_root / "report.pdf"
        render_json(payload, json_path)
        render_pdf(payload, pdf_path)

        report.json_url = storage.upload(report_id, "report.json", json_path)
        report.pdf_url = storage.upload(report_id, "report.pdf", pdf_path)
        report.status = "ready"
        report.error_message = None
        await db.commit()

        return {"status": "ready", "report_id": report_id}

