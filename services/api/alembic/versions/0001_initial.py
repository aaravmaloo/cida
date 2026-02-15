"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-02-15 13:40:00
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "analysis_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("analysis_id", sa.String(length=64), nullable=False),
        sa.Column("text_hash", sa.String(length=64), nullable=False),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("ai_probability", sa.Float(), nullable=False),
        sa.Column("human_score", sa.Float(), nullable=False),
        sa.Column("confidence_band", sa.String(length=16), nullable=False),
        sa.Column("readability_grade", sa.Float(), nullable=False),
        sa.Column("complexity_score", sa.Float(), nullable=False),
        sa.Column("burstiness_score", sa.Float(), nullable=False),
        sa.Column("vocab_diversity_score", sa.Float(), nullable=False),
        sa.Column("word_count", sa.Integer(), nullable=False),
        sa.Column("estimated_read_time", sa.Float(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_analysis_events_analysis_id", "analysis_events", ["analysis_id"], unique=True)
    op.create_index("ix_analysis_events_text_hash", "analysis_events", ["text_hash"], unique=False)

    op.create_table(
        "humanize_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("humanize_id", sa.String(length=64), nullable=False),
        sa.Column("source_hash", sa.String(length=64), nullable=False),
        sa.Column("style", sa.String(length=24), nullable=False),
        sa.Column("strength", sa.Integer(), nullable=False),
        sa.Column("input_word_count", sa.Integer(), nullable=False),
        sa.Column("output_word_count", sa.Integer(), nullable=False),
        sa.Column("readability_delta", sa.Float(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_humanize_events_humanize_id", "humanize_events", ["humanize_id"], unique=True)
    op.create_index("ix_humanize_events_source_hash", "humanize_events", ["source_hash"], unique=False)

    op.create_table(
        "report_jobs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("report_id", sa.String(length=64), nullable=False),
        sa.Column("analysis_id", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("format", sa.String(length=16), nullable=False),
        sa.Column("json_url", sa.Text(), nullable=True),
        sa.Column("pdf_url", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_report_jobs_report_id", "report_jobs", ["report_id"], unique=True)
    op.create_index("ix_report_jobs_analysis_id", "report_jobs", ["analysis_id"], unique=False)

    op.create_table(
        "usage_buckets",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("bucket_key", sa.String(length=128), nullable=False),
        sa.Column("bucket_type", sa.String(length=32), nullable=False),
        sa.Column("count", sa.Integer(), nullable=False),
        sa.Column("period_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("period_end", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_usage_buckets_bucket_key", "usage_buckets", ["bucket_key"], unique=True)

    op.create_table(
        "admin_audit_logs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.Column("ip_address", sa.String(length=64), nullable=False),
        sa.Column("user_agent", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "model_registry",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("artifact_uri", sa.Text(), nullable=False),
        sa.Column("metrics_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_model_registry_model_version", "model_registry", ["model_version"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_model_registry_model_version", table_name="model_registry")
    op.drop_table("model_registry")
    op.drop_table("admin_audit_logs")
    op.drop_index("ix_usage_buckets_bucket_key", table_name="usage_buckets")
    op.drop_table("usage_buckets")
    op.drop_index("ix_report_jobs_analysis_id", table_name="report_jobs")
    op.drop_index("ix_report_jobs_report_id", table_name="report_jobs")
    op.drop_table("report_jobs")
    op.drop_index("ix_humanize_events_source_hash", table_name="humanize_events")
    op.drop_index("ix_humanize_events_humanize_id", table_name="humanize_events")
    op.drop_table("humanize_events")
    op.drop_index("ix_analysis_events_text_hash", table_name="analysis_events")
    op.drop_index("ix_analysis_events_analysis_id", table_name="analysis_events")
    op.drop_table("analysis_events")

