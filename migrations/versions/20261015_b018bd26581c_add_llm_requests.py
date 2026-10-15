"""add llm requests

One row per LLM request of the daily pipeline, for monitoring latency, failures and rerouting.

Revision ID: b018bd26581c
Revises: 48d7e6b888dd
Create Date: 2026-10-15 23:35:30.856303

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b018bd26581c"
down_revision: str | Sequence[str] | None = "48d7e6b888dd"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "llm_requests",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("job_run_id", sa.Integer(), nullable=True),
        sa.Column("video_id", sa.Integer(), nullable=True),
        sa.Column("model", sa.String(length=200), nullable=False),
        sa.Column("answered_by", sa.String(length=200), nullable=True),
        sa.Column("provider", sa.String(length=100), nullable=True),
        sa.Column("result", sa.String(length=32), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "result IN ('ok', 'timeout', 'rate_limited', 'invalid', 'error')",
            name=op.f("ck_llm_requests_llmresult"),
        ),
        sa.ForeignKeyConstraint(
            ["job_run_id"],
            ["job_runs.id"],
            name=op.f("fk_llm_requests_job_run_id_job_runs"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["video_id"], ["videos.id"], name=op.f("fk_llm_requests_video_id_videos"), ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_llm_requests")),
    )
    op.create_index(op.f("ix_llm_requests_job_run_id"), "llm_requests", ["job_run_id"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_llm_requests_job_run_id"), table_name="llm_requests")
    op.drop_table("llm_requests")
