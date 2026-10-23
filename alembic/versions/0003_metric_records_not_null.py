"""metric_records 必填列收紧为 NOT NULL：对齐 ORM 模型的既有意图

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-23

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 必填列收紧 NOT NULL：写入路径始终提供这些字段，库里无 NULL 行
    op.alter_column(
        "metric_records", "user_id",
        existing_type=postgresql.UUID(as_uuid=True), nullable=False,
    )
    op.alter_column(
        "metric_records", "metric_code",
        existing_type=sa.String(length=50), nullable=False,
    )
    op.alter_column(
        "metric_records", "value",
        existing_type=sa.Numeric(), nullable=False,
    )
    op.alter_column(
        "metric_records", "unit",
        existing_type=sa.String(length=20), nullable=False,
    )
    op.alter_column(
        "metric_records", "recorded_date",
        existing_type=sa.Date(), nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "metric_records", "user_id",
        existing_type=postgresql.UUID(as_uuid=True), nullable=True,
    )
    op.alter_column(
        "metric_records", "metric_code",
        existing_type=sa.String(length=50), nullable=True,
    )
    op.alter_column(
        "metric_records", "value",
        existing_type=sa.Numeric(), nullable=True,
    )
    op.alter_column(
        "metric_records", "unit",
        existing_type=sa.String(length=20), nullable=True,
    )
    op.alter_column(
        "metric_records", "recorded_date",
        existing_type=sa.Date(), nullable=True,
    )
