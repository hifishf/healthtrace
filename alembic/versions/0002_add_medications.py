"""add medications：最小化用药清单（第 6 周迁移验收用，第 9 周扩展交互检查）

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-23

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 用药清单：字段与文档第八节保持一致，第 9 周在此之上做交互规则引擎
    op.create_table(
        "medications",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("dosage", sa.String(length=100), nullable=True),
        sa.Column("frequency", sa.String(length=100), nullable=True),
        sa.Column("start_date", sa.Date(), nullable=True),
        sa.Column("end_date", sa.Date(), nullable=True),
        sa.Column("active", sa.Boolean(), server_default=sa.text("true"), nullable=True),
    )
    op.create_index("ix_medications_user_id", "medications", ["user_id"])
    op.execute("COMMENT ON TABLE medications IS '用药清单（第 6 周最小化，第 9 周扩展交互检查）'")
    op.execute("COMMENT ON COLUMN medications.id IS '自增主键'")
    op.execute("COMMENT ON COLUMN medications.user_id IS '用户唯一标识'")
    op.execute("COMMENT ON COLUMN medications.name IS '药物/保健品名称'")
    op.execute("COMMENT ON COLUMN medications.dosage IS '剂量（如 500mg/次）'")
    op.execute("COMMENT ON COLUMN medications.frequency IS '频次（如 每日两次）'")
    op.execute("COMMENT ON COLUMN medications.start_date IS '开始服用日期'")
    op.execute("COMMENT ON COLUMN medications.end_date IS '停止服用日期（空=仍在服用）'")
    op.execute("COMMENT ON COLUMN medications.active IS '是否在服，默认 true'")


def downgrade() -> None:
    op.drop_index("ix_medications_user_id", table_name="medications")
    op.drop_table("medications")
