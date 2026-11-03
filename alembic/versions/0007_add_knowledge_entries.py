"""add knowledge entries：第 10 周 RAG 原子知识、来源与 pgvector 向量字段。

Revision ID: 0007
Revises: 0006
Create Date: 2026-11-03
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from app.db_types import Vector

revision: str = "0007"
down_revision: Union[str, None] = "0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """创建 RAG 知识条目表；embedding 在第 10 周第 2 步批量回填。"""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "knowledge_entries",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("knowledge_key", sa.String(length=100), nullable=False, unique=True),
        sa.Column("title", sa.String(length=300), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("category", sa.String(length=30), nullable=False),
        sa.Column("metric_codes", sa.ARRAY(sa.String(length=50)), nullable=True),
        sa.Column("source_name", sa.String(length=200), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("source_version", sa.String(length=100), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("embedding", Vector(512), nullable=True),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_knowledge_entries_category",
        "knowledge_entries",
        ["category"],
    )
    op.execute(
        "COMMENT ON TABLE knowledge_entries IS "
        "'RAG 知识条目（第 10 周，原子化演示摘要，带来源与语义检索向量）'"
    )
    op.execute("COMMENT ON COLUMN knowledge_entries.id IS '自增主键'")
    op.execute(
        "COMMENT ON COLUMN knowledge_entries.knowledge_key IS '稳定知识键（种子幂等更新用）'"
    )
    op.execute("COMMENT ON COLUMN knowledge_entries.title IS '知识条目标题'")
    op.execute(
        "COMMENT ON COLUMN knowledge_entries.content IS '原子知识正文（演示摘要，非诊疗依据）'"
    )
    op.execute(
        "COMMENT ON COLUMN knowledge_entries.category IS "
        "'知识分类：symptom/disease/association/lab/drug/safety'"
    )
    op.execute(
        "COMMENT ON COLUMN knowledge_entries.metric_codes IS "
        "'适用检验指标代码；空表示不限定指标'"
    )
    op.execute("COMMENT ON COLUMN knowledge_entries.source_name IS '来源名称'")
    op.execute("COMMENT ON COLUMN knowledge_entries.source_url IS '来源检索或原文地址'")
    op.execute("COMMENT ON COLUMN knowledge_entries.source_version IS '来源版本或采集日期'")
    op.execute(
        "COMMENT ON COLUMN knowledge_entries.content_hash IS "
        "'正文 SHA-256，内容变化时重建向量'"
    )
    op.execute(
        "COMMENT ON COLUMN knowledge_entries.embedding IS "
        "'512 维语义检索向量；种子入库后由第 10 周第 2 步回填'"
    )
    op.execute("COMMENT ON COLUMN knowledge_entries.updated_at IS '最后更新时间'")


def downgrade() -> None:
    """删除 RAG 知识条目表及其索引。"""
    op.drop_index("ix_knowledge_entries_category", table_name="knowledge_entries")
    op.drop_table("knowledge_entries")
