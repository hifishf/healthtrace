"""add knowledge embedding index：为 512 维余弦检索增加 HNSW 索引。

Revision ID: 0008
Revises: 0007
Create Date: 2026-11-03
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0008"
down_revision: Union[str, None] = "0007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """创建知识向量余弦距离索引，供近似最近邻检索使用。"""
    op.execute(
        "CREATE INDEX ix_knowledge_entries_embedding_hnsw "
        "ON knowledge_entries USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    """删除知识向量索引。"""
    op.drop_index(
        "ix_knowledge_entries_embedding_hnsw",
        table_name="knowledge_entries",
    )
