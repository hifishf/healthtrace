"""add episodic memories：统一会话/情景记忆表（chat/triage/insight/report 都记）。

第 7 周把文档第八节的 triage_sessions 泛化为统一情景记忆表：每个子图的每次交互
（用户原文、摘要、triage 专属字段、结构化 payload）都落一条，按 user_id 可检索、可删除。

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-26
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "episodic_memories",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("subgraph", sa.String(length=20), nullable=False),
        sa.Column("session_id", sa.String(length=100), nullable=True),
        sa.Column("user_text", sa.Text(), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("symptom_text", sa.Text(), nullable=True),
        sa.Column("final_level", sa.String(length=20), nullable=True),
        sa.Column("red_flag_triggered", sa.Boolean(), nullable=True),
        sa.Column("payload", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_episodic_memories_user_id", "episodic_memories", ["user_id"])
    op.execute("COMMENT ON TABLE episodic_memories IS '统一会话/情景记忆（chat/triage/insight/report 每次交互一条）'")
    op.execute("COMMENT ON COLUMN episodic_memories.id IS '自增主键'")
    op.execute("COMMENT ON COLUMN episodic_memories.user_id IS '用户唯一标识'")
    op.execute("COMMENT ON COLUMN episodic_memories.subgraph IS '子图类型：chat/triage/insight/report'")
    op.execute("COMMENT ON COLUMN episodic_memories.session_id IS '会话标识（triage 的 thread_id，chat 等无则空）'")
    op.execute("COMMENT ON COLUMN episodic_memories.user_text IS '用户原始输入'")
    op.execute("COMMENT ON COLUMN episodic_memories.summary IS '本轮回复/追问/建议摘要'")
    op.execute("COMMENT ON COLUMN episodic_memories.symptom_text IS '已确认症状文本（triage 专用）'")
    op.execute("COMMENT ON COLUMN episodic_memories.final_level IS '分级建议等级（triage 专用）'")
    op.execute("COMMENT ON COLUMN episodic_memories.red_flag_triggered IS '是否触发红旗（triage 专用）'")
    op.execute("COMMENT ON COLUMN episodic_memories.payload IS '子图专属结构化结果（JSONB）'")
    op.execute("COMMENT ON COLUMN episodic_memories.created_at IS '创建时间'")


def downgrade() -> None:
    op.drop_index("ix_episodic_memories_user_id", table_name="episodic_memories")
    op.drop_table("episodic_memories")
