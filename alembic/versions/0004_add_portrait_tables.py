"""add portrait tables：画像扩到 30+ 语义字段（病情/过敏/偏好/待跟进）。

第 7 周语义记忆完整化：在既有 health_profiles（基础）与 medications（用药）之上，
新增 conditions（病情）/ allergies（过敏）/ preferences（偏好）/ followups（待跟进）四张子表，
与第 8 节数据模型保持一致，均为关系表、按 user_id 检索。

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-26
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1) 病情：用户已确诊/已知的健康问题（与知识库 diseases 表不同，这是用户画像）
    op.create_table(
        "conditions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("condition_name", sa.String(length=200), nullable=False),
        sa.Column("diagnosed_date", sa.Date(), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
    )
    op.create_index("ix_conditions_user_id", "conditions", ["user_id"])
    op.execute("COMMENT ON TABLE conditions IS '病情（用户已知健康问题，画像语义记忆；与知识库 diseases 表不同）'")
    op.execute("COMMENT ON COLUMN conditions.id IS '自增主键'")
    op.execute("COMMENT ON COLUMN conditions.user_id IS '用户唯一标识'")
    op.execute("COMMENT ON COLUMN conditions.condition_name IS '病情名称（如 高血压、2型糖尿病）'")
    op.execute("COMMENT ON COLUMN conditions.diagnosed_date IS '确诊日期'")
    op.execute("COMMENT ON COLUMN conditions.status IS '状态（active在患/remission缓解/resolved已愈）'")
    op.execute("COMMENT ON COLUMN conditions.notes IS '备注'")

    # 2) 过敏：用户过敏原与反应，跨会话保留（避免重复询问）
    op.create_table(
        "allergies",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("allergen", sa.String(length=200), nullable=False),
        sa.Column("severity", sa.String(length=20), nullable=True),
        sa.Column("reaction", sa.String(length=200), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
    )
    op.create_index("ix_allergies_user_id", "allergies", ["user_id"])
    op.execute("COMMENT ON TABLE allergies IS '过敏（用户过敏原画像，语义记忆）'")
    op.execute("COMMENT ON COLUMN allergies.id IS '自增主键'")
    op.execute("COMMENT ON COLUMN allergies.user_id IS '用户唯一标识'")
    op.execute("COMMENT ON COLUMN allergies.allergen IS '过敏原（如 青霉素、花生）'")
    op.execute("COMMENT ON COLUMN allergies.severity IS '严重程度（mild/moderate/severe）'")
    op.execute("COMMENT ON COLUMN allergies.reaction IS '过敏反应表现（如 皮疹、呼吸困难）'")
    op.execute("COMMENT ON COLUMN allergies.notes IS '备注'")

    # 3) 偏好：用户的长期偏好（沟通风格、饮食等），画像注入用
    op.create_table(
        "preferences",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("preference_key", sa.String(length=100), nullable=False),
        sa.Column("preference_value", sa.String(length=200), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
    )
    op.create_index("ix_preferences_user_id", "preferences", ["user_id"])
    op.execute("COMMENT ON TABLE preferences IS '偏好（用户长期偏好，画像语义记忆）'")
    op.execute("COMMENT ON COLUMN preferences.id IS '自增主键'")
    op.execute("COMMENT ON COLUMN preferences.user_id IS '用户唯一标识'")
    op.execute("COMMENT ON COLUMN preferences.preference_key IS '偏好键（如 沟通风格、饮食偏好）'")
    op.execute("COMMENT ON COLUMN preferences.preference_value IS '偏好值'")
    op.execute("COMMENT ON COLUMN preferences.notes IS '备注'")

    # 4) 待跟进事项：跨会话要记住的后续动作（复诊、复查等）
    op.create_table(
        "followups",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("item", sa.String(length=200), nullable=False),
        sa.Column("due_date", sa.Date(), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
    )
    op.create_index("ix_followups_user_id", "followups", ["user_id"])
    op.execute("COMMENT ON TABLE followups IS '待跟进事项（用户后续动作，画像语义记忆）'")
    op.execute("COMMENT ON COLUMN followups.id IS '自增主键'")
    op.execute("COMMENT ON COLUMN followups.user_id IS '用户唯一标识'")
    op.execute("COMMENT ON COLUMN followups.item IS '待跟进事项内容'")
    op.execute("COMMENT ON COLUMN followups.due_date IS '计划跟进日期'")
    op.execute("COMMENT ON COLUMN followups.status IS '状态（pending待处理/done已完成）'")
    op.execute("COMMENT ON COLUMN followups.notes IS '备注'")


def downgrade() -> None:
    op.drop_index("ix_followups_user_id", table_name="followups")
    op.drop_table("followups")
    op.drop_index("ix_preferences_user_id", table_name="preferences")
    op.drop_table("preferences")
    op.drop_index("ix_allergies_user_id", table_name="allergies")
    op.drop_table("allergies")
    op.drop_index("ix_conditions_user_id", table_name="conditions")
    op.drop_table("conditions")
