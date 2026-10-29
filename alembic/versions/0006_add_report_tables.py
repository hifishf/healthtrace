"""add report tables：检验参考范围 + 药物交互规则（第 9 周报告解读/用药冲突）。

第 9 周数据契约（冻结）：
- lab_reference_ranges（metric_code / unit / low / high / age_group / sex）
- drug_interactions（drug_a / drug_b / severity / description）

均为关系表，走 Alembic 迁移；字段注释与 app/models.py 保持一致，`alembic check` 无漂移。

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-29
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1) 检验参考范围：同指标按单位/年龄组/性别给参考区间，用于趋势 vs 参考范围对比
    op.create_table(
        "lab_reference_ranges",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("metric_code", sa.String(length=50), nullable=False),
        sa.Column("unit", sa.String(length=20), nullable=False),
        sa.Column("low", sa.Numeric(), nullable=True),
        sa.Column("high", sa.Numeric(), nullable=True),
        sa.Column("age_group", sa.String(length=20), nullable=True),
        sa.Column("sex", sa.String(length=10), nullable=True),
    )
    op.create_index(
        "ix_lab_reference_ranges_metric_code",
        "lab_reference_ranges",
        ["metric_code"],
    )
    op.execute("COMMENT ON TABLE lab_reference_ranges IS '检验指标参考范围（按指标代码/单位/年龄组/性别给参考区间）'")
    op.execute("COMMENT ON COLUMN lab_reference_ranges.id IS '自增主键'")
    op.execute("COMMENT ON COLUMN lab_reference_ranges.metric_code IS '指标代码（如 ldl_cholesterol）'")
    op.execute("COMMENT ON COLUMN lab_reference_ranges.unit IS '单位（如 mmol/L）'")
    op.execute("COMMENT ON COLUMN lab_reference_ranges.low IS '参考范围下限（可空，如仅上限约束）'")
    op.execute("COMMENT ON COLUMN lab_reference_ranges.high IS '参考范围上限（可空，如仅下限约束）'")
    op.execute("COMMENT ON COLUMN lab_reference_ranges.age_group IS '适用年龄组（如 adult，空=不限）'")
    op.execute("COMMENT ON COLUMN lab_reference_ranges.sex IS '适用性别（male/female，空=不限）'")

    # 2) 药物交互规则：确定性规则表 + 风险分级（禁忌/严重/中等/轻微）
    op.create_table(
        "drug_interactions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("drug_a", sa.String(length=200), nullable=False),
        sa.Column("drug_b", sa.String(length=200), nullable=False),
        sa.Column("severity", sa.String(length=20), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.UniqueConstraint("drug_a", "drug_b", name="uq_drug_interactions_drug_a_drug_b"),
    )
    op.execute("COMMENT ON TABLE drug_interactions IS '药物交互规则表（确定性规则 + 风险分级：禁忌/严重/中等/轻微）'")
    op.execute("COMMENT ON COLUMN drug_interactions.id IS '自增主键'")
    op.execute("COMMENT ON COLUMN drug_interactions.drug_a IS '药物 A 名称'")
    op.execute("COMMENT ON COLUMN drug_interactions.drug_b IS '药物 B 名称'")
    op.execute("COMMENT ON COLUMN drug_interactions.severity IS '风险等级：禁忌/严重/中等/轻微'")
    op.execute("COMMENT ON COLUMN drug_interactions.description IS '交互说明（含来源标注）'")


def downgrade() -> None:
    op.drop_table("drug_interactions")
    op.drop_index("ix_lab_reference_ranges_metric_code", table_name="lab_reference_ranges")
    op.drop_table("lab_reference_ranges")
