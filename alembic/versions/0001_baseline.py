"""baseline：接管现有 schema（与 schema/init.sql 一致）

Revision ID: 0001
Revises:
Create Date: 2026-10-23

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1) 启用 pgvector（第 9 周向量检索用，提前就位）
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # 2) 健康画像主表（单用户场景：种子一条默认用户）
    op.create_table(
        "health_profiles",
        sa.Column("user_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("birth_year", sa.Integer(), nullable=True),
        sa.Column("sex", sa.String(length=10), nullable=True),
        sa.Column("height_cm", sa.Numeric(), nullable=True),
        sa.Column("weight_kg", sa.Numeric(), nullable=True),
        sa.Column(
            "bmi",
            sa.Numeric(),
            sa.Computed("weight_kg / ((height_cm / 100) ^ 2)", persisted=True),
            nullable=True,
        ),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=True),
    )
    op.execute("COMMENT ON TABLE health_profiles IS '健康画像主表（单用户场景，默认种子一条用户）'")
    op.execute("COMMENT ON COLUMN health_profiles.user_id IS '用户唯一标识（单用户，固定 UUID）'")
    op.execute("COMMENT ON COLUMN health_profiles.birth_year IS '出生年份'")
    op.execute("COMMENT ON COLUMN health_profiles.sex IS '性别'")
    op.execute("COMMENT ON COLUMN health_profiles.height_cm IS '身高（厘米）'")
    op.execute("COMMENT ON COLUMN health_profiles.weight_kg IS '体重（公斤）'")
    op.execute(
        "COMMENT ON COLUMN health_profiles.bmi IS "
        "'BMI，生成列，= 体重 / (身高/100)^2，随身高体重自动计算'"
    )
    op.execute("COMMENT ON COLUMN health_profiles.updated_at IS '最后更新时间'")

    # 3) 指标记录表（睡眠/运动/饮食/检验），同日期同指标覆盖语义
    op.create_table(
        "metric_records",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("metric_code", sa.String(length=50), nullable=True),
        sa.Column("value", sa.Numeric(), nullable=True),
        sa.Column("unit", sa.String(length=20), nullable=True),
        sa.Column("recorded_date", sa.Date(), nullable=True),
        sa.Column("source", sa.String(length=30), nullable=True),
        sa.UniqueConstraint("user_id", "metric_code", "recorded_date"),
    )
    op.execute("COMMENT ON TABLE metric_records IS '指标记录表（睡眠/运动/饮食/检验，同日期同指标覆盖语义）'")
    op.execute("COMMENT ON COLUMN metric_records.id IS '自增主键'")
    op.execute("COMMENT ON COLUMN metric_records.user_id IS '用户唯一标识'")
    op.execute("COMMENT ON COLUMN metric_records.metric_code IS '指标代码（如 sleep_duration）'")
    op.execute("COMMENT ON COLUMN metric_records.value IS '指标数值'")
    op.execute("COMMENT ON COLUMN metric_records.unit IS '单位（如 h/kg）'")
    op.execute("COMMENT ON COLUMN metric_records.recorded_date IS '记录日期'")
    op.execute("COMMENT ON COLUMN metric_records.source IS '数据来源（如 manual）'")

    # 4) 种子：默认单用户（固定 UUID，代码侧使用同一常量）
    op.execute(
        "INSERT INTO health_profiles (user_id, birth_year, sex, height_cm, weight_kg, updated_at) "
        "VALUES ('00000000-0000-0000-0000-000000000001', NULL, NULL, NULL, NULL, now()) "
        "ON CONFLICT (user_id) DO NOTHING"
    )

    # 5) 知识库：症状词典
    op.create_table(
        "symptoms",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("synonyms", postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column("category", sa.String(length=50), nullable=True),
        sa.UniqueConstraint("name"),
    )
    op.execute("COMMENT ON TABLE symptoms IS '知识库症状词典（追问分诊用，name 唯一，synonyms 支持口语化归一化）'")
    op.execute("COMMENT ON COLUMN symptoms.id IS '自增主键'")
    op.execute("COMMENT ON COLUMN symptoms.name IS '症状标准名（归一化后的唯一名）'")
    op.execute("COMMENT ON COLUMN symptoms.synonyms IS '同义词数组（口语化/错别字归一化匹配用）'")
    op.execute("COMMENT ON COLUMN symptoms.category IS '症状分类（疲劳乏力/胸痛心慌/头晕头痛/胃肠消化/全身症状）'")

    # 6) 知识库：疾病/病种
    op.create_table(
        "diseases",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("department", sa.String(length=100), nullable=True),
        sa.Column("urgency_hint", sa.String(length=30), nullable=True),
        sa.UniqueConstraint("name"),
    )
    op.execute("COMMENT ON TABLE diseases IS '知识库疾病表（候选疾病集，原文档 conditions，改名 diseases 避免与病情表冲突）'")
    op.execute("COMMENT ON COLUMN diseases.id IS '自增主键'")
    op.execute("COMMENT ON COLUMN diseases.name IS '疾病名（唯一）'")
    op.execute("COMMENT ON COLUMN diseases.department IS '所属科室'")
    op.execute(
        "COMMENT ON COLUMN diseases.urgency_hint IS "
        "'分级建议等级：observe观察/adjust调整/online_consult线上问诊/seek_care建议就医'"
    )

    # 7) 知识库：疾病-症状关联
    op.create_table(
        "disease_symptoms",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("disease_id", sa.Integer(), sa.ForeignKey("diseases.id"), nullable=False),
        sa.Column("symptom_id", sa.Integer(), sa.ForeignKey("symptoms.id"), nullable=False),
        sa.Column("prevalence", sa.Numeric(), nullable=True),
        sa.Column("discrimination_weight", sa.Integer(), nullable=True),
        sa.UniqueConstraint("disease_id", "symptom_id"),
    )
    op.execute("COMMENT ON TABLE disease_symptoms IS '疾病-症状关联表（候选集反查 + 鉴别价值追问，原文档 condition_symptoms）'")
    op.execute("COMMENT ON COLUMN disease_symptoms.id IS '自增主键'")
    op.execute("COMMENT ON COLUMN disease_symptoms.disease_id IS '疾病 id，引用 diseases.id'")
    op.execute("COMMENT ON COLUMN disease_symptoms.symptom_id IS '症状 id，引用 symptoms.id'")
    op.execute("COMMENT ON COLUMN disease_symptoms.prevalence IS '该症状在该疾病中的出现率（演示近似值）'")
    op.execute(
        "COMMENT ON COLUMN disease_symptoms.discrimination_weight IS "
        "'人工标注鉴别价值 1-5，第 4 周启用排序（当前为逆文档频率近似值）'"
    )


def downgrade() -> None:
    op.drop_table("disease_symptoms")
    op.drop_table("diseases")
    op.drop_table("symptoms")
    op.drop_table("metric_records")
    op.drop_table("health_profiles")
