"""ORM 模型（与 alembic/versions 迁移保持一致，字段注释即数据库 COMMENT）。"""
from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import (
    ARRAY,
    TIMESTAMP,
    Boolean,
    Computed,
    Date,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """ORM 基类。"""


class MetricRecord(Base):
    """指标记录（睡眠/运动/饮食/检验）。"""

    __tablename__ = "metric_records"
    __table_args__ = (
        UniqueConstraint("user_id", "metric_code", "recorded_date"),
        {"comment": "指标记录表（睡眠/运动/饮食/检验，同日期同指标覆盖语义）"},
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, comment="自增主键")
    user_id: Mapped[UUID] = mapped_column(Uuid, nullable=False, comment="用户唯一标识")
    metric_code: Mapped[str] = mapped_column(
        String(50), nullable=False, comment="指标代码（如 sleep_duration）"
    )
    value: Mapped[Decimal] = mapped_column(Numeric, nullable=False, comment="指标数值")
    unit: Mapped[str] = mapped_column(String(20), nullable=False, comment="单位（如 h/kg）")
    recorded_date: Mapped[date] = mapped_column(Date, nullable=False, comment="记录日期")
    source: Mapped[str | None] = mapped_column(
        String(30), nullable=True, comment="数据来源（如 manual）"
    )


class HealthProfile(Base):
    """健康画像主表（单用户，种子一条默认用户）。"""

    __tablename__ = "health_profiles"
    __table_args__ = {"comment": "健康画像主表（单用户场景，默认种子一条用户）"}

    user_id: Mapped[UUID] = mapped_column(
        Uuid, primary_key=True, comment="用户唯一标识（单用户，固定 UUID）"
    )
    birth_year: Mapped[int | None] = mapped_column(Integer, nullable=True, comment="出生年份")
    sex: Mapped[str | None] = mapped_column(String(10), nullable=True, comment="性别")
    height_cm: Mapped[Decimal | None] = mapped_column(
        Numeric, nullable=True, comment="身高（厘米）"
    )
    weight_kg: Mapped[Decimal | None] = mapped_column(
        Numeric, nullable=True, comment="体重（公斤）"
    )
    bmi: Mapped[Decimal | None] = mapped_column(
        Numeric,
        Computed("weight_kg / ((height_cm / 100) ^ 2)", persisted=True),
        nullable=True,
        comment="BMI，生成列，= 体重 / (身高/100)^2，随身高体重自动计算",
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True, comment="最后更新时间"
    )


class Symptom(Base):
    """知识库症状（追问分诊的症状词典，name 唯一）。"""

    __tablename__ = "symptoms"
    __table_args__ = {
        "comment": "知识库症状词典（追问分诊用，name 唯一，synonyms 支持口语化归一化）"
    }

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, comment="自增主键")
    name: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False, comment="症状标准名（归一化后的唯一名）"
    )
    synonyms: Mapped[list[str] | None] = mapped_column(
        ARRAY(Text), nullable=True, comment="同义词数组（口语化/错别字归一化匹配用）"
    )
    category: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        comment="症状分类（疲劳乏力/胸痛心慌/头晕头痛/胃肠消化/全身症状）",
    )


class Disease(Base):
    """知识库疾病（候选疾病集，name 唯一；urgency_hint 决定分级建议）。"""

    __tablename__ = "diseases"
    __table_args__ = {
        "comment": "知识库疾病表（候选疾病集，原文档 conditions，改名 diseases 避免与病情表冲突）"
    }

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, comment="自增主键")
    name: Mapped[str] = mapped_column(
        String(200), unique=True, nullable=False, comment="疾病名（唯一）"
    )
    department: Mapped[str | None] = mapped_column(
        String(100), nullable=True, comment="所属科室"
    )
    urgency_hint: Mapped[str | None] = mapped_column(
        String(30),
        nullable=True,
        comment="分级建议等级：observe观察/adjust调整/online_consult线上问诊/seek_care建议就医",
    )


class DiseaseSymptom(Base):
    """疾病-症状关联（候选集反查 + 鉴别价值追问）。"""

    __tablename__ = "disease_symptoms"
    __table_args__ = (
        UniqueConstraint("disease_id", "symptom_id"),
        {
            "comment": "疾病-症状关联表（候选集反查 + 鉴别价值追问，原文档 condition_symptoms）"
        },
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, comment="自增主键")
    disease_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("diseases.id"), nullable=False, comment="疾病 id，引用 diseases.id"
    )
    symptom_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("symptoms.id"), nullable=False, comment="症状 id，引用 symptoms.id"
    )
    prevalence: Mapped[Decimal | None] = mapped_column(
        Numeric, nullable=True, comment="该症状在该疾病中的出现率（演示近似值）"
    )
    discrimination_weight: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="人工标注鉴别价值 1-5，第 4 周启用排序（当前为逆文档频率近似值）",
    )


class Medication(Base):
    """用药清单（第 6 周最小化，字段与文档第八节一致；第 9 周扩展交互检查）。"""

    __tablename__ = "medications"
    __table_args__ = {"comment": "用药清单（第 6 周最小化，第 9 周扩展交互检查）"}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, comment="自增主键")
    user_id: Mapped[UUID] = mapped_column(
        Uuid, nullable=False, index=True, comment="用户唯一标识"
    )
    name: Mapped[str] = mapped_column(
        String(200), nullable=False, comment="药物/保健品名称"
    )
    dosage: Mapped[str | None] = mapped_column(
        String(100), nullable=True, comment="剂量（如 500mg/次）"
    )
    frequency: Mapped[str | None] = mapped_column(
        String(100), nullable=True, comment="频次（如 每日两次）"
    )
    start_date: Mapped[date | None] = mapped_column(
        Date, nullable=True, comment="开始服用日期"
    )
    end_date: Mapped[date | None] = mapped_column(
        Date, nullable=True, comment="停止服用日期（空=仍在服用）"
    )
    active: Mapped[bool | None] = mapped_column(
        Boolean, server_default=text("true"), nullable=True, comment="是否在服，默认 true"
    )


class Condition(Base):
    """病情（用户已知健康问题，画像语义记忆；与知识库 diseases 表不同）。"""

    __tablename__ = "conditions"
    __table_args__ = {"comment": "病情（用户已知健康问题，画像语义记忆；与知识库 diseases 表不同）"}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, comment="自增主键")
    user_id: Mapped[UUID] = mapped_column(
        Uuid, nullable=False, index=True, comment="用户唯一标识"
    )
    condition_name: Mapped[str] = mapped_column(
        String(200), nullable=False, comment="病情名称（如 高血压、2型糖尿病）"
    )
    diagnosed_date: Mapped[date | None] = mapped_column(
        Date, nullable=True, comment="确诊日期"
    )
    status: Mapped[str | None] = mapped_column(
        String(20), nullable=True, comment="状态（active在患/remission缓解/resolved已愈）"
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True, comment="备注")


class Allergy(Base):
    """过敏（用户过敏原画像，语义记忆）。"""

    __tablename__ = "allergies"
    __table_args__ = {"comment": "过敏（用户过敏原画像，语义记忆）"}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, comment="自增主键")
    user_id: Mapped[UUID] = mapped_column(
        Uuid, nullable=False, index=True, comment="用户唯一标识"
    )
    allergen: Mapped[str] = mapped_column(
        String(200), nullable=False, comment="过敏原（如 青霉素、花生）"
    )
    severity: Mapped[str | None] = mapped_column(
        String(20), nullable=True, comment="严重程度（mild/moderate/severe）"
    )
    reaction: Mapped[str | None] = mapped_column(
        String(200), nullable=True, comment="过敏反应表现（如 皮疹、呼吸困难）"
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True, comment="备注")


class Preference(Base):
    """偏好（用户长期偏好，画像语义记忆）。"""

    __tablename__ = "preferences"
    __table_args__ = {"comment": "偏好（用户长期偏好，画像语义记忆）"}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, comment="自增主键")
    user_id: Mapped[UUID] = mapped_column(
        Uuid, nullable=False, index=True, comment="用户唯一标识"
    )
    preference_key: Mapped[str] = mapped_column(
        String(100), nullable=False, comment="偏好键（如 沟通风格、饮食偏好）"
    )
    preference_value: Mapped[str | None] = mapped_column(
        String(200), nullable=True, comment="偏好值"
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True, comment="备注")


class Followup(Base):
    """待跟进事项（用户后续动作，画像语义记忆）。"""

    __tablename__ = "followups"
    __table_args__ = {"comment": "待跟进事项（用户后续动作，画像语义记忆）"}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, comment="自增主键")
    user_id: Mapped[UUID] = mapped_column(
        Uuid, nullable=False, index=True, comment="用户唯一标识"
    )
    item: Mapped[str] = mapped_column(
        String(200), nullable=False, comment="待跟进事项内容"
    )
    due_date: Mapped[date | None] = mapped_column(
        Date, nullable=True, comment="计划跟进日期"
    )
    status: Mapped[str | None] = mapped_column(
        String(20), nullable=True, comment="状态（pending待处理/done已完成）"
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True, comment="备注")
