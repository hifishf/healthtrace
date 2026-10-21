"""ORM 模型（与 schema/init.sql 保持一致）。"""
from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import (
    ARRAY,
    TIMESTAMP,
    Computed,
    Date,
    ForeignKey,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """ORM 基类。"""


class MetricRecord(Base):
    """指标记录（睡眠/运动/饮食/检验）。

    幂等约束 (user_id, metric_code, recorded_date) 由 init.sql 里的 UNIQUE 兜底，
    同日期同指标为覆盖语义。
    """

    __tablename__ = "metric_records"
    __table_args__ = (
        UniqueConstraint("user_id", "metric_code", "recorded_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[UUID] = mapped_column(Uuid)
    metric_code: Mapped[str] = mapped_column(String(50))
    value: Mapped[Decimal] = mapped_column(Numeric)
    unit: Mapped[str] = mapped_column(String(20))
    recorded_date: Mapped[date] = mapped_column(Date)
    source: Mapped[str | None] = mapped_column(String(30), nullable=True)


class HealthProfile(Base):
    """健康画像主表（单用户，种子一条默认用户）。

    bmi 为生成列（GENERATED ALWAYS AS ... STORED），用 Computed 标记为只读，
    不参与 INSERT/UPDATE。
    """

    __tablename__ = "health_profiles"

    user_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    birth_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sex: Mapped[str | None] = mapped_column(String(10), nullable=True)
    height_cm: Mapped[Decimal | None] = mapped_column(Numeric, nullable=True)
    weight_kg: Mapped[Decimal | None] = mapped_column(Numeric, nullable=True)
    bmi: Mapped[Decimal | None] = mapped_column(
        Numeric,
        Computed("weight_kg / ((height_cm / 100) ^ 2)", persisted=True),
        nullable=True,
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )


class Symptom(Base):
    """知识库症状（追问分诊的症状词典，name 唯一）。"""

    __tablename__ = "symptoms"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)  # 自增主键
    name: Mapped[str] = mapped_column(String(100), unique=True)  # 症状标准名（归一化唯一名）
    synonyms: Mapped[list[str] | None] = mapped_column(ARRAY(String), nullable=True)  # 同义词数组（口语化/错别字归一化）
    category: Mapped[str | None] = mapped_column(String(50), nullable=True)  # 症状分类


class Disease(Base):
    """知识库疾病（候选疾病集，name 唯一；urgency_hint 决定分级建议）。"""

    __tablename__ = "diseases"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)  # 自增主键
    name: Mapped[str] = mapped_column(String(200), unique=True)  # 疾病名（唯一）
    department: Mapped[str | None] = mapped_column(String(100), nullable=True)  # 所属科室
    urgency_hint: Mapped[str | None] = mapped_column(String(30), nullable=True)  # 分级建议等级：observe/adjust/online_consult/seek_care


class DiseaseSymptom(Base):
    """疾病-症状关联（候选集反查 + 鉴别价值追问）。"""

    __tablename__ = "disease_symptoms"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)  # 自增主键
    disease_id: Mapped[int] = mapped_column(Integer, ForeignKey("diseases.id"))  # 疾病 id，引用 diseases.id
    symptom_id: Mapped[int] = mapped_column(Integer, ForeignKey("symptoms.id"))  # 症状 id，引用 symptoms.id
    prevalence: Mapped[Decimal | None] = mapped_column(Numeric, nullable=True)  # 该症状在该疾病中的出现率（演示近似值）
    discrimination_weight: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 鉴别价值 1-5，第 4 周启用排序
