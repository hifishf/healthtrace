"""ORM 模型（与 schema/init.sql 保持一致）。"""
from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import (
    TIMESTAMP,
    Computed,
    Date,
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
