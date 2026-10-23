"""指标记录与用药清单的数据访问（session 内部管理，调用方不持有会话）。"""
from datetime import date
from decimal import Decimal
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.db import async_session
from app.models import Medication, MetricRecord


async def upsert_metric(
    user_id: UUID,
    metric_code: str,
    value: Decimal,
    unit: str,
    recorded_date: date,
    source: str = "manual",
) -> int:
    """同日期同指标覆盖写入（upsert），返回记录 id。

    覆盖语义：以 (user_id, metric_code, recorded_date) 为唯一键，
    已存在时更新 value/unit/source，不新增行。
    """
    stmt = insert(MetricRecord).values(
        user_id=user_id,
        metric_code=metric_code,
        value=value,
        unit=unit,
        recorded_date=recorded_date,
        source=source,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["user_id", "metric_code", "recorded_date"],
        set_={
            "value": stmt.excluded.value,
            "unit": stmt.excluded.unit,
            "source": stmt.excluded.source,
        },
    ).returning(MetricRecord.id)

    async with async_session() as session:
        result = await session.execute(stmt)
        await session.commit()
        return result.scalar_one()


async def query_metrics(
    user_id: UUID,
    metric_code: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[dict]:
    """按用户查询指标记录，返回 JSON 可序列化的 dict 列表（按日期升序）。

    metric_code / start_date / end_date 为可选过滤条件，闭区间。
    """
    stmt = select(MetricRecord).where(MetricRecord.user_id == user_id)
    if metric_code:
        stmt = stmt.where(MetricRecord.metric_code == metric_code)
    if start_date:
        stmt = stmt.where(MetricRecord.recorded_date >= start_date)
    if end_date:
        stmt = stmt.where(MetricRecord.recorded_date <= end_date)
    stmt = stmt.order_by(MetricRecord.recorded_date, MetricRecord.id)

    async with async_session() as session:
        rows = (await session.execute(stmt)).scalars().all()

    return [
        {
            "id": row.id,
            "user_id": str(row.user_id),
            "metric_code": row.metric_code,
            "value": float(row.value),
            "unit": row.unit,
            "recorded_date": row.recorded_date.isoformat(),
            "source": row.source,
        }
        for row in rows
    ]


async def list_medications(user_id: UUID) -> list[dict]:
    """列出用户用药清单，返回 JSON 可序列化的 dict 列表（按 id 升序）。"""
    stmt = select(Medication).where(Medication.user_id == user_id).order_by(Medication.id)
    async with async_session() as session:
        rows = (await session.execute(stmt)).scalars().all()

    return [
        {
            "id": row.id,
            "user_id": str(row.user_id),
            "name": row.name,
            "dosage": row.dosage,
            "frequency": row.frequency,
            "start_date": row.start_date.isoformat() if row.start_date else None,
            "end_date": row.end_date.isoformat() if row.end_date else None,
            "active": row.active,
        }
        for row in rows
    ]
