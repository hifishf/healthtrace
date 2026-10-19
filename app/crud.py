"""指标记录的 upsert 逻辑。"""
from datetime import date
from decimal import Decimal
from uuid import UUID

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import MetricRecord


async def upsert_metric(
    session: AsyncSession,
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

    result = await session.execute(stmt)
    await session.commit()
    return result.scalar_one()
