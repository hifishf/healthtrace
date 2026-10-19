"""FastAPI 应用入口。"""
from fastapi import Depends, FastAPI
from sqlalchemy.ext.asyncio import AsyncSession

from app import crud
from app.config import settings
from app.db import async_session
from app.schemas import MetricCreate, MetricResponse

app = FastAPI(title="HealthTrace API", version="0.1.0")


async def get_session() -> AsyncSession:
    """请求级数据库会话依赖。"""
    async with async_session() as session:
        yield session


@app.post("/metrics", response_model=MetricResponse)
async def create_metric(
    payload: MetricCreate,
    session: AsyncSession = Depends(get_session),
) -> MetricResponse:
    """写入一条指标记录（同日期同指标覆盖）。"""
    record_id = await crud.upsert_metric(
        session=session,
        user_id=settings.default_user_id,
        metric_code=payload.metric_code,
        value=payload.value,
        unit=payload.unit,
        recorded_date=payload.recorded_date,
    )
    return MetricResponse(ok=True, id=record_id)
