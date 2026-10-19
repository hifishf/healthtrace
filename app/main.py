"""FastAPI 应用入口。"""
from fastapi import Depends, FastAPI
from sqlalchemy.ext.asyncio import AsyncSession

from app import crud
from app.config import settings
from app.db import async_session
from app.observability import get_tracer, init_observability
from app.schemas import MetricCreate, MetricResponse

app = FastAPI(title="HealthTrace API", version="0.1.0")

init_observability()
tracer = get_tracer()


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
    with tracer.start_as_current_span("record_metric") as root_span:
        root_span.set_attribute("langfuse.trace.name", "记录健康指标")
        root_span.set_attribute("langfuse.user.id", str(settings.default_user_id))

        record_id = await crud.upsert_metric(
            session=session,
            user_id=settings.default_user_id,
            metric_code=payload.metric_code,
            value=payload.value,
            unit=payload.unit,
            recorded_date=payload.recorded_date,
        )

        # mock 一次 LLM 调用，先占住 trace 链路；第 2 周接真实 LLM 时替换
        with tracer.start_as_current_span("mock_llm_call") as llm_span:
            llm_span.set_attribute("langfuse.observation.type", "generation")
            llm_span.set_attribute("gen_ai.system", "mock")
            llm_span.set_attribute("gen_ai.request.model", "mock-model")
            llm_span.set_attribute(
                "langfuse.observation.input",
                f"记录 {payload.metric_code}={payload.value}{payload.unit}",
            )
            llm_span.set_attribute("langfuse.observation.output", "ok")

        return MetricResponse(ok=True, id=record_id)
