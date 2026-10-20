"""Agent 工具：把落库等能力包装成 LLM 可调用的函数。"""
import datetime
from decimal import Decimal

from langchain_core.tools import tool

from app import crud, portrait
from app.config import settings
from app.db import async_session


@tool
async def record_metric(
    metric_code: str,
    value: float,
    unit: str,
    date: str | None = None,
) -> dict:
    """记录一条健康指标（覆盖语义：同一天同一指标只保留最后一次）。

    参数：
    - metric_code：指标代码，例如 sleep_duration（睡眠时长）。
    - value：指标数值，例如 6。
    - unit：单位，例如 h。
    - date：记录日期（YYYY-MM-DD），不传则默认今天。
    示例：用户说“昨晚睡了 6 小时”→ record_metric(metric_code="sleep_duration", value=6, unit="h")。
    返回 {ok: true, id, ...} 或 {ok: false, error_code}。
    """
    # 基础校验：metric_code 非空
    if not metric_code or not metric_code.strip():
        return {"ok": False, "error_code": "INVALID_METRIC_CODE"}

    # 日期解析：缺省今天，解析失败返回错误码（不让 LLM 猜失败原因）
    try:
        record_date = datetime.date.fromisoformat(date) if date else datetime.date.today()
    except ValueError:
        return {"ok": False, "error_code": "INVALID_DATE"}

    async with async_session() as session:
        record_id = await crud.upsert_metric(
            session=session,
            user_id=settings.default_user_id,
            metric_code=metric_code.strip(),
            value=Decimal(str(value)),
            unit=unit,
            recorded_date=record_date,
            source="chat",
        )
        # 记录指标 = 画像数据变过，刷新画像 updated_at（“画像更新”）
        await portrait.touch_portrait(session, settings.default_user_id)

    return {
        "ok": True,
        "id": record_id,
        "metric_code": metric_code.strip(),
        "value": value,
        "unit": unit,
        "date": record_date.isoformat(),
    }


@tool
async def query_profile() -> dict:
    """查询当前用户的健康画像（出生年份、性别、身高、体重、BMI 等）。

    无参数（单用户）。返回画像字段 dict；无画像时返回空 dict。
    """
    async with async_session() as session:
        return await portrait.get_health_portrait(session, settings.default_user_id)
