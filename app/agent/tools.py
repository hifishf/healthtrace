"""Agent 工具：把落库/查画像能力包装成 LLM 可调用的函数（经 MCP 客户端访问数据）。"""
from langchain_core.tools import tool

from app.identity import get_current_user_id
from app.mcp.client import call_tool


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
    - date：用户原话中的日期或日期表达，例如“今天”“昨晚”“三天前”；
      也可以直接传 YYYY-MM-DD。不要由模型自行换算相对日期，不传则默认今天。
    示例：用户说“昨晚睡了 6 小时”→
    record_metric(metric_code="sleep_duration", value=6, unit="h", date="昨晚")。
    返回 {ok: true, id, ...} 或 {ok: false, error_code}。
    """
    return await call_tool(
        "record_metric",
        {
            "user_id": str(get_current_user_id()),
            "metric_code": metric_code,
            "value": value,
            "unit": unit,
            "date_text": date,
        },
    )


@tool
async def query_profile() -> dict:
    """查询当前用户的健康画像（出生年份、性别、身高、体重、BMI 等）。

    无参数（单用户）。返回画像字段 dict；无画像时返回空 dict。
    """
    return await call_tool(
        "get_health_portrait", {"user_id": str(get_current_user_id())}
    )
