"""FastMCP 数据服务：把健康数据读写封装成 MCP 工具，Agent 层不再直接写 SQL。"""
from datetime import date
from decimal import Decimal
from uuid import UUID

from fastmcp import FastMCP

from app import crud, portrait
from app.dates import resolve_metric_date

mcp = FastMCP("HealthTrace Data")


@mcp.tool
async def get_health_portrait(user_id: str) -> dict:
    """读取用户健康画像（出生年份/性别/身高/体重/BMI 等），无画像返回空 dict。"""
    return await portrait.get_health_portrait(UUID(user_id))


@mcp.tool
async def query_metrics(
    user_id: str,
    metric_code: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """按用户查询指标记录，可按指标代码与日期范围（闭区间）过滤，返回 {"metrics": [...]}。"""
    metrics = await crud.query_metrics(
        UUID(user_id),
        metric_code=metric_code,
        start_date=date.fromisoformat(start_date) if start_date else None,
        end_date=date.fromisoformat(end_date) if end_date else None,
    )
    return {"metrics": metrics}


@mcp.tool
async def import_metrics(user_id: str, source: str, metrics: list[dict]) -> dict:
    """预留：真实 HealthKit 数据导入未实现（第 9 周接入），当前返回未实现标记。"""
    return {
        "ok": False,
        "error_code": "NOT_IMPLEMENTED",
        "message": "HealthKit 数据导入第 9 周实现",
    }


@mcp.tool
async def list_medications(user_id: str) -> dict:
    """列出用户用药清单（第 6 周为空表，第 9 周扩展交互检查），返回 {"medications": [...]}。"""
    medications = await crud.list_medications(UUID(user_id))
    return {"medications": medications}


@mcp.tool
async def check_drug_interaction(user_id: str, medication_names: list[str]) -> dict:
    """预留：药物交互规则引擎未实现（第 9 周接入），当前返回未实现标记。"""
    return {
        "ok": False,
        "error_code": "NOT_IMPLEMENTED",
        "message": "药物交互检查第 9 周实现",
    }


@mcp.tool
async def record_metric(
    user_id: str,
    metric_code: str,
    value: float,
    unit: str,
    date_text: str | None = None,
    source: str = "chat",
) -> dict:
    """记录一条健康指标（覆盖语义：同一天同一指标只保留最后一次）。

    date_text 为用户原话中的日期表达（如“昨晚”“三天前”）或 YYYY-MM-DD，
    不传则默认今天；相对日期由业务代码确定性换算。
    """
    if not metric_code or not metric_code.strip():
        return {"ok": False, "error_code": "INVALID_METRIC_CODE"}

    try:
        record_date = resolve_metric_date(date_text)
    except ValueError:
        return {"ok": False, "error_code": "INVALID_DATE"}

    uid = UUID(user_id)
    record_id = await crud.upsert_metric(
        user_id=uid,
        metric_code=metric_code.strip(),
        value=Decimal(str(value)),
        unit=unit,
        recorded_date=record_date,
        source=source,
    )
    # 记录指标 = 画像数据变过，刷新画像 updated_at
    await portrait.touch_portrait(uid)

    return {
        "ok": True,
        "id": record_id,
        "metric_code": metric_code.strip(),
        "value": value,
        "unit": unit,
        "date": record_date.isoformat(),
    }
