"""FastMCP 数据服务：把健康数据读写封装成 MCP 工具，Agent 层不再直接写 SQL。"""
from datetime import date
from decimal import Decimal
from uuid import UUID

from fastmcp import FastMCP

from app import crud, memory, portrait
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


def _parse_date(value: str | None) -> date | None:
    """把 ISO 日期字符串解析为 date；空值返回 None，非法抛 ValueError。"""
    return date.fromisoformat(value) if value else None


@mcp.tool
async def record_medication(
    user_id: str,
    name: str,
    dosage: str | None = None,
    frequency: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    active: bool = True,
) -> dict:
    """新增一条用药记录（如「在吃二甲双胍」），返回 {"ok": true, "id": ...}。"""
    if not name or not name.strip():
        return {"ok": False, "error_code": "INVALID_NAME"}

    try:
        start = _parse_date(start_date)
        end = _parse_date(end_date)
    except ValueError:
        return {"ok": False, "error_code": "INVALID_DATE"}

    record_id = await crud.add_medication(
        user_id=UUID(user_id),
        name=name.strip(),
        dosage=dosage,
        frequency=frequency,
        start_date=start,
        end_date=end,
        active=active,
    )
    await portrait.touch_portrait(UUID(user_id))
    return {"ok": True, "id": record_id, "name": name.strip()}


@mcp.tool
async def record_condition(
    user_id: str,
    condition_name: str,
    diagnosed_date: str | None = None,
    status: str | None = None,
    notes: str | None = None,
) -> dict:
    """新增一条病情记录（如 高血压、2型糖尿病），返回 {"ok": true, "id": ...}。"""
    if not condition_name or not condition_name.strip():
        return {"ok": False, "error_code": "INVALID_CONDITION_NAME"}

    try:
        diagnosed = _parse_date(diagnosed_date)
    except ValueError:
        return {"ok": False, "error_code": "INVALID_DATE"}

    record_id = await crud.add_condition(
        user_id=UUID(user_id),
        condition_name=condition_name.strip(),
        diagnosed_date=diagnosed,
        status=status,
        notes=notes,
    )
    await portrait.touch_portrait(UUID(user_id))
    return {"ok": True, "id": record_id, "condition_name": condition_name.strip()}


@mcp.tool
async def record_allergy(
    user_id: str,
    allergen: str,
    severity: str | None = None,
    reaction: str | None = None,
    notes: str | None = None,
) -> dict:
    """新增一条过敏记录（如 青霉素过敏），返回 {"ok": true, "id": ...}。"""
    if not allergen or not allergen.strip():
        return {"ok": False, "error_code": "INVALID_ALLERGEN"}

    record_id = await crud.add_allergy(
        user_id=UUID(user_id),
        allergen=allergen.strip(),
        severity=severity,
        reaction=reaction,
        notes=notes,
    )
    await portrait.touch_portrait(UUID(user_id))
    return {"ok": True, "id": record_id, "allergen": allergen.strip()}


@mcp.tool
async def record_preference(
    user_id: str,
    preference_key: str,
    preference_value: str | None = None,
    notes: str | None = None,
) -> dict:
    """新增一条偏好记录（如 沟通风格=简洁），返回 {"ok": true, "id": ...}。"""
    if not preference_key or not preference_key.strip():
        return {"ok": False, "error_code": "INVALID_PREFERENCE_KEY"}

    record_id = await crud.add_preference(
        user_id=UUID(user_id),
        preference_key=preference_key.strip(),
        preference_value=preference_value,
        notes=notes,
    )
    await portrait.touch_portrait(UUID(user_id))
    return {"ok": True, "id": record_id, "preference_key": preference_key.strip()}


@mcp.tool
async def record_followup(
    user_id: str,
    item: str,
    due_date: str | None = None,
    status: str | None = None,
    notes: str | None = None,
) -> dict:
    """新增一条待跟进事项（如 三个月后复查血糖），返回 {"ok": true, "id": ...}。"""
    if not item or not item.strip():
        return {"ok": False, "error_code": "INVALID_ITEM"}

    try:
        due = _parse_date(due_date)
    except ValueError:
        return {"ok": False, "error_code": "INVALID_DATE"}

    record_id = await crud.add_followup(
        user_id=UUID(user_id),
        item=item.strip(),
        due_date=due,
        status=status,
        notes=notes,
    )
    await portrait.touch_portrait(UUID(user_id))
    return {"ok": True, "id": record_id, "item": item.strip()}


@mcp.tool
async def list_episodic_memories(
    user_id: str,
    subgraph: str | None = None,
    limit: int = 20,
) -> dict:
    """按用户检索情景记忆（可按 subgraph 过滤，按时间倒序），返回 {"memories": [...]}。"""
    memories = await memory.list_episodic(UUID(user_id), subgraph=subgraph, limit=limit)
    return {"memories": memories}


@mcp.tool
async def delete_episodic_memory(user_id: str, memory_id: int) -> dict:
    """按用户 + 记录 id 删除一条情景记忆，返回 {"ok": true, "deleted": bool}。"""
    deleted = await memory.delete_episodic(UUID(user_id), memory_id)
    return {"ok": True, "deleted": deleted}
