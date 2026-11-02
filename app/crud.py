"""指标记录与用药清单的数据访问（session 内部管理，调用方不持有会话）。"""
from datetime import date
from decimal import Decimal
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.db import async_session
from app.models import (
    Allergy,
    Condition,
    DrugInteraction,
    Followup,
    LabReferenceRange,
    Medication,
    MetricRecord,
    Preference,
)


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


async def add_medication(
    user_id: UUID,
    name: str,
    dosage: str | None = None,
    frequency: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    active: bool = True,
) -> int:
    """新增一条用药记录（第 7 周：供「在吃二甲双胍」这类对话式记录使用），返回记录 id。"""
    async with async_session() as session:
        row = Medication(
            user_id=user_id,
            name=name,
            dosage=dosage,
            frequency=frequency,
            start_date=start_date,
            end_date=end_date,
            active=active,
        )
        session.add(row)
        await session.commit()
        return row.id


async def list_conditions(user_id: UUID) -> list[dict]:
    """列出用户病情，返回 JSON 可序列化的 dict 列表（按 id 升序）。"""
    stmt = select(Condition).where(Condition.user_id == user_id).order_by(Condition.id)
    async with async_session() as session:
        rows = (await session.execute(stmt)).scalars().all()
    return [
        {
            "id": row.id,
            "user_id": str(row.user_id),
            "condition_name": row.condition_name,
            "diagnosed_date": row.diagnosed_date.isoformat() if row.diagnosed_date else None,
            "status": row.status,
            "notes": row.notes,
        }
        for row in rows
    ]


async def add_condition(
    user_id: UUID,
    condition_name: str,
    diagnosed_date: date | None = None,
    status: str | None = None,
    notes: str | None = None,
) -> int:
    """新增一条病情记录，返回记录 id。"""
    async with async_session() as session:
        row = Condition(
            user_id=user_id,
            condition_name=condition_name,
            diagnosed_date=diagnosed_date,
            status=status,
            notes=notes,
        )
        session.add(row)
        await session.commit()
        return row.id


async def list_allergies(user_id: UUID) -> list[dict]:
    """列出用户过敏原，返回 JSON 可序列化的 dict 列表（按 id 升序）。"""
    stmt = select(Allergy).where(Allergy.user_id == user_id).order_by(Allergy.id)
    async with async_session() as session:
        rows = (await session.execute(stmt)).scalars().all()
    return [
        {
            "id": row.id,
            "user_id": str(row.user_id),
            "allergen": row.allergen,
            "severity": row.severity,
            "reaction": row.reaction,
            "notes": row.notes,
        }
        for row in rows
    ]


async def add_allergy(
    user_id: UUID,
    allergen: str,
    severity: str | None = None,
    reaction: str | None = None,
    notes: str | None = None,
) -> int:
    """新增一条过敏记录，返回记录 id。"""
    async with async_session() as session:
        row = Allergy(
            user_id=user_id,
            allergen=allergen,
            severity=severity,
            reaction=reaction,
            notes=notes,
        )
        session.add(row)
        await session.commit()
        return row.id


async def list_preferences(user_id: UUID) -> list[dict]:
    """列出用户偏好，返回 JSON 可序列化的 dict 列表（按 id 升序）。"""
    stmt = select(Preference).where(Preference.user_id == user_id).order_by(Preference.id)
    async with async_session() as session:
        rows = (await session.execute(stmt)).scalars().all()
    return [
        {
            "id": row.id,
            "user_id": str(row.user_id),
            "preference_key": row.preference_key,
            "preference_value": row.preference_value,
            "notes": row.notes,
        }
        for row in rows
    ]


async def add_preference(
    user_id: UUID,
    preference_key: str,
    preference_value: str | None = None,
    notes: str | None = None,
) -> int:
    """新增一条偏好记录，返回记录 id。"""
    async with async_session() as session:
        row = Preference(
            user_id=user_id,
            preference_key=preference_key,
            preference_value=preference_value,
            notes=notes,
        )
        session.add(row)
        await session.commit()
        return row.id


async def list_followups(user_id: UUID) -> list[dict]:
    """列出用户待跟进事项，返回 JSON 可序列化的 dict 列表（按 id 升序）。"""
    stmt = select(Followup).where(Followup.user_id == user_id).order_by(Followup.id)
    async with async_session() as session:
        rows = (await session.execute(stmt)).scalars().all()
    return [
        {
            "id": row.id,
            "user_id": str(row.user_id),
            "item": row.item,
            "due_date": row.due_date.isoformat() if row.due_date else None,
            "status": row.status,
            "notes": row.notes,
        }
        for row in rows
    ]


async def add_followup(
    user_id: UUID,
    item: str,
    due_date: date | None = None,
    status: str | None = None,
    notes: str | None = None,
) -> int:
    """新增一条待跟进事项，返回记录 id。"""
    async with async_session() as session:
        row = Followup(
            user_id=user_id,
            item=item,
            due_date=due_date,
            status=status,
            notes=notes,
        )
        session.add(row)
        await session.commit()
        return row.id


async def get_reference_units(metric_codes: list[str]) -> dict[str, set[str]]:
    """返回 metric_code -> 该指标在参考范围表中出现过的单位集合（导入单位校验用）。

    某指标没有参考范围记录时不进结果集（调用方据此跳过单位一致性校验）。
    """
    if not metric_codes:
        return {}
    stmt = select(LabReferenceRange.metric_code, LabReferenceRange.unit).where(
        LabReferenceRange.metric_code.in_(metric_codes)
    )
    async with async_session() as session:
        rows = (await session.execute(stmt)).all()

    result: dict[str, set[str]] = {}
    for metric_code, unit in rows:
        result.setdefault(metric_code, set()).add(unit)
    return result


async def bulk_upsert_metrics(
    user_id: UUID,
    rows: list[dict],
    source: str = "csv",
) -> None:
    """批量覆盖写入指标（同 user_id+metric_code+recorded_date 覆盖），调用方保证 rows 已校验。

    同批次内同键重复时最后一条生效（先在 Python 内按顺序去重），与单条覆盖语义一致。
    """
    if not rows:
        return

    # 同批次去重：以 (metric_code, recorded_date) 为键，后出现的覆盖先出现的
    deduped: dict[tuple[str, date], dict] = {}
    for row in rows:
        key = (row["metric_code"], row["recorded_date"])
        deduped[key] = row

    values = [
        {
            "user_id": user_id,
            "metric_code": row["metric_code"],
            "value": Decimal(str(row["value"])),
            "unit": row["unit"],
            "recorded_date": row["recorded_date"],
            "source": source,
        }
        for row in deduped.values()
    ]

    stmt = insert(MetricRecord).values(values)
    stmt = stmt.on_conflict_do_update(
        index_elements=["user_id", "metric_code", "recorded_date"],
        set_={
            "value": stmt.excluded.value,
            "unit": stmt.excluded.unit,
            "source": stmt.excluded.source,
        },
    )
    async with async_session() as session:
        await session.execute(stmt)
        await session.commit()


async def find_drug_interactions(drug_names: list[str]) -> list[dict]:
    """返回 drug_interactions 中两个端点都在给定药物集合内的规则（按 id 升序）。

    端点集合判定天然方向无关：drug_a / drug_b 只要都在 drug_names 里即命中，
    调用方（规则引擎）负责把命中按风险等级排序。
    """
    if len(drug_names) < 2:
        return []
    stmt = (
        select(DrugInteraction)
        .where(
            DrugInteraction.drug_a.in_(drug_names),
            DrugInteraction.drug_b.in_(drug_names),
        )
        .order_by(DrugInteraction.id)
    )
    async with async_session() as session:
        rows = (await session.execute(stmt)).scalars().all()

    return [
        {
            "drug_a": row.drug_a,
            "drug_b": row.drug_b,
            "severity": row.severity,
            "description": row.description,
        }
        for row in rows
    ]
