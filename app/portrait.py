"""健康画像读写：组装完整画像（基础 + 病情/用药/过敏/偏好/待跟进）、刷新 updated_at。"""
import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from app import crud
from app.db import async_session
from app.models import HealthProfile


def _to_float(value: Decimal | None) -> float | None:
    """Decimal -> float（JSON 序列化用），None 保持 None。"""
    return float(value) if value is not None else None


async def get_health_portrait(user_id: UUID) -> dict[str, Any]:
    """读取完整健康画像（第 7 周语义记忆完整化）。

    返回 JSON 可序列化的 dict：基础字段 + conditions/medications/allergies/
    preferences/followups 五个子表列表；无画像行时返回空 dict。
    """
    async with async_session() as session:
        profile = await session.get(HealthProfile, user_id)
        if profile is None:
            return {}
        base = {
            "user_id": str(profile.user_id),
            "birth_year": profile.birth_year,
            "sex": profile.sex,
            "height_cm": _to_float(profile.height_cm),
            "weight_kg": _to_float(profile.weight_kg),
            "bmi": _to_float(profile.bmi),
            "updated_at": profile.updated_at.isoformat() if profile.updated_at else None,
        }
    return {
        **base,
        "conditions": await crud.list_conditions(user_id),
        "medications": await crud.list_medications(user_id),
        "allergies": await crud.list_allergies(user_id),
        "preferences": await crud.list_preferences(user_id),
        "followups": await crud.list_followups(user_id),
    }


def summarize(portrait: dict[str, Any]) -> str:
    """把画像 dict 压成一段中文摘要，供注入 LLM 上下文；无有效字段返回「暂无画像信息」。

    基础字段用「，」连接，病情/用药/过敏/偏好/待跟进各成一小节、用「；」连接，
    让新会话注入时能一眼看出已确认的病史与用药，避免重复追问。
    """
    parts: list[str] = []

    sex = portrait.get("sex")
    if sex:
        parts.append(str(sex))

    birth_year = portrait.get("birth_year")
    if birth_year is not None:
        parts.append(f"{birth_year}年生")

    height_cm = portrait.get("height_cm")
    if height_cm is not None:
        parts.append(f"身高{height_cm}cm")

    weight_kg = portrait.get("weight_kg")
    if weight_kg is not None:
        parts.append(f"体重{weight_kg}kg")

    bmi = portrait.get("bmi")
    if bmi is not None:
        parts.append(f"BMI {float(bmi):.1f}")

    sections: list[str] = []
    if parts:
        sections.append("，".join(parts))

    conditions = portrait.get("conditions") or []
    if conditions:
        names = [c.get("condition_name") for c in conditions if c.get("condition_name")]
        if names:
            sections.append("病史：" + "、".join(names))

    medications = portrait.get("medications") or []
    if medications:
        descriptions = [
            " ".join(filter(None, [m.get("name"), m.get("dosage"), m.get("frequency")]))
            for m in medications
            if m.get("name")
        ]
        if descriptions:
            sections.append("在服：" + "、".join(descriptions))

    allergies = portrait.get("allergies") or []
    if allergies:
        descriptions = [
            " ".join(filter(None, [a.get("allergen"), a.get("severity")]))
            for a in allergies
            if a.get("allergen")
        ]
        if descriptions:
            sections.append("过敏：" + "、".join(descriptions))

    preferences = portrait.get("preferences") or []
    if preferences:
        descriptions = [
            f"{p.get('preference_key')}:{p.get('preference_value')}"
            if p.get("preference_value")
            else p.get("preference_key")
            for p in preferences
            if p.get("preference_key")
        ]
        if descriptions:
            sections.append("偏好：" + "、".join(descriptions))

    followups = portrait.get("followups") or []
    if followups:
        descriptions = [
            f"{f.get('item')}（{f.get('due_date')}）" if f.get("due_date") else f.get("item")
            for f in followups
            if f.get("item")
        ]
        if descriptions:
            sections.append("待跟进：" + "、".join(descriptions))

    if not sections:
        return "暂无画像信息"
    return "；".join(sections)


async def touch_portrait(user_id: UUID) -> None:
    """刷新画像 updated_at（表示画像最近一次被更新）。

    无画像行时静默跳过（单用户种子已存在，正常不会发生）。
    """
    async with async_session() as session:
        profile = await session.get(HealthProfile, user_id)
        if profile is None:
            return
        profile.updated_at = datetime.datetime.now(datetime.timezone.utc)
        await session.commit()
