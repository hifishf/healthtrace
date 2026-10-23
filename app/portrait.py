"""健康画像读写：读画像字段、刷新 updated_at（session 内部管理）。"""
import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from app.db import async_session
from app.models import HealthProfile


def _to_float(value: Decimal | None) -> float | None:
    """Decimal -> float（JSON 序列化用），None 保持 None。"""
    return float(value) if value is not None else None


async def get_health_portrait(user_id: UUID) -> dict[str, Any]:
    """读取健康画像，返回 JSON 可序列化的 dict；无画像时返回空 dict。"""
    async with async_session() as session:
        profile = await session.get(HealthProfile, user_id)
        if profile is None:
            return {}
        return {
            "user_id": str(profile.user_id),
            "birth_year": profile.birth_year,
            "sex": profile.sex,
            "height_cm": _to_float(profile.height_cm),
            "weight_kg": _to_float(profile.weight_kg),
            "bmi": _to_float(profile.bmi),
            "updated_at": profile.updated_at.isoformat() if profile.updated_at else None,
        }


def summarize(portrait: dict[str, Any]) -> str:
    """把画像 dict 压成一行中文摘要，供注入 LLM 上下文；无有效字段返回「暂无画像信息」。"""
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

    if not parts:
        return "暂无画像信息"
    return "，".join(parts)


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
