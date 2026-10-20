"""健康画像读写：读画像字段、刷新 updated_at。"""
import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import HealthProfile


def _to_float(value: Decimal | None) -> float | None:
    """Decimal -> float（JSON 序列化用），None 保持 None。"""
    return float(value) if value is not None else None


async def get_health_portrait(session: AsyncSession, user_id: UUID) -> dict[str, Any]:
    """读取健康画像，返回 JSON 可序列化的 dict；无画像时返回空 dict。"""
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


async def touch_portrait(session: AsyncSession, user_id: UUID) -> None:
    """刷新画像 updated_at（表示画像最近一次被更新）。

    无画像行时静默跳过（单用户种子已存在，正常不会发生）。
    """
    profile = await session.get(HealthProfile, user_id)
    if profile is None:
        return
    profile.updated_at = datetime.datetime.now(datetime.timezone.utc)
    await session.commit()
