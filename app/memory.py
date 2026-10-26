"""情景记忆：统一会话记录（chat/triage/insight/report）的写、查、删（session 自管）。"""
import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import delete, select

from app.db import async_session
from app.models import EpisodicMemory


def _to_dict(row: EpisodicMemory) -> dict[str, Any]:
    """ORM 行 -> JSON 可序列化 dict。"""
    return {
        "id": row.id,
        "user_id": str(row.user_id),
        "subgraph": row.subgraph,
        "session_id": row.session_id,
        "user_text": row.user_text,
        "summary": row.summary,
        "symptom_text": row.symptom_text,
        "final_level": row.final_level,
        "red_flag_triggered": row.red_flag_triggered,
        "payload": row.payload,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


async def record_episodic(
    user_id: UUID,
    subgraph: str,
    user_text: str | None,
    summary: str | None = None,
    *,
    session_id: str | None = None,
    symptom_text: str | None = None,
    final_level: str | None = None,
    red_flag_triggered: bool | None = None,
    payload: dict | None = None,
) -> int:
    """写入一条情景记忆，返回记录 id（created_at 由应用侧显式设置）。"""
    async with async_session() as session:
        row = EpisodicMemory(
            user_id=user_id,
            subgraph=subgraph,
            session_id=session_id,
            user_text=user_text,
            summary=summary,
            symptom_text=symptom_text,
            final_level=final_level,
            red_flag_triggered=red_flag_triggered,
            payload=payload,
            created_at=datetime.datetime.now(datetime.timezone.utc),
        )
        session.add(row)
        await session.commit()
        return row.id


async def list_episodic(
    user_id: UUID,
    subgraph: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """按用户检索情景记忆（可选按 subgraph 过滤，按创建时间倒序），返回 dict 列表。"""
    stmt = select(EpisodicMemory).where(EpisodicMemory.user_id == user_id)
    if subgraph:
        stmt = stmt.where(EpisodicMemory.subgraph == subgraph)
    stmt = stmt.order_by(EpisodicMemory.id.desc()).limit(limit)
    async with async_session() as session:
        rows = (await session.execute(stmt)).scalars().all()
    return [_to_dict(row) for row in rows]


async def delete_episodic(user_id: UUID, memory_id: int) -> bool:
    """按用户 + 记录 id 删除一条情景记忆，返回是否真的删除了。"""
    async with async_session() as session:
        result = await session.execute(
            delete(EpisodicMemory).where(
                EpisodicMemory.id == memory_id,
                EpisodicMemory.user_id == user_id,
            )
        )
        await session.commit()
        return result.rowcount > 0


def summarize_episodic(memories: list[dict]) -> str:
    """把最近情景记忆压成一段中文摘要，供新会话注入；无记录返回空串。"""
    lines: list[str] = []
    for item in memories[:5]:
        subgraph = item.get("subgraph", "")
        pieces: list[str] = []
        summary = (item.get("summary") or "").strip()
        if summary:
            pieces.append(summary)
        symptom_text = (item.get("symptom_text") or "").strip()
        if symptom_text:
            pieces.append(f"症状：{symptom_text}")
        if pieces:
            lines.append(f"[{subgraph}] " + "；".join(pieces))
    return "\n".join(lines)
