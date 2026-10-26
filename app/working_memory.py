"""Redis 会话工作记忆：当前会话轮次/已采集槽位（与 checkpointer 职责分离）。
checkpointer 持久化「图执行状态」（挂起在哪、下一步是什么）；这里存「业务会话态」
（这一场问过什么、轮到第几轮），TTL 到期或会话结束即清。Redis 不可用时静默降级，
不影响主流程——长期记忆在 Postgres 情景/语义层，不依赖这里。
"""
import json
import logging
from typing import Any
from uuid import UUID

import redis.asyncio as redis

from app.config import settings

logger = logging.getLogger(__name__)


class WorkingMemory:
    """按 (user_id, thread_id) 存/取/清会话工作记忆，Redis 不可用时降级为 no-op。"""

    def __init__(
        self,
        url: str,
        ttl_seconds: int,
        client: redis.Redis | None = None,
    ) -> None:
        self._url = url
        self._ttl_seconds = ttl_seconds
        self._client = client

    def _get_client(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.from_url(
                self._url,
                decode_responses=True,
                socket_connect_timeout=1,
                socket_timeout=1,
            )
        return self._client

    @staticmethod
    def _key(user_id: UUID, thread_id: str) -> str:
        return f"ht:wm:{user_id}:{thread_id}"

    async def set(
        self,
        user_id: UUID,
        thread_id: str,
        *,
        round_: int = 0,
        collected_slots: list[str] | None = None,
    ) -> None:
        """写入本轮轮次与已采集槽位，并刷新 TTL。"""
        try:
            client = self._get_client()
            key = self._key(user_id, thread_id)
            await client.hset(
                key,
                mapping={
                    "round": str(round_),
                    "collected_slots": json.dumps(
                        collected_slots or [], ensure_ascii=False
                    ),
                },
            )
            await client.expire(key, self._ttl_seconds)
        except Exception as exc:  # Redis 不可用：降级，不阻断主流程
            logger.warning("工作记忆 set 降级：%s", exc)

    async def get(self, user_id: UUID, thread_id: str) -> dict[str, Any]:
        """读取当前会话工作记忆；无记录或 Redis 不可用时返回空 dict。"""
        try:
            data = await self._get_client().hgetall(self._key(user_id, thread_id))
            if not data:
                return {}
            return {
                "round": int(data.get("round", 0)),
                "collected_slots": json.loads(data.get("collected_slots", "[]")),
            }
        except Exception as exc:
            logger.warning("工作记忆 get 降级：%s", exc)
            return {}

    async def clear(self, user_id: UUID, thread_id: str) -> None:
        """会话结束时清除工作记忆。"""
        try:
            await self._get_client().delete(self._key(user_id, thread_id))
        except Exception as exc:
            logger.warning("工作记忆 clear 降级：%s", exc)


working_memory = WorkingMemory(
    settings.redis_url,
    ttl_seconds=settings.working_memory_ttl_seconds,
)
