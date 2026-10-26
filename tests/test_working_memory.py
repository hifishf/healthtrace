"""第 7 周 Step 3：Redis 会话工作记忆的写/读/清/TTL 与降级。"""
import json
import uuid

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.agent.triage import build_triage_graph, initial_triage_state
from app.config import settings
from app.schemas import SymptomExtraction
from app.working_memory import WorkingMemory

UID = settings.default_user_id


class _FakeRedis:
    """内存版 redis.asyncio 替身，只实现 WorkingMemory 用到的几个方法。"""

    def __init__(self, fail: bool = False) -> None:
        self.store: dict[str, dict] = {}
        self.expirations: dict[str, int] = {}
        self.fail = fail

    async def hset(self, key, mapping=None):
        if self.fail:
            raise RuntimeError("redis down")
        self.store.setdefault(key, {}).update(mapping or {})

    async def expire(self, key, seconds):
        if self.fail:
            raise RuntimeError("redis down")
        self.expirations[key] = seconds

    async def hgetall(self, key):
        if self.fail:
            raise RuntimeError("redis down")
        return dict(self.store.get(key, {}))

    async def delete(self, key):
        if self.fail:
            raise RuntimeError("redis down")
        return bool(self.store.pop(key, None))


async def test_working_memory_set_get_roundtrip() -> None:
    """写入轮次+槽位后能原样读回。"""
    wm = WorkingMemory("redis://unused", 3600, client=_FakeRedis())
    await wm.set(UID, "t1", round_=2, collected_slots=["乏力", "失眠"])

    data = await wm.get(UID, "t1")
    assert data == {"round": 2, "collected_slots": ["乏力", "失眠"]}


async def test_working_memory_get_missing_is_empty() -> None:
    """无记录时返回空 dict。"""
    wm = WorkingMemory("redis://unused", 3600, client=_FakeRedis())
    assert await wm.get(UID, "missing") == {}


async def test_working_memory_clear_removes() -> None:
    """clear 后读不到。"""
    wm = WorkingMemory("redis://unused", 3600, client=_FakeRedis())
    await wm.set(UID, "t1", round_=1, collected_slots=["乏力"])
    await wm.clear(UID, "t1")
    assert await wm.get(UID, "t1") == {}


async def test_working_memory_set_refreshes_ttl() -> None:
    """set 会设置 TTL（到期自动清理的兜底）。"""
    fake = _FakeRedis()
    wm = WorkingMemory("redis://unused", 3600, client=fake)
    await wm.set(UID, "t1", round_=1, collected_slots=["乏力"])
    assert fake.expirations.get(f"ht:wm:{UID}:t1") == 3600


async def test_working_memory_degrades_when_redis_down() -> None:
    """Redis 不可用时 set/get/clear 都不抛异常，get 返回空 dict。"""
    wm = WorkingMemory("redis://unused", 3600, client=_FakeRedis(fail=True))
    await wm.set(UID, "t1", round_=1, collected_slots=["乏力"])
    assert await wm.get(UID, "t1") == {}
    await wm.clear(UID, "t1")


class _FakeStructuredRunnable:
    async def ainvoke(self, messages) -> SymptomExtraction:
        return SymptomExtraction(symptoms=["乏力"], subject="self")


class _FakeLLM:
    def with_structured_output(self, schema, method):
        return _FakeStructuredRunnable()


async def test_triage_updates_working_memory(monkeypatch) -> None:
    """追问分诊跑一轮后，把轮次+已采集槽位写进工作记忆（与 checkpointer 分离）。"""
    fake = _FakeRedis()
    wm = WorkingMemory("redis://unused", 3600, client=fake)
    monkeypatch.setattr("app.agent.triage.working_memory", wm)

    async with AsyncPostgresSaver.from_conn_string(
        settings.checkpoint_database_url
    ) as saver:
        await saver.setup()
        graph = build_triage_graph(llm=_FakeLLM(), checkpointer=saver)
        tid = f"wm-{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": tid}}
        result = await graph.ainvoke(initial_triage_state("最近总是累"), config=config)

    assert result.get("__interrupt__") is not None
    key = f"ht:wm:{UID}:{tid}"
    assert key in fake.store
    assert json.loads(fake.store[key]["collected_slots"]) == ["乏力"]
    assert fake.store[key]["round"] == "0"
