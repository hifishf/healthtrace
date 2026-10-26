"""第 7 周 Step 2：统一情景记忆的写、检索、删除（按用户）。"""
from app import memory
from app.config import settings

UID = settings.default_user_id


async def test_record_and_list_episodic_by_subgraph() -> None:
    """chat/triage 各记一条后，按用户可检索、可按 subgraph 过滤、后写的排前。"""
    created: list[int] = []
    try:
        chat_id = await memory.record_episodic(
            UID, "chat", "昨晚睡了 6 小时", summary="已记录"
        )
        triage_id = await memory.record_episodic(
            UID,
            "triage",
            "最近总是累",
            summary="请问最近有没有乏力的情况？",
            session_id="s-ep1",
        )
        created = [chat_id, triage_id]

        all_memories = await memory.list_episodic(UID)
        ids = {m["id"] for m in all_memories}
        assert chat_id in ids and triage_id in ids

        triage_idx = next(i for i, m in enumerate(all_memories) if m["id"] == triage_id)
        chat_idx = next(i for i, m in enumerate(all_memories) if m["id"] == chat_id)
        assert triage_idx < chat_idx

        chat_only = await memory.list_episodic(UID, subgraph="chat")
        assert all(m["subgraph"] == "chat" for m in chat_only)
        assert any(m["id"] == chat_id for m in chat_only)
    finally:
        for mid in created:
            await memory.delete_episodic(UID, mid)


async def test_delete_episodic_removes_row() -> None:
    """删除后检索不到；对已删记录再删返回 False。"""
    mid = await memory.record_episodic(UID, "chat", "临时记录", summary="待删")
    try:
        assert await memory.delete_episodic(UID, mid) is True
        remaining = await memory.list_episodic(UID)
        assert all(m["id"] != mid for m in remaining)
        assert await memory.delete_episodic(UID, mid) is False
    finally:
        await memory.delete_episodic(UID, mid)


async def test_record_episodic_payload_roundtrip() -> None:
    """payload 以 JSONB 存、读回后结构一致。"""
    payload = {"kind": "advice", "level": "seek_care", "conditions": ["偏头痛"]}
    mid = await memory.record_episodic(
        UID,
        "triage",
        "胸痛",
        summary="建议就医",
        payload=payload,
        final_level="seek_care",
    )
    try:
        memories = await memory.list_episodic(UID, subgraph="triage")
        row = next(m for m in memories if m["id"] == mid)
        assert row["payload"] == payload
        assert row["final_level"] == "seek_care"
    finally:
        await memory.delete_episodic(UID, mid)
