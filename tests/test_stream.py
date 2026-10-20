"""SSE 事件流：route -> chat token -> done 的确定性断言。"""
import json

from langchain_core.messages import AIMessageChunk, ToolMessage

from app.agent.stream import chat_event_stream
from app.schemas import RouteDecision


class _FakeRouterRunnable:
    def __init__(self, decision: RouteDecision) -> None:
        self.decision = decision

    async def ainvoke(self, messages) -> RouteDecision:
        return self.decision


class _FakeRouterLLM:
    def __init__(self, decision: RouteDecision) -> None:
        self.decision = decision

    def with_structured_output(self, schema, method) -> _FakeRouterRunnable:
        return _FakeRouterRunnable(self.decision)


class _FakeChatGraph:
    def __init__(self, chunks) -> None:
        self._chunks = chunks

    def astream(self, input, stream_mode=None):
        async def gen():
            for chunk in self._chunks:
                yield (chunk, {})

        return gen()


async def _collect(stream):
    return [frame async for frame in stream]


async def test_chat_stream_filters_tool_message_and_emits_order() -> None:
    """chat 意图：过滤 ToolMessage，按 route/chat/chat/done 顺序产出。"""
    router = _FakeRouterLLM(
        RouteDecision(intent="chat", confidence=0.9, extracted_slots={})
    )
    chat = _FakeChatGraph(
        [
            ToolMessage(content='{"ok": true, "id": 1}', tool_call_id="call_1"),
            AIMessageChunk(content="已"),
            AIMessageChunk(content="记录"),
        ]
    )

    frames = await _collect(chat_event_stream("昨晚睡了 6 小时", router, chat))
    events = [json.loads(f[len("data:"):].strip()) for f in frames]

    assert [e["event"] for e in events] == ["route", "chat", "chat", "done"]
    assert events[0]["data"]["intent"] == "chat"
    assert events[1]["data"] == "已"
    assert events[2]["data"] == "记录"
    assert events[3]["data"]["reply"] == "已记录"


async def test_chat_stream_non_chat_uses_stub() -> None:
    """非 chat 意图：走 stub，回复含“后续周实现”。"""
    router = _FakeRouterLLM(
        RouteDecision(intent="triage", confidence=0.9, extracted_slots={})
    )
    chat = _FakeChatGraph([])

    frames = await _collect(chat_event_stream("最近总是累", router, chat))
    events = [json.loads(f[len("data:"):].strip()) for f in frames]

    assert [e["event"] for e in events] == ["route", "chat", "done"]
    assert "后续周实现" in events[1]["data"]
