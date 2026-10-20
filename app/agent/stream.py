"""SSE 事件流：把 Router + chat 子图串成可流式返回的事件生成器。"""
import json
from typing import Any, AsyncIterator

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, HumanMessage

from app.agent.graph import stub_reply
from app.agent.router import route


def sse_event(event: str, data: Any) -> str:
    """把一个事件编码成一条 SSE 数据帧（data: {json}\\n\\n）。"""
    payload = json.dumps({"event": event, "data": data}, ensure_ascii=False)
    return f"data: {payload}\n\n"


async def chat_event_stream(
    text: str,
    router_llm: BaseChatModel,
    chat_graph,
) -> AsyncIterator[str]:
    """按顺序产出 route / chat / done 三类 SSE 事件。

    Router 非流式跑一次；chat 意图用 astream(stream_mode="messages")
    逐 token 产出最终回复。过滤掉 ToolMessage 与空内容，
    只把最终回复的 token 作为 chat 事件推给前端。
    """
    decision = await route(text, router_llm)
    yield sse_event(
        "route", {"intent": decision.intent, "confidence": decision.confidence}
    )

    reply_parts: list[str] = []
    if decision.intent == "chat":
        async for chunk, _ in chat_graph.astream(
            {"messages": [HumanMessage(content=text)]},
            stream_mode="messages",
        ):
            if (
                isinstance(chunk, AIMessageChunk)
                and isinstance(chunk.content, str)
                and chunk.content
            ):
                reply_parts.append(chunk.content)
                yield sse_event("chat", chunk.content)
    else:
        reply = stub_reply(decision.intent)
        reply_parts.append(reply)
        yield sse_event("chat", reply)

    yield sse_event(
        "done",
        {"intent": decision.intent, "reply": "".join(reply_parts)},
    )
