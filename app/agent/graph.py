"""父图（遗留 chat 路径）：Router 结构化输出 -> chat 子图或占位节点。

统一入口见 app/agent/entry.py 的 dispatch：那里把 triage 也路由到追问分诊子图；
本父图仅供非流式 /chat 使用，triage/insight/report 均回占位。
"""
from typing import Annotated, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.agent.chat import build_chat_graph
from app.agent.llm import get_llm
from app.agent.router import route

# 占位回复：insight/report 子图未实现（triage 由统一入口 /agent 分发）
_STUB_TEMPLATE = "你这句话被路由到「{intent}」意图，该能力将在后续周实现，本周暂不支持。"


def stub_reply(intent: str) -> str:
    """非 chat 意图的占位回复。"""
    return _STUB_TEMPLATE.format(intent=intent)


class AgentState(TypedDict):
    """父图状态：对话消息 + 路由结果。"""

    messages: Annotated[list, add_messages]
    intent: str
    confidence: float
    extracted_slots: dict


def build_agent_graph(
    router_llm: BaseChatModel | None = None,
    chat_graph=None,
):
    """构建父图：route 节点 -> 条件边 -> chat 子图 / stub 节点。

    router_llm 与 chat_graph 均可注入，便于用 fake 做确定性集成测试。
    """
    if router_llm is None:
        router_llm = get_llm()
    if chat_graph is None:
        chat_graph = build_chat_graph()

    async def route_node(state: AgentState) -> dict:
        # 数据准备：取最后一条用户消息原文做意图分类
        text = state["messages"][-1].content
        decision = await route(text, router_llm)
        return {
            "intent": decision.intent,
            "confidence": decision.confidence,
            "extracted_slots": decision.extracted_slots,
        }

    async def stub_node(state: AgentState) -> dict:
        # 非 chat 意图占位：返回一句后续周实现的提示，不崩溃
        reply = stub_reply(state["intent"])
        return {"messages": [AIMessage(content=reply)]}

    graph = StateGraph(AgentState)
    graph.add_node("route", route_node)
    graph.add_node("chat", chat_graph)
    graph.add_node("stub", stub_node)

    graph.add_edge(START, "route")
    # 条件边：chat 进 chat 子图，其余进 stub
    graph.add_conditional_edges(
        "route",
        lambda state: state["intent"],
        {"chat": "chat", "triage": "stub", "insight": "stub", "report": "stub"},
    )
    graph.add_edge("chat", END)
    graph.add_edge("stub", END)

    return graph.compile()
