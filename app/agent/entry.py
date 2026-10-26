"""统一入口：按 Router 意图把一句话分发到 chat / triage 子图（insight / report 回 stub）。"""
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app.agent.graph import stub_reply
from app.agent.router import route
from app.agent.triage import initial_triage_state


def _triage_payload(tid: str, result: dict) -> dict:
    """把 triage 子图结果转成统一入口的返回结构（question / advice）。"""
    interrupts = result.get("__interrupt__")
    candidate_conditions = result.get("candidate_conditions", [])
    candidate_count = result.get("candidate_count", len(candidate_conditions))
    top_tier_count = result.get("top_tier_count", 0)
    base = {
        "intent": "triage",
        "thread_id": tid,
        "candidate_count": candidate_count,
        "top_tier_count": top_tier_count,
        "candidate_conditions": candidate_conditions,
    }
    if interrupts:
        return {
            **base,
            "kind": "question",
            "question": interrupts[0].value.get("question"),
        }
    return {
        **base,
        "kind": "advice",
        "final_advice": result.get("final_advice"),
        "red_flag_triggered": result.get("red_flag_triggered", False),
    }


async def dispatch(text, router_llm, chat_graph, triage_graph, thread_id=None) -> dict:
    """统一入口：路由意图后分发到对应子图。

    - chat：跑 chat 子图，返回最终回复；
    - triage：跑追问分诊子图（首次 / 带 thread_id 续跑），返回追问或分级建议；
    - insight / report：返回占位 stub。
    """
    decision = await route(text, router_llm)
    intent = decision.intent

    if intent == "chat":
        result = await chat_graph.ainvoke({"messages": [HumanMessage(content=text)]})
        return {
            "intent": intent,
            "kind": "reply",
            "reply": result["messages"][-1].content,
        }

    if intent == "triage":
        tid = thread_id or str(uuid4())
        config = {"configurable": {"thread_id": tid}}
        if thread_id is None:
            result = await triage_graph.ainvoke(initial_triage_state(text), config=config)
        else:
            result = await triage_graph.ainvoke(Command(resume=text), config=config)
        return _triage_payload(tid, result)

    return {"intent": intent, "kind": "stub", "reply": stub_reply(intent)}
