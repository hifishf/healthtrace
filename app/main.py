"""FastAPI 应用入口。"""
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app import crud, memory
from app.agent.chat import build_chat_graph
from app.agent.entry import dispatch
from app.agent.graph import build_agent_graph
from app.agent.llm import get_llm
from app.agent.stream import chat_event_stream
from app.agent.triage import build_triage_graph, initial_triage_state
from app.checkpoint import open_checkpointer
from app.config import settings
from app.identity import get_current_user_id
from app.observability import get_tracer, init_observability
from app.schemas import (
    AgentRequest,
    AgentResponse,
    ChatRequest,
    ChatResponse,
    MetricCreate,
    MetricResponse,
    TriageRequest,
    TriageResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：打开 Postgres checkpointer 并构建追问分诊图（interrupt 依赖它）。"""
    async with open_checkpointer() as saver:
        app.state.triage_graph = build_triage_graph(checkpointer=saver)
        yield


app = FastAPI(title="HealthTrace API", version="0.1.0", lifespan=lifespan)

init_observability()
tracer = get_tracer()

# Router LLM + chat 子图 + 父图：模块级构建一次，复用（流式与非流式共用）
router_llm = get_llm()
chat_graph = build_chat_graph()
agent_graph = build_agent_graph(router_llm=router_llm, chat_graph=chat_graph)


def _agent_summary(result: dict) -> str | None:
    """从统一入口结果里取一句摘要（chat/stub 取 reply，triage 取追问或建议）。"""
    if result.get("kind") in ("reply", "stub"):
        return result.get("reply")
    if result.get("kind") == "question":
        return result.get("question")
    advice = result.get("final_advice")
    return advice.get("suggestion") if advice else None


def _conditions_text(advice: dict | None) -> str | None:
    """把 final_advice 里的候选方向拼成文本（情景记忆的 symptom_text 用）。"""
    if not advice:
        return None
    conditions = advice.get("conditions") or []
    return "、".join(conditions) if conditions else None


@app.post("/metrics", response_model=MetricResponse)
async def create_metric(
    payload: MetricCreate,
) -> MetricResponse:
    """写入一条指标记录（同日期同指标覆盖）。"""
    with tracer.start_as_current_span("record_metric") as root_span:
        root_span.set_attribute("langfuse.trace.name", "记录健康指标")
        root_span.set_attribute("langfuse.user.id", str(get_current_user_id()))

        record_id = await crud.upsert_metric(
            user_id=get_current_user_id(),
            metric_code=payload.metric_code,
            value=payload.value,
            unit=payload.unit,
            recorded_date=payload.recorded_date,
        )

        # mock 一次 LLM 调用，先占住 trace 链路；第 2 周接真实 LLM 时替换
        with tracer.start_as_current_span("mock_llm_call") as llm_span:
            llm_span.set_attribute("langfuse.observation.type", "generation")
            llm_span.set_attribute("gen_ai.system", "mock")
            llm_span.set_attribute("gen_ai.request.model", "mock-model")
            llm_span.set_attribute(
                "langfuse.observation.input",
                f"记录 {payload.metric_code}={payload.value}{payload.unit}",
            )
            llm_span.set_attribute("langfuse.observation.output", "ok")

        return MetricResponse(ok=True, id=record_id)


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    """对话入口：跑一遍父图（Router -> chat/占位），返回路由结果与回复。"""
    with tracer.start_as_current_span("chat") as root_span:
        root_span.set_attribute("langfuse.trace.name", "健康对话")
        root_span.set_attribute("langfuse.user.id", str(get_current_user_id()))

        result = await agent_graph.ainvoke(
            {"messages": [HumanMessage(content=payload.text)]}
        )
        reply = result["messages"][-1].content
        await memory.record_episodic(
            user_id=get_current_user_id(),
            subgraph=result.get("intent", "chat"),
            user_text=payload.text,
            summary=reply,
            payload={
                "intent": result.get("intent"),
                "confidence": result.get("confidence"),
                "reply": reply,
            },
        )

        return ChatResponse(
            intent=result["intent"],
            confidence=result["confidence"],
            reply=reply,
        )


@app.post("/agent", response_model=AgentResponse)
async def agent(payload: AgentRequest, request: Request) -> AgentResponse:
    """统一入口：按意图把一句话路由到 chat / triage（insight / report 回 stub）。"""
    with tracer.start_as_current_span("agent") as root_span:
        root_span.set_attribute("langfuse.trace.name", "统一入口")
        root_span.set_attribute("langfuse.user.id", str(get_current_user_id()))

        result = await dispatch(
            payload.text,
            router_llm=router_llm,
            chat_graph=chat_graph,
            triage_graph=request.app.state.triage_graph,
            thread_id=payload.thread_id,
        )
        if result.get("thread_id"):
            root_span.set_attribute("langfuse.session.id", result["thread_id"])
        await memory.record_episodic(
            user_id=get_current_user_id(),
            subgraph=result["intent"],
            session_id=result.get("thread_id"),
            user_text=payload.text,
            summary=_agent_summary(result),
            symptom_text=_conditions_text(result.get("final_advice")),
            final_level=(result.get("final_advice") or {}).get("level"),
            red_flag_triggered=result.get("red_flag_triggered"),
            payload=result,
        )
        return AgentResponse(**result)


@app.post("/chat/stream")
async def chat_stream(payload: ChatRequest) -> StreamingResponse:
    """SSE 流式对话：route -> chat token 片段 -> done，并在结束时落情景记忆。"""

    async def record_done(intent: str, reply: str) -> None:
        await memory.record_episodic(
            user_id=get_current_user_id(),
            subgraph=intent,
            user_text=payload.text,
            summary=reply,
            payload={"intent": intent, "reply": reply},
        )

    return StreamingResponse(
        chat_event_stream(payload.text, router_llm, chat_graph, on_done=record_done),
        media_type="text/event-stream",
    )


@app.post("/triage", response_model=TriageResponse)
async def triage(payload: TriageRequest, request: Request) -> TriageResponse:
    """追问分诊入口：首次调用启动会话；带 thread_id 的调用用 Command(resume) 续跑。"""
    graph = request.app.state.triage_graph
    thread_id = payload.thread_id or str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    with tracer.start_as_current_span("triage") as root_span:
        root_span.set_attribute("langfuse.trace.name", "追问分诊")
        root_span.set_attribute("langfuse.session.id", thread_id)
        root_span.set_attribute("langfuse.user.id", str(get_current_user_id()))
        if payload.thread_id is None:
            result = await graph.ainvoke(initial_triage_state(payload.text), config=config)
        else:
            result = await graph.ainvoke(Command(resume=payload.text), config=config)

    # 图挂起（interrupt）：返回追问问题，等下一轮 resume
    interrupts = result.get("__interrupt__")
    candidate_conditions = result.get("candidate_conditions", [])
    candidate_count = result.get("candidate_count", len(candidate_conditions))
    top_tier_count = result.get("top_tier_count", 0)
    advice = result.get("final_advice")
    await memory.record_episodic(
        user_id=get_current_user_id(),
        subgraph="triage",
        session_id=thread_id,
        user_text=payload.text,
        summary=(
            interrupts[0].value.get("question")
            if interrupts
            else (advice or {}).get("suggestion")
        ),
        symptom_text=_conditions_text(advice),
        final_level=(advice or {}).get("level"),
        red_flag_triggered=result.get("red_flag_triggered", False),
        payload={
            "kind": "question" if interrupts else "advice",
            "final_advice": advice,
            "candidate_conditions": candidate_conditions,
        },
    )
    if interrupts:
        return TriageResponse(
            thread_id=thread_id,
            kind="question",
            question=interrupts[0].value.get("question"),
            candidate_count=candidate_count,
            top_tier_count=top_tier_count,
            candidate_conditions=candidate_conditions,
        )

    return TriageResponse(
        thread_id=thread_id,
        kind="advice",
        final_advice=result.get("final_advice"),
        red_flag_triggered=result.get("red_flag_triggered", False),
        candidate_count=candidate_count,
        top_tier_count=top_tier_count,
        candidate_conditions=candidate_conditions,
    )
