"""FastAPI 应用入口。"""
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app import crud
from app.agent.chat import build_chat_graph
from app.agent.graph import build_agent_graph
from app.agent.llm import get_llm
from app.agent.stream import chat_event_stream
from app.agent.triage import build_triage_graph, initial_triage_state
from app.checkpoint import open_checkpointer
from app.config import settings
from app.observability import get_tracer, init_observability
from app.schemas import (
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


@app.post("/metrics", response_model=MetricResponse)
async def create_metric(
    payload: MetricCreate,
) -> MetricResponse:
    """写入一条指标记录（同日期同指标覆盖）。"""
    with tracer.start_as_current_span("record_metric") as root_span:
        root_span.set_attribute("langfuse.trace.name", "记录健康指标")
        root_span.set_attribute("langfuse.user.id", str(settings.default_user_id))

        record_id = await crud.upsert_metric(
            user_id=settings.default_user_id,
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
        root_span.set_attribute("langfuse.user.id", str(settings.default_user_id))

        result = await agent_graph.ainvoke(
            {"messages": [HumanMessage(content=payload.text)]}
        )
        reply = result["messages"][-1].content

        return ChatResponse(
            intent=result["intent"],
            confidence=result["confidence"],
            reply=reply,
        )


@app.post("/chat/stream")
async def chat_stream(payload: ChatRequest) -> StreamingResponse:
    """SSE 流式对话：route -> chat token 片段 -> done。"""
    return StreamingResponse(
        chat_event_stream(payload.text, router_llm, chat_graph),
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
        root_span.set_attribute("langfuse.user.id", str(settings.default_user_id))
        if payload.thread_id is None:
            result = await graph.ainvoke(initial_triage_state(payload.text), config=config)
        else:
            result = await graph.ainvoke(Command(resume=payload.text), config=config)

    # 图挂起（interrupt）：返回追问问题，等下一轮 resume
    interrupts = result.get("__interrupt__")
    candidate_conditions = result.get("candidate_conditions", [])
    candidate_count = result.get("candidate_count", len(candidate_conditions))
    top_tier_count = result.get("top_tier_count", 0)
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
