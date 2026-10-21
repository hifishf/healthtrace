"""追问分诊子图：TriageState 状态机 + interrupt/Command 挂起-恢复 + 硬上限。"""
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from app import portrait
from app.agent.extraction import extract_symptoms
from app.agent.llm import get_llm
from app.agent.red_flags import check_red_flag
from app.config import settings
from app.db import async_session
from app.kb import (
    ADVICE_LEVELS,
    classify_advice_level_by_hints,
    compute_candidates,
    get_symptom_names,
    match_symptoms,
    select_followup_questions,
)

# 追问硬上限（状态机边条件，不依赖 LLM 自觉）
MAX_FOLLOWUP_ROUNDS = 3


class TriageState(TypedDict):
    """追问分诊状态（第 7.2 节冻结契约）。"""

    messages: Annotated[list, add_messages]
    health_portrait: dict
    collected_symptoms: dict  # str(symptom_id) -> 症状名
    ruled_out_symptoms: dict  # str(symptom_id) -> 症状名（已明确排除，不再追问）
    candidate_conditions: list  # [{"id", "name", "urgency_hint"}]
    followup_round: int
    red_flag_triggered: bool
    final_advice: dict | None
    pending_question: dict | None  # 下一轮追问（内部实现细节）


_EMERGENCY_ADVICE: dict = {
    "level": "emergency",
    "label": "立即就医",
    "conditions": [],
    "suggestion": "你描述的症状属于紧急情况，请立即就医或拨打急救电话，不要等待线上建议。",
}

_ACCOMPANY_ADVICE: dict = {
    "level": "accompany",
    "label": "建议陪同就医",
    "conditions": [],
    "suggestion": "这是他人的症状，无法代替本人分诊，建议陪同就医。",
}

_SUGGESTIONS: dict[str, str] = {
    "observe": "根据你描述的症状，目前倾向先观察休息并注意变化，如症状加重请及时就医。",
    "adjust": "建议先调整生活方式（规律作息、均衡饮食、适量运动），观察症状是否缓解。",
    "online_consult": "建议进行线上问诊，由医生进一步评估和指导。",
    "seek_care": "建议尽快到医院就诊，由医生进一步检查明确原因。",
}

# 追问回答的肯定/否定表面形式（确定性，不依赖 LLM；否定优先于肯定，
# 因为“没有”里包含“有”，先判否定才不会被误判成肯定）
_NEGATIVE_MARKERS: tuple[str, ...] = ("没有", "没", "无", "不是", "不会", "否")
_AFFIRMATIVE_MARKERS: tuple[str, ...] = ("有", "是", "对", "嗯", "会", "出现过", "偶尔会")


def answer_polarity(answer: str) -> str:
    """判定追问回答的极性：affirm / deny / unknown（确定性，否定优先）。"""
    value = (answer or "").strip()
    if not value:
        return "unknown"
    for marker in _NEGATIVE_MARKERS:
        if marker in value:
            return "deny"
    for marker in _AFFIRMATIVE_MARKERS:
        if marker in value:
            return "affirm"
    return "unknown"


def initial_triage_state(text: str) -> dict:
    """构造一次追问分诊的初始状态（单用户，无登录）。"""
    return {
        "messages": [HumanMessage(content=text)],
        "health_portrait": {},
        "collected_symptoms": {},
        "ruled_out_symptoms": {},
        "candidate_conditions": [],
        "followup_round": 0,
        "red_flag_triggered": False,
        "final_advice": None,
        "pending_question": None,
    }


def _build_advice(level: str, candidates: list[dict]) -> dict:
    """由确定性级别组装 final_advice（LLM 转写在后续周接入）。"""
    return {
        "level": level,
        "label": ADVICE_LEVELS[level],
        "conditions": [candidate["name"] for candidate in candidates],
        "suggestion": _SUGGESTIONS.get(level, "建议咨询医生进一步评估。"),
    }


def route_after_extract(state: TriageState) -> str:
    """extract 之后的确定性路由：红旗/他人/硬上限/无可追问 -> 建议，否则追问。"""
    if state.get("red_flag_triggered") or state.get("final_advice") is not None:
        return "advise"
    if state.get("followup_round", 0) >= MAX_FOLLOWUP_ROUNDS:
        return "advise"
    if state.get("pending_question") is None:
        return "advise"
    return "ask"


def ask_node(state: TriageState) -> dict:
    """挂起等用户回答下一轮追问，恢复后记录回答并递增轮次。"""
    question = state["pending_question"]
    answer = interrupt(
        {
            "type": "followup_question",
            "question": question["text"],
            "round": state["followup_round"],
        }
    )
    return {
        "messages": [HumanMessage(content=answer)],
        "followup_round": state["followup_round"] + 1,
    }


def advise_node(state: TriageState) -> dict:
    """生成最终分级建议（红旗/他人已在 extract 里定好，这里只补软建议）。"""
    if state.get("final_advice") is not None:
        return {}
    level = classify_advice_level_by_hints(
        [candidate.get("urgency_hint") for candidate in state["candidate_conditions"]]
    )
    return {"final_advice": _build_advice(level, state["candidate_conditions"])}


def build_triage_graph(llm=None, checkpointer=None):
    """构建追问分诊子图（extract -> ask(interrupt) 循环 -> advise）。

    llm 可注入 fake 做确定性测试；checkpointer 必须提供（interrupt 依赖断点持久化）。
    """
    if llm is None:
        llm = get_llm()

    async def extract_node(state: TriageState) -> dict:
        """入口：加载画像 -> 抽症状 -> 红旗/主体边界 -> 归一化 -> 候选集 -> 选下一问。"""
        text = state["messages"][-1].content
        # 上一轮追问（若本轮是对它的回答，需确定性回填/记排除）
        pending_q = state.get("pending_question")
        answered_polarity = answer_polarity(text) if pending_q else "unknown"

        async with async_session() as session:
            loaded_portrait = await portrait.get_health_portrait(
                session, settings.default_user_id
            )

        extraction = await extract_symptoms(
            text, llm, portrait_summary=portrait.summarize(loaded_portrait)
        )
        red_flag = check_red_flag(text, extraction.subject)

        if red_flag.triggered:
            return {
                "health_portrait": loaded_portrait,
                "red_flag_triggered": True,
                "final_advice": _EMERGENCY_ADVICE,
            }
        if extraction.subject == "other":
            return {
                "health_portrait": loaded_portrait,
                "final_advice": _ACCOMPANY_ADVICE,
            }

        async with async_session() as session:
            symptom_ids = await match_symptoms(session, extraction.symptoms)
            # 肯定回答回填：答「有」时 LLM 抽不到症状，直接用上一轮 pending 症状补上
            if answered_polarity == "affirm" and pending_q and pending_q.get("id") is not None:
                pending_id = int(pending_q["id"])
                if pending_id not in symptom_ids:
                    symptom_ids.append(pending_id)
            names = await get_symptom_names(session, symptom_ids)
            collected = dict(state["collected_symptoms"])
            for symptom_id in symptom_ids:
                collected[str(symptom_id)] = names.get(symptom_id, "")
            collected_ids = [int(key) for key in collected]
            # 否定回答记排除：答「没有」的症状不再追问
            ruled_out = dict(state.get("ruled_out_symptoms", {}))
            if answered_polarity == "deny" and pending_q and pending_q.get("id") is not None:
                pending_id = int(pending_q["id"])
                ruled_out[str(pending_id)] = (
                    pending_q.get("name") or names.get(pending_id, "")
                )
            ruled_out_ids = [int(key) for key in ruled_out]
            candidates = await compute_candidates(session, collected_ids)
            # 已到追问硬上限：不再选下一问，避免最后一轮做无用的追问查询
            pending = []
            if state["followup_round"] < MAX_FOLLOWUP_ROUNDS:
                pending = await select_followup_questions(
                    session,
                    [candidate.id for candidate in candidates],
                    set(collected_ids),
                    ruled_out_symptom_ids=set(ruled_out_ids),
                    k=3,
                )

        next_question = None
        if pending:
            name = pending[0]["name"]
            next_question = {
                "id": pending[0]["id"],
                "name": name,
                "text": f"请问你最近有没有「{name}」的情况？",
            }

        return {
            "health_portrait": loaded_portrait,
            "collected_symptoms": collected,
            "ruled_out_symptoms": ruled_out,
            "candidate_conditions": [
                {
                    "id": c.id,
                    "name": c.name,
                    "urgency_hint": c.urgency_hint,
                    "hit_count": c.hit_count,
                }
                for c in candidates
            ],
            "pending_question": next_question,
        }

    graph = StateGraph(TriageState)  # 以 TriageState 为状态结构创建状态图
    graph.add_node("extract", extract_node)  # 入口节点：抽症状/红旗/主体边界/候选集/选下一问
    graph.add_node("ask", ask_node)  # 追问节点：interrupt 挂起等回答，恢复后轮次 +1
    graph.add_node("advise", advise_node)  # 建议节点：按候选集生成 final_advice

    graph.add_edge(START, "extract")  # 从 START 进入 extract（首次输入的处理入口）
    graph.add_conditional_edges(  # extract 之后按路由函数返回值做条件分流
        "extract",
        route_after_extract,  # 路由：红旗/他人/硬上限/无可追问 -> advise，否则 -> ask
        {"ask": "ask", "advise": "advise"},  # 返回值到目标节点的映射
    )
    graph.add_edge("ask", "extract")  # 用户回答追问后回到 extract，形成追问循环
    graph.add_edge("advise", END)  # 生成建议后结束本次会话

    return graph.compile(checkpointer=checkpointer)
