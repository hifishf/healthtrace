"""追问分诊子图：TriageState 状态机 + interrupt/Command 挂起-恢复 + 硬上限。"""
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
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
    top_tier_size,
)
from app.observability import llm_span, set_attr, set_output, traced

# 追问硬上限（状态机边条件，不依赖 LLM 自觉）
MAX_FOLLOWUP_ROUNDS = 3


class TriageState(TypedDict):
    """追问分诊状态（第 7.2 节冻结契约）。"""

    messages: Annotated[list, add_messages]
    health_portrait: dict
    collected_symptoms: dict  # str(symptom_id) -> 症状名
    ruled_out_symptoms: dict  # str(symptom_id) -> 症状名（已明确排除，不再追问）
    candidate_conditions: list  # [{"id", "name", "urgency_hint"}]
    candidate_count: int  # 候选疾病总数（并集基数）
    top_tier_count: int  # 高置信档大小（命中数并列最高的候选个数）
    followup_round: int
    red_flag_triggered: bool
    final_advice: dict | None
    pending_question: dict | None  # 下一轮追问（内部实现细节）


# 特殊级别的「安全核心」逐字固定；LLM 只写前面的共情/承接话术。
_EMERGENCY_CORE = "请立即就医或拨打急救电话，不要等待线上建议。"
_ACCOMPANY_CORE = "这是他人的症状，无法代替本人分诊，建议陪同就医。"
_NO_SYMPTOM_CORE = "请再描述一下具体症状（比如：最近总是累、胸口疼）。"

_SUGGESTIONS: dict[str, str] = {
    "observe": "根据你描述的症状，目前倾向先观察休息并注意变化，如症状加重请及时就医。",
    "adjust": "建议先调整生活方式（规律作息、均衡饮食、适量运动），观察症状是否缓解。",
    "online_consult": "建议进行线上问诊，由医生进一步评估和指导。",
    "seek_care": "建议尽快到医院就诊，由医生进一步检查明确原因。",
}

# LLM 只负责「怎么说」，不负责「怎么判断」：结论（级别/症状/候选）由确定性规则给出，
# LLM 仅把结论转写成自然语言；失败时回退到模板，安全边界不受影响。
_QUESTION_REPHRASE_SYSTEM = (
    "你是 HealthTrace 的健康管家。请把待确认的症状改写成一句自然、温和的中文追问。"
    "只输出这一句话，不要加引号，不要引入新的症状，不要给出诊断或医疗建议。"
    "不要声称用户提到过该症状（不要写「您提到有…」）；用询问的方式，例如「最近有没有…」。"
)
_ADVICE_WRITE_SYSTEM = (
    "你是 HealthTrace 的健康管家。根据给定的确定性分级和候选方向，写一句温和、清楚的中文建议。"
    "必须遵守：不输出诊断结论（不说「你得了某病」），不推荐处方药，紧急时建议就医。"
    "只能使用给定的候选方向、已采集症状和画像，不要编造用户未提供的信息。"
    "只输出建议这一句，不要加引号，不要改写给定级别。"
)
_SPECIAL_WRITE_SYSTEM = (
    "你是 HealthTrace 的健康管家。请根据级别和上下文，写一句简短、温和的中文共情/承接话术。"
    "只输出这一句话，不要加引号，不要给出诊断，不要推荐处方药；"
    "不要给出就医/拨打急救电话等行动建议（行动指令会由系统追加），也不要重复安全指令。"
)

# 追问回答的肯定/否定表面形式（确定性，不依赖 LLM；否定优先于肯定，
# 因为“没有”里包含“有”，先判否定才不会被误判成肯定）
_NEGATIVE_MARKERS: tuple[str, ...] = ("没有", "没", "无", "不是", "不会", "否")
_AFFIRMATIVE_MARKERS: tuple[str, ...] = ("有", "是", "对", "嗯", "会", "出现过", "偶尔会")
# 方向性症状的正反词对：回答与待问方向相反时，视为对该症状的否定（不再追问）
_OPPOSITE_PAIRS: tuple[tuple[str, str], ...] = (
    ("下降", "上升"),
    ("减少", "增加"),
    ("降低", "升高"),
)


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


def contradicts_pending(answer: str, pending_name: str) -> bool:
    """回答是否明确与待问症状方向相反（如待问体重下降、回答体重上升）。"""
    if not pending_name:
        return False
    for positive, opposite in _OPPOSITE_PAIRS:
        if positive in pending_name and opposite in answer:
            return True
        if opposite in pending_name and positive in answer:
            return True
    return False


def initial_triage_state(text: str) -> dict:
    """构造一次追问分诊的初始状态（单用户，无登录）。"""
    return {
        "messages": [HumanMessage(content=text)],
        "health_portrait": {},
        "collected_symptoms": {},
        "ruled_out_symptoms": {},
        "candidate_conditions": [],
        "candidate_count": 0,
        "top_tier_count": 0,
        "followup_round": 0,
        "red_flag_triggered": False,
        "final_advice": None,
        "pending_question": None,
    }


def _advice_payload(
    level: str,
    label: str,
    suggestion: str,
    conditions: list[str] | None = None,
) -> dict:
    """组装 final_advice 字典（级别/标签/条件/建议）。"""
    return {
        "level": level,
        "label": label,
        "conditions": conditions or [],
        "suggestion": suggestion,
    }


def _question_template(name: str) -> str:
    """追问的确定性模板（LLM 转写失败的降级兜底）。"""
    return f"请问你最近有没有「{name}」的情况？"


async def _rephrase_question(
    llm,
    name: str,
    collected_names: list[str],
    portrait_summary: str,
) -> str:
    """把选中的症状转写成自然语言追问；失败回退确定性模板。"""
    fallback = _question_template(name)
    try:
        context = (
            f"待确认症状：{name}\n"
            f"已采集症状：{'、'.join(collected_names) if collected_names else '无'}\n"
            f"健康画像：{portrait_summary or '暂无'}"
        )
        with llm_span(
            "llm.triage.rephrase_question",
            {
                "symptom": name,
                "collected": collected_names,
                "portrait": portrait_summary,
            },
        ):
            response = await llm.ainvoke(
                [SystemMessage(content=_QUESTION_REPHRASE_SYSTEM), HumanMessage(content=context)]
            )
            set_output((response.content or "").strip())
        text = (response.content or "").strip()
        return text or fallback
    except Exception:
        return fallback


async def _write_advice(
    llm,
    level: str,
    label: str,
    condition_names: list[str],
    collected_names: list[str],
    portrait_summary: str,
) -> str:
    """把确定性分级转写成自然语言建议；失败回退该级别的模板。"""
    fallback = _SUGGESTIONS.get(level, "建议咨询医生进一步评估。")
    try:
        context = (
            f"建议级别：{label}（{level}）\n"
            f"候选方向：{'、'.join(condition_names) if condition_names else '无'}\n"
            f"已采集症状：{'、'.join(collected_names) if collected_names else '无'}\n"
            f"健康画像：{portrait_summary or '暂无'}"
        )
        with llm_span(
            "llm.triage.write_advice",
            {
                "level": level,
                "conditions": condition_names,
                "collected": collected_names,
                "portrait": portrait_summary,
            },
        ):
            response = await llm.ainvoke(
                [SystemMessage(content=_ADVICE_WRITE_SYSTEM), HumanMessage(content=context)]
            )
            set_output((response.content or "").strip())
        text = (response.content or "").strip()
        if text:
            set_attr("triage.advice_source", "llm")
            return text
        set_attr("triage.advice_source", "template_empty")
        return fallback
    except Exception:
        set_attr("triage.advice_source", "template_error")
        return fallback


async def _write_special_advice(
    writer_llm,
    level: str,
    label: str,
    core: str,
    context: str,
) -> str:
    """特殊级别：LLM 写共情前缀 + 固定安全核心；失败只用核心。"""
    try:
        with llm_span(
            "llm.triage.write_special_advice",
            {"level": level, "context": context},
        ):
            response = await writer_llm.ainvoke(
                [
                    SystemMessage(content=_SPECIAL_WRITE_SYSTEM),
                    HumanMessage(
                        content=f"级别：{label}（{level}）\n上下文：{context}\n请写一句承接话术。"
                    ),
                ]
            )
            prefix = (response.content or "").strip()
            set_output(prefix)
        if prefix:
            set_attr("triage.advice_source", "llm")
            separator = "" if prefix.endswith(("。", "！", "？", "…")) else "。"
            return f"{prefix}{separator}{core}"
        set_attr("triage.advice_source", "template_empty")
        return core
    except Exception:
        set_attr("triage.advice_source", "template_error")
        return core


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


def build_triage_graph(llm=None, writer_llm=None, checkpointer=None):
    """构建追问分诊子图（extract -> ask(interrupt) 循环 -> advise）。

    llm 负责确定性抽取（temperature=0）；writer_llm 负责话术转写（较高 temperature）。
    两者都可注入 fake 做确定性测试；checkpointer 必须提供（interrupt 依赖断点持久化）。
    """
    injected_llm = llm is not None
    if llm is None:
        llm = get_llm()
    if writer_llm is None:
        writer_llm = (
            llm
            if injected_llm
            else get_llm(temperature=settings.llm_writer_temperature)
        )

    @traced("graph.triage.advise")
    async def advise_node(state: TriageState) -> dict:
        """生成最终分级建议：级别由确定性规则算，措辞由 LLM 转写（失败回退模板）。"""
        if state.get("final_advice") is not None:
            set_attr("triage.level", state["final_advice"].get("level"))
            return {}
        # 未识别到症状：不给健康建议，提示重新描述
        if not state.get("collected_symptoms") or not state.get("candidate_conditions"):
            set_attr("triage.level", "no_symptom")
            suggestion = await _write_special_advice(
                writer_llm,
                "no_symptom",
                "未识别到症状",
                _NO_SYMPTOM_CORE,
                f"用户输入：{state['messages'][-1].content}；"
                f"画像：{portrait.summarize(state.get('health_portrait') or {})}",
            )
            return {
                "final_advice": _advice_payload(
                    "no_symptom", "未识别到症状", suggestion
                )
            }
        level = classify_advice_level_by_hints(
            [candidate.get("urgency_hint") for candidate in state["candidate_conditions"]]
        )
        candidates = state["candidate_conditions"]
        suggestion = await _write_advice(
            writer_llm,
            level,
            ADVICE_LEVELS[level],
            [candidate["name"] for candidate in candidates],
            list(state["collected_symptoms"].values()),
            portrait.summarize(state.get("health_portrait") or {}),
        )
        set_attr("triage.level", level)
        set_attr("triage.conditions", [candidate["name"] for candidate in candidates])
        set_attr("triage.suggestion", suggestion)
        return {
            "final_advice": {
                "level": level,
                "label": ADVICE_LEVELS[level],
                "conditions": [candidate["name"] for candidate in candidates],
                "suggestion": suggestion,
            }
        }

    @traced("graph.triage.extract")
    async def extract_node(state: TriageState) -> dict:
        """入口：加载画像 -> 抽症状 -> 红旗/主体边界 -> 归一化 -> 候选集 -> 选下一问。"""
        text = state["messages"][-1].content
        pending_q = state.get("pending_question")
        set_attr("triage.text", text)
        set_attr("triage.pending_question", pending_q.get("text") if pending_q else None)

        async with async_session() as session:
            loaded_portrait = await portrait.get_health_portrait(
                session, settings.default_user_id
            )

        extraction = await extract_symptoms(
            text,
            llm,
            portrait_summary=portrait.summarize(loaded_portrait),
            pending_question=pending_q.get("text") if pending_q else None,
        )
        # 追问回答判定以 LLM 结构化结果为主（理解语义），状态更新仍确定性执行；
        # LLM 判不清时回退到确定性极性/反向词兜底。
        answered_verdict = extraction.answer_verdict if pending_q else "not_applicable"
        if pending_q and answered_verdict in ("unclear", "not_applicable"):
            if not extraction.symptoms:
                answered_verdict = answer_polarity(text)
            elif contradicts_pending(text, pending_q.get("name", "")):
                answered_verdict = "contradict"
        red_flag = check_red_flag(text, extraction.subject)
        set_attr("triage.symptoms", extraction.symptoms)
        set_attr("triage.subject", extraction.subject)
        set_attr("triage.answer_verdict", answered_verdict)
        set_attr("triage.red_flag", red_flag.triggered)

        portrait_text = portrait.summarize(loaded_portrait)
        if red_flag.triggered:
            suggestion = await _write_special_advice(
                writer_llm,
                "emergency",
                "立即就医",
                _EMERGENCY_CORE,
                f"用户描述：{text}；匹配红旗：{'、'.join(red_flag.matched) or '无'}；"
                f"已抽症状：{'、'.join(extraction.symptoms) or '无'}；画像：{portrait_text}",
            )
            return {
                "health_portrait": loaded_portrait,
                "red_flag_triggered": True,
                "final_advice": _advice_payload("emergency", "立即就医", suggestion),
            }
        if extraction.subject == "other":
            suggestion = await _write_special_advice(
                writer_llm,
                "accompany",
                "建议陪同就医",
                _ACCOMPANY_CORE,
                f"用户描述：{text}；已抽症状：{'、'.join(extraction.symptoms) or '无'}；"
                f"画像：{portrait_text}",
            )
            return {
                "health_portrait": loaded_portrait,
                "final_advice": _advice_payload(
                    "accompany", "建议陪同就医", suggestion
                ),
            }

        async with async_session() as session:
            symptom_ids = await match_symptoms(session, extraction.symptoms)
            # 肯定回答回填：答「有」时 LLM 抽不到症状，直接用上一轮 pending 症状补上
            if answered_verdict == "affirm" and pending_q and pending_q.get("id") is not None:
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
            if (
                answered_verdict in ("deny", "contradict")
                and pending_q
                and pending_q.get("id") is not None
            ):
                pending_id = int(pending_q["id"])
                ruled_out[str(pending_id)] = (
                    pending_q.get("name") or names.get(pending_id, "")
                )
            ruled_out_ids = [int(key) for key in ruled_out]
            candidates = await compute_candidates(session, collected_ids)
            set_attr("triage.collected", list(collected.values()))
            set_attr("triage.ruled_out", list(ruled_out.values()))
            set_attr("triage.candidate_count", len(candidates))
            set_attr("triage.top_tier_count", top_tier_size(candidates))
            set_attr("triage.candidates", [candidate.name for candidate in candidates])
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
            question_text = await _rephrase_question(
                writer_llm,
                name,
                list(collected.values()),
                portrait_text,
            )
            next_question = {
                "id": pending[0]["id"],
                "name": name,
                "text": question_text,
            }
        set_attr("triage.next_question", next_question["text"] if next_question else None)

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
            "candidate_count": len(candidates),
            "top_tier_count": top_tier_size(candidates),
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
