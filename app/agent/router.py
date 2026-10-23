"""意图路由器：LLM 结构化输出 4 意图，低置信度回落 chat。"""
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.llm import LLMUnavailable, invoke_llm
from app.config import settings
from app.observability import set_attr
from app.schemas import RouteDecision

# 明确“对话式记录”属于 chat，避免把“睡了 6 小时”误判为 triage
SYSTEM_PROMPT = """你是 HealthTrace 的意图路由器，把用户的一句话分到 4 个意图之一：
- chat：日常寒暄，或陈述/记录一条健康指标数据（睡眠、运动、饮食、体重等），意图是“记下来”。例如“昨晚睡了 6 小时”“今天走了 8000 步”“中午吃了沙拉”。
- triage：描述身体不适或症状，需要进一步追问分诊。例如“最近总是累”“胸口疼”。
- insight：要求分析健康指标之间的关联、趋势或因果。例如“为什么我喝咖啡就睡不好”。
- report：上传或解读体检报告。例如“帮我看下这份体检报告”。
只输出结构化结果，不要额外解释。"""


def apply_confidence_fallback(decision: RouteDecision, threshold: float) -> RouteDecision:
    """低置信度回落 chat（纯函数，确定性）。

    LLM 的 confidence 未经校准、不是概率，低于阈值时强制 chat，
    让 chat 子图用工具兜底澄清意图：宁少路由，勿错路由。
    """
    if decision.confidence < threshold:
        return RouteDecision(
            intent="chat",
            confidence=decision.confidence,
            extracted_slots=decision.extracted_slots,
        )
    return decision


async def route(text: str, llm: BaseChatModel) -> RouteDecision:
    """对一句话做意图分类（结构化输出 + 阈值回落）。

    method=function_calling：DeepSeek 支持函数调用但不支持 OpenAI 的
    json_schema 严格模式，因此显式指定 function calling 而非默认值。
    """
    runnable = llm.with_structured_output(RouteDecision, method="function_calling")
    try:
        decision = await invoke_llm(
            runnable,
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=text)],
        )
    except LLMUnavailable as exc:
        # 路由 LLM 不可用：宁少路由，勿错路由，强制回落 chat 兜底
        set_attr("fallback_reason", exc.reason)
        return RouteDecision(intent="chat", confidence=0.0, extracted_slots={})
    return apply_confidence_fallback(decision, settings.router_confidence_threshold)
