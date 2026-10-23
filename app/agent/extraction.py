"""症状抽取：LLM 结构化输出 + 确定性主体边界兜底。"""
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.llm import invoke_llm
from app.observability import llm_span, set_output
from app.schemas import SymptomExtraction


SYSTEM_PROMPT = """你是 HealthTrace 的症状抽取器。从用户这句话里抽取提到的身体症状，并判断这些症状是谁的。
规则：
- symptoms：抽取症状的规范短名称，尽量接近医学通用说法（如 乏力/胸痛/头晕/心悸），不要带程度和数量；没提到症状就返回空列表。
- subject：这些症状是用户本人（self）还是他人（other）。他人指“我爸/我妈/孩子/朋友/他/她”等第三方。
- answer_verdict：仅当收到「上一轮追问」时判断回答类型（affirm 肯定/deny 否定/contradict 方向相反/unrelated 答非所问/unclear 无法判断）；没有追问时填 not_applicable。
只输出结构化结果，不要额外解释。"""


# 他人指代词（确定性，先于 LLM 判定，保证“我爸爸胸痛”这类边界不依赖 LLM）
_OTHER_PRONOUNS: tuple[str, ...] = (
    "我爸爸", "我爸", "我父亲", "我妈妈", "我妈", "我母亲",
    "我儿子", "我女儿", "我孩子", "我小孩", "我老婆", "我老公",
    "我丈夫", "我妻子", "我朋友", "我同事", "我领导",
    "我奶奶", "我爷爷", "我姥姥", "我外公", "我外婆",
    "我家老人", "我邻居", "我室友",
    "爸爸", "妈妈", "父亲", "母亲", "儿子", "女儿", "孩子",
    "老婆", "老公", "丈夫", "妻子", "朋友", "同事", "领导",
    "奶奶", "爷爷", "姥姥", "外公", "外婆", "老人", "邻居", "室友",
)
_SELF_PRONOUNS: tuple[str, ...] = ("我", "自己", "本人")


def detect_subject(text: str) -> str | None:
    """确定性主体判定：先扫他人指代词，再扫本人指代词；无法判定返回 None。

    顺序很重要：“我爸爸”同时包含“我”，必须先判 other，否则会误判成 self。
    """
    for pronoun in _OTHER_PRONOUNS:
        if pronoun in text:
            return "other"
    for pronoun in _SELF_PRONOUNS:
        if pronoun in text:
            return "self"
    return None


def resolve_subject(text: str, llm_subject: str) -> str:
    """确定性判定优先；无法判定时回退到 LLM 的 subject。"""
    return detect_subject(text) or llm_subject


def build_system_prompt(
    portrait_summary: str | None = None,
    pending_question: str | None = None,
) -> str:
    """构造症状抽取的系统提示词；有画像摘要/上一轮追问时追加到上下文里。"""
    prompt = SYSTEM_PROMPT
    if portrait_summary:
        prompt += (
            f"\n\n用户健康画像：{portrait_summary}。"
            "抽取症状时可参考，但不要因为画像就臆造症状。"
        )
    if pending_question:
        prompt += (
            f"\n\n上一轮追问：{pending_question}\n"
            "请判断用户这句话是对该追问的回答："
            "affirm=肯定（有/是/会），deny=否定（没有/不会/不是），"
            "contradict=方向相反（如问体重下降却答体重上升），"
            "unrelated=答非所问，unclear=无法判断。"
            "把结果填入 answer_verdict；同时照常抽取 symptoms。"
        )
    return prompt


async def extract_symptoms(
    text: str,
    llm: BaseChatModel,
    portrait_summary: str | None = None,
    pending_question: str | None = None,
) -> SymptomExtraction:
    """对一句话抽取症状 + 主体（LLM 结构化输出，主体经确定性兜底修正）。

    portrait_summary 为画像一行摘要；pending_question 为上一轮追问，
    用于让 LLM 判定这句话是肯定/否定/相反/无关/不清楚。
    """
    with llm_span(
        "llm.triage.extract_symptoms",
        {
            "text": text,
            "portrait": portrait_summary,
            "pending_question": pending_question,
        },
    ):
        runnable = llm.with_structured_output(SymptomExtraction, method="function_calling")
        result = await invoke_llm(
            runnable,
            [
                SystemMessage(
                    content=build_system_prompt(portrait_summary, pending_question)
                ),
                HumanMessage(content=text),
            ],
        )
        set_output(result)
    # 主体边界：确定性指代词优先，LLM 只是补充，避免“我爸爸胸痛”被误判成 self
    return result.model_copy(update={"subject": resolve_subject(text, result.subject)})
