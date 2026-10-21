"""API 请求/响应模型与 Agent 结构化输出。"""
from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field


class MetricCreate(BaseModel):
    """POST /metrics 请求体：一条指标记录（同日期同指标覆盖）。"""

    metric_code: str = Field(min_length=1, max_length=50)
    value: Decimal
    unit: str = Field(min_length=1, max_length=20)
    recorded_date: date


class MetricResponse(BaseModel):
    """POST /metrics 响应：{ok, id}。"""

    ok: bool
    id: int


class RouteDecision(BaseModel):
    """Router 结构化输出：一句话的意图分类结果。"""

    intent: Literal["triage", "insight", "report", "chat"] = Field(
        description=(
            "意图分类：triage=追问分诊 / insight=因果分析 / "
            "report=报告解读 / chat=闲聊或记录指标"
        )
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description="0~1 之间的把握程度，越大越确定；LLM 自评、未经校准，不是概率",
    )
    extracted_slots: dict = Field(
        default_factory=dict,
        description='从用户话里抽出的槽位，如 {"symptom": "疲劳"}；本周占位，第 3 周启用',
    )


class SymptomExtraction(BaseModel):
    """症状抽取结构化输出：症状列表 + 主体边界 + 追问回答判定。"""

    symptoms: list[str] = Field(
        default_factory=list,
        description="抽取的症状规范短名称，如 乏力/胸痛/头晕；没有症状则为空列表",
    )
    subject: Literal["self", "other"] = Field(
        default="self",
        description="症状属于用户本人 self 还是他人 other（他人如 我爸爸/我妈/孩子/朋友）",
    )
    answer_verdict: Literal[
        "affirm", "deny", "contradict", "unrelated", "unclear", "not_applicable"
    ] = Field(
        default="not_applicable",
        description=(
            "仅当上一轮是追问时判断用户回答：affirm 肯定 / deny 否定 / "
            "contradict 方向相反（如问体重下降却答体重上升）/ unrelated 答非所问 / "
            "unclear 无法判断；没有上一轮追问时填 not_applicable"
        ),
    )


class ChatRequest(BaseModel):
    """POST /chat 请求体：用户一句话。"""

    text: str = Field(min_length=1)


class ChatResponse(BaseModel):
    """POST /chat 响应：路由结果 + 最终回复（第 4 步非流式）。"""

    intent: str
    confidence: float
    reply: str


class TriageRequest(BaseModel):
    """POST /triage 请求体：症状描述 + 可选 thread_id（多轮续跑用）。"""

    text: str = Field(min_length=1)
    thread_id: str | None = None


class TriageResponse(BaseModel):
    """POST /triage 响应：本轮返回追问问题，或返回最终分级建议。"""

    thread_id: str
    kind: Literal["question", "advice"]
    question: str | None = None
    final_advice: dict | None = None
    red_flag_triggered: bool = False
    candidate_count: int = 0
    top_tier_count: int = 0
    candidate_conditions: list[dict] = Field(default_factory=list)
