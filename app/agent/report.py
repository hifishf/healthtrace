"""报告解读子图：MCP 数据读取 -> 确定性分析 -> 知识检索 -> 带溯源建议。"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Awaitable, Callable
from datetime import date
from typing import Annotated, Any, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.identity import get_current_user_id
from app.trends import (
    TREND_INSUFFICIENT,
    analyze_metric_trend,
    metric_label,
    reference_status,
)

ToolCaller = Callable[[str, dict[str, Any]], Awaitable[Any]]


class ReportState(TypedDict):
    """报告解读状态：原始数据 -> 分析发现 -> 检索证据 -> 最终建议。"""

    messages: Annotated[list, add_messages]
    metrics: list[dict]
    medications: list[dict]
    reference_ranges: list[dict]
    portrait: dict
    interactions: list[dict]
    findings: list[dict]
    sources: list[dict]
    finding_sources: list[list[str]]
    final_report: dict | None


def initial_report_state(text: str) -> ReportState:
    """构造单次报告解读的初始状态。"""
    return {"messages": [HumanMessage(content=text)]}


def _age_group(portrait: dict) -> str | None:
    """按出生年份推导演示参考范围年龄组；画像无出生年份时不限定。"""
    birth_year = portrait.get("birth_year")
    if not birth_year:
        return None
    age = date.today().year - int(birth_year)
    return "adult" if age >= 18 else "child"


def _select_reference(
    metric_code: str,
    unit: str,
    reference_ranges: list[dict],
    portrait: dict,
) -> dict | None:
    """选择同指标同单位下最匹配年龄/性别的参考范围。"""
    candidates = [
        ref
        for ref in reference_ranges
        if ref.get("metric_code") == metric_code and ref.get("unit") == unit
    ]
    if not candidates:
        return None
    age_group = _age_group(portrait)
    sex = portrait.get("sex")

    def match_score(reference: dict) -> tuple[int, int, int]:
        """优先精确年龄组和性别，空限制次之，最后按 id 稳定排序。"""
        age_match = 2 if reference.get("age_group") == age_group else 1
        sex_match = 2 if reference.get("sex") == sex else 1
        return age_match, sex_match, -int(reference.get("id", 0))

    return max(candidates, key=match_score)


def build_metric_findings(
    metrics: list[dict],
    reference_ranges: list[dict],
    portrait: dict,
) -> list[dict]:
    """把同指标记录转为趋势和参考范围发现，未知指标不进入报告。"""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for metric in metrics:
        grouped[str(metric.get("metric_code") or "")].append(metric)

    findings: list[dict] = []
    for metric_code, points in grouped.items():
        label = metric_label(metric_code)
        if label == metric_code:
            continue
        latest = max(points, key=lambda point: point["recorded_date"])
        unit = latest.get("unit") or ""
        same_unit_points = [point for point in points if point.get("unit") == unit]
        reference = _select_reference(metric_code, unit, reference_ranges, portrait)
        trend_result = analyze_metric_trend(
            same_unit_points,
            reference=reference,
            label=label,
        )
        status = reference_status(float(latest["value"]), reference)
        status_text = f"最近一次结果{status}" if status else "暂无匹配参考范围"
        findings.append(
            {
                "kind": "metric_trend",
                "metric_code": metric_code,
                "label": label,
                "trend": trend_result["trend"],
                "status": status,
                "latest_value": float(latest["value"]),
                "unit": unit,
                "reference": reference,
                "text": f"{trend_result['text']}{status_text}。",
            }
        )
    return findings


def build_interaction_findings(interactions: list[dict]) -> list[dict]:
    """把确定性药物交互命中转为统一发现结构。"""
    return [
        {
            "kind": "drug_interaction",
            "drug_a": hit["drug_a"],
            "drug_b": hit["drug_b"],
            "severity": hit["severity"],
            "high_risk": bool(hit.get("high_risk")),
            "text": (
                f"发现{hit['drug_a']}与{hit['drug_b']}存在{hit['severity']}级交互。"
                f"{hit['description']}"
            ),
        }
        for hit in interactions
    ]


def _queries_for_findings(findings: list[dict]) -> list[dict]:
    """为每条发现构造检索任务；指标发现同时传入指标代码过滤。"""
    tasks: list[dict] = []
    generic_seen = False
    for finding in findings:
        if finding["kind"] == "metric_trend":
            tasks.append(
                {
                    "query": (
                        f"{finding['label']} {finding['trend']} "
                        f"{finding['status'] or '参考范围'} 复查建议"
                    ),
                    "metric_codes": [finding["metric_code"]],
                }
            )
        else:
            generic_seen = True
            tasks.append(
                {
                    "query": (
                        f"{finding['drug_a']}与{finding['drug_b']} "
                        f"{finding['severity']}级药物交互如何处理"
                    ),
                    "metric_codes": None,
                }
            )
    if not generic_seen:
        tasks.append(
            {
                "query": "体检报告解读 参考范围 趋势复查 医疗建议来源",
                "metric_codes": None,
            }
        )
    return tasks


def _merge_sources(source_lists: list[list[dict]]) -> list[dict]:
    """按 knowledge_key 去重来源，保留最高相似度并稳定排序。"""
    by_key: dict[str, dict] = {}
    for sources in source_lists:
        for source in sources:
            key = source["knowledge_key"]
            if key not in by_key or source["score"] > by_key[key]["score"]:
                by_key[key] = source
    return sorted(by_key.values(), key=lambda source: (-source["score"], source["knowledge_key"]))


def _assemble_advice(
    findings: list[dict],
    finding_sources: list[list[str]],
    sources: list[dict],
) -> list[dict]:
    """根据确定性发现生成建议，并绑定本次真实检索到的来源 key。"""
    advice: list[dict] = []
    fallback_key = sources[0]["knowledge_key"] if sources else None
    for index, finding in enumerate(findings):
        source_keys = finding_sources[index] if index < len(finding_sources) else []
        if not source_keys and fallback_key:
            source_keys = [fallback_key]
        if finding["kind"] == "drug_interaction":
            text = (
                f"{finding['text']}建议让处方医生或药师核对全部在用药物，"
                "不要自行停药或调整剂量。"
            )
        else:
            text = (
                f"{finding['text']}建议携带连续报告咨询专业人员；"
                "是否需要复查或治疗由医生结合症状和病史判断。"
            )
        advice.append({"text": text, "source_keys": source_keys})
    return advice


def build_report_graph(tool_caller: ToolCaller):
    """构建无 LLM 的报告解读子图，所有外部数据访问通过 MCP 工具完成。"""

    async def collect_node(state: ReportState) -> dict:
        """步骤 1：读取指标、参考范围、画像和用药，并执行确定性交互检查。"""
        user_id = str(get_current_user_id())
        metrics_result = await tool_caller("query_metrics", {"user_id": user_id})
        reference_result = await tool_caller("query_lab_reference_ranges", {})
        portrait = await tool_caller("get_health_portrait", {"user_id": user_id})
        medications_result = await tool_caller("list_medications", {"user_id": user_id})
        medications = medications_result.get("medications", [])
        names = [
            medication["name"]
            for medication in medications
            if medication.get("active", True) and medication.get("name")
        ]
        interaction_result = (
            await tool_caller(
                "check_drug_interaction",
                {"user_id": user_id, "medication_names": names},
            )
            if names
            else {"interactions": []}
        )
        return {
            "metrics": metrics_result.get("metrics", []),
            "reference_ranges": reference_result.get("reference_ranges", []),
            "portrait": portrait,
            "medications": medications,
            "interactions": interaction_result.get("interactions", []),
        }

    async def analyze_node(state: ReportState) -> dict:
        """步骤 2：计算指标趋势、参考范围状态和药物交互发现。"""
        findings = build_metric_findings(
            state["metrics"],
            state["reference_ranges"],
            state["portrait"],
        )
        findings.extend(build_interaction_findings(state["interactions"]))
        return {"findings": findings}

    async def retrieve_node(state: ReportState) -> dict:
        """步骤 3：逐发现检索知识，记录每条发现的来源映射。"""
        source_lists: list[list[dict]] = []
        finding_sources: list[list[str]] = []
        for task in _queries_for_findings(state["findings"]):
            result = await tool_caller(
                "search_knowledge",
                {
                    "query": task["query"],
                    "top_k": 3,
                    "metric_codes": task["metric_codes"],
                },
            )
            sources = result.get("results", [])
            source_lists.append(sources)
            finding_sources.append([source["knowledge_key"] for source in sources])
        return {
            "sources": _merge_sources(source_lists),
            "finding_sources": finding_sources,
        }

    async def advise_node(state: ReportState) -> dict:
        """步骤 4：组装建议、摘要和可回溯来源清单。"""
        findings = state["findings"]
        sources = state["sources"]
        metric_count = sum(finding["kind"] == "metric_trend" for finding in findings)
        interaction_count = len(findings) - metric_count
        if findings:
            summary = (
                f"本次解读分析 {metric_count} 个检验指标，"
                f"发现 {interaction_count} 条药物交互提示。"
            )
        else:
            summary = "暂无可解读的检验指标或药物交互规则命中。"
        final_report = {
            "summary": summary,
            "findings": findings,
            "advice": _assemble_advice(
                findings,
                state["finding_sources"],
                sources,
            ),
            "sources": sources,
        }
        return {"final_report": final_report}

    graph = StateGraph(ReportState)
    graph.add_node("collect", collect_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("advise", advise_node)
    graph.add_edge(START, "collect")
    graph.add_edge("collect", "analyze")
    graph.add_edge("analyze", "retrieve")
    graph.add_edge("retrieve", "advise")
    graph.add_edge("advise", END)
    return graph.compile()
