"""报告解读子图确定性断言：趋势发现、药物告警和建议来源绑定。"""
from collections.abc import Awaitable, Callable
from typing import Any

from app.agent.report import (
    build_metric_findings,
    build_report_graph,
    initial_report_state,
)

ToolCaller = Callable[[str, dict[str, Any]], Awaitable[Any]]


def _source(key: str, score: float) -> dict:
    """构造一条与 MCP search_knowledge 返回结构一致的知识来源。"""
    return {
        "knowledge_key": key,
        "title": key,
        "content": f"{key} 内容",
        "excerpt": f"{key} 内容",
        "category": "safety",
        "metric_codes": [],
        "source_name": "测试知识库",
        "source_url": "https://example.com/knowledge",
        "source_version": "test",
        "score": score,
    }


def test_metric_findings_compute_trend_status_and_ignore_unknown_metrics() -> None:
    """LDL 三年上升且末次偏高；未知睡眠指标不进入报告发现。"""
    findings = build_metric_findings(
        [
            {
                "metric_code": "ldl_cholesterol",
                "value": 3.1,
                "unit": "mmol/L",
                "recorded_date": "2024-01-15",
            },
            {
                "metric_code": "ldl_cholesterol",
                "value": 3.8,
                "unit": "mmol/L",
                "recorded_date": "2025-01-15",
            },
            {
                "metric_code": "sleep_duration",
                "value": 6,
                "unit": "h",
                "recorded_date": "2025-01-15",
            },
        ],
        [
            {
                "id": 1,
                "metric_code": "ldl_cholesterol",
                "unit": "mmol/L",
                "low": None,
                "high": 3.4,
                "age_group": None,
                "sex": None,
            }
        ],
        {},
    )

    assert len(findings) == 1
    assert findings[0]["trend"] == "持续上升"
    assert findings[0]["status"] == "偏高"
    assert "持续上升" in findings[0]["text"]


async def test_report_graph_attaches_real_retrieval_sources_to_every_advice() -> None:
    """完整子图执行后，每条建议引用的 key 均存在于本次检索来源中。"""
    calls: list[tuple[str, dict]] = []

    async def fake_tool(name: str, arguments: dict) -> dict:
        calls.append((name, arguments))
        if name == "query_metrics":
            return {
                "metrics": [
                    {
                        "metric_code": "ldl_cholesterol",
                        "value": 3.1,
                        "unit": "mmol/L",
                        "recorded_date": "2024-01-15",
                    },
                    {
                        "metric_code": "ldl_cholesterol",
                        "value": 3.4,
                        "unit": "mmol/L",
                        "recorded_date": "2025-01-15",
                    },
                    {
                        "metric_code": "ldl_cholesterol",
                        "value": 3.8,
                        "unit": "mmol/L",
                        "recorded_date": "2026-01-15",
                    },
                ]
            }
        if name == "query_lab_reference_ranges":
            return {
                "reference_ranges": [
                    {
                        "id": 1,
                        "metric_code": "ldl_cholesterol",
                        "unit": "mmol/L",
                        "low": None,
                        "high": 3.4,
                        "age_group": None,
                        "sex": None,
                    }
                ]
            }
        if name == "get_health_portrait":
            return {}
        if name == "list_medications":
            return {
                "medications": [
                    {"name": "华法林", "active": True},
                    {"name": "阿司匹林", "active": True},
                ]
            }
        if name == "check_drug_interaction":
            return {
                "interactions": [
                    {
                        "drug_a": "华法林",
                        "drug_b": "阿司匹林",
                        "severity": "严重",
                        "description": "可能增加出血风险。",
                        "high_risk": True,
                    }
                ]
            }
        if name == "search_knowledge":
            if arguments.get("metric_codes") == ["ldl_cholesterol"]:
                return {"results": [_source("lab:ldl:followup", 0.9)]}
            return {"results": [_source("safety:medication", 0.8)]}
        raise AssertionError(f"未模拟工具：{name}")

    graph = build_report_graph(tool_caller=fake_tool)
    result = await graph.ainvoke(initial_report_state("帮我看下体检报告"))
    report = result["final_report"]

    assert [finding["kind"] for finding in report["findings"]] == [
        "metric_trend",
        "drug_interaction",
    ]
    assert report["findings"][0]["trend"] == "持续上升"
    assert report["findings"][1]["severity"] == "严重"
    source_keys = {source["knowledge_key"] for source in report["sources"]}
    assert all(item["source_keys"] for item in report["advice"])
    assert all(set(item["source_keys"]) <= source_keys for item in report["advice"])
    assert [call[0] for call in calls[:5]] == [
        "query_metrics",
        "query_lab_reference_ranges",
        "get_health_portrait",
        "list_medications",
        "check_drug_interaction",
    ]


async def test_report_without_retrieval_results_does_not_fabricate_sources() -> None:
    """知识检索为空时，建议保留但 source_keys 为空。"""

    async def fake_tool(name: str, arguments: dict) -> dict:
        if name == "query_metrics":
            return {
                "metrics": [
                    {
                        "metric_code": "ldl_cholesterol",
                        "value": 3.8,
                        "unit": "mmol/L",
                        "recorded_date": "2026-01-15",
                    }
                ]
            }
        if name == "query_lab_reference_ranges":
            return {"reference_ranges": []}
        if name == "get_health_portrait":
            return {}
        if name == "list_medications":
            return {"medications": []}
        if name == "search_knowledge":
            return {"results": []}
        raise AssertionError(f"未模拟工具：{name}")

    graph = build_report_graph(tool_caller=fake_tool)
    result = await graph.ainvoke(initial_report_state("帮我看下体检报告"))
    report = result["final_report"]

    assert report["sources"] == []
    assert report["advice"][0]["source_keys"] == []
