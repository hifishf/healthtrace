"""报告评估运行器：对每条 golden 输入执行真实 report 子图。"""
from app.agent.report import build_report_graph, initial_report_state
from app.mcp.client import call_tool


async def run_report_batch(texts: list[str]) -> list[dict]:
    """批量执行真实报告子图，返回每条输入的 final_report。"""
    graph = build_report_graph(tool_caller=call_tool)
    reports: list[dict] = []
    for text in texts:
        result = await graph.ainvoke(initial_report_state(text))
        reports.append(result.get("final_report") or {})
    return reports
