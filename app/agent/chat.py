"""chat 子图：LLM + record_metric 工具，承载闲聊与对话式记录。"""
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.agent.llm import get_llm
from app.agent.tools import query_profile, record_metric

_CHAT_TOOLS = [record_metric, query_profile]


def build_chat_graph(model: Runnable | None = None):
    """构建 chat 子图：agent 节点（LLM 绑定工具）+ tools 节点（ToolNode）。

    model 为可选注入：生产走 get_llm().bind_tools(...)，测试注入 fake 模型，
    从而在不开真 LLM 的情况下验证「工具被调用 + 循环闭环」。
    """
    if model is None:
        model = get_llm().bind_tools(_CHAT_TOOLS)

    graph = StateGraph(MessagesState)

    async def agent(state: MessagesState) -> dict:
        # 数据准备：把历史消息交给绑定工具的模型，得到下一步 AIMessage（可能带 tool_calls）
        message = await model.ainvoke(state["messages"])
        return {"messages": [message]}

    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(_CHAT_TOOLS))

    graph.add_edge(START, "agent")
    # 条件边：有 tool_calls 则走 tools 继续，否则结束
    graph.add_conditional_edges(
        "agent", tools_condition, {"tools": "tools", "__end__": END}
    )
    graph.add_edge("tools", "agent")

    return graph.compile()
