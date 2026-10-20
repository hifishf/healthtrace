"""chat 子图：LLM + record_metric 工具，承载闲聊与对话式记录。"""
import datetime

from langchain_core.messages import SystemMessage
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.agent.llm import get_llm
from app.agent.tools import metric_today, query_profile, record_metric

_CHAT_TOOLS = [record_metric, query_profile]


def build_chat_system_prompt(today: datetime.date | None = None) -> str:
    """构造带业务日期口径的 chat 系统提示。"""
    current_date = today or metric_today()
    return f"""你是 HealthTrace 的健康管理助手。
当前日期是 {current_date.isoformat()}，业务时区为 Asia/Shanghai。

记录健康指标时必须遵守：
1. 用户使用相对日期时，把原话中的日期表达传给 record_metric 的 date 参数，
   例如“今天”“昨晚”“三天前”“昨天的昨天的昨天”。
2. 不要在模型侧把相对日期换算成 YYYY-MM-DD，工具会按当前日期确定换算。
3. 只有用户完全没有提到日期时才省略 date；无法判断日期时先询问用户。
"""


def build_chat_graph(model: Runnable | None = None):
    """构建 chat 子图：agent 节点（LLM 绑定工具）+ tools 节点（ToolNode）。

    model 为可选注入：生产走 get_llm().bind_tools(...)，测试注入 fake 模型，
    从而在不开真 LLM 的情况下验证「工具被调用 + 循环闭环」。
    """
    if model is None:
        model = get_llm().bind_tools(_CHAT_TOOLS)

    graph = StateGraph(MessagesState)

    async def agent(state: MessagesState) -> dict:
        # 数据准备：把当前状态消息交给模型，得到下一步 AIMessage
        # 模型可能直接回复，也可能返回 tool_calls。
        messages = [
            SystemMessage(content=build_chat_system_prompt()),
            *state["messages"],
        ]
        message = await model.ainvoke(messages)
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
