# ADR：SSE 流式路径绕开父图、直接流 chat 子图

日期：2026-10-20
状态：已接受

## 背景

第 2 周要求聊天回复以 SSE 逐字流式返回（「昨晚睡了 6 小时」落库、画像更新、前端打字机效果）。
系统采用「路由 + 子图」架构，chat 是父图里嵌套的 LangGraph 子图。
尝试让父图直接 `astream(stream_mode="messages")` 时发现：嵌套子图的 LLM token 不穿透父图边界，
父图只能拿到子图跑完后的整条 AIMessage，拿不到逐 token 的 AIMessageChunk。

## 决策

`/chat/stream` 不流父图，改为三步：

1. 直接调 `route()` 函数（非流式）拿意图与置信度；
2. 若 `intent == "chat"`，直接 `chat_graph.astream(stream_mode="messages")` 拿逐 token；
3. 非 chat 意图用 stub 占位回复。

## 备选方案

- 父图直接流式：被否。子图 token 不穿透，只能整段输出，无打字机效果。
- chat 内联进父图（放弃子图）：可逐字流式，但违背「路由 + 子图」决策，模块化受损。
- 降级轮询（文档允许的兜底）：可跑但丢失流式体验，SSE spike 已成功，故未采用。

## 后果

- 流式（`route()` + `chat_graph`）与非流式（`agent_graph`）两条路径共享同一 `route()` 与 `chat_graph` 实例，
  逻辑不重复、编排有轻微重复，需保持同步。
- 第 3 周引入 checkpointer / interrupt 后，若需要单张有状态图流式，需重新评估
  （升级 LangGraph / 内联 / 接受整段输出）。
