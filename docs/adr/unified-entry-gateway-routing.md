# ADR：统一入口用入口层网关路由，而非父图真嵌套 triage 子图

日期：2026-10-26
状态：已接受

## 背景

第 6 周要求「父图接回四个子图（统一入口）」，验收是「同一入口一句话路由到 chat / triage，
insight / report 回 stub」。triage 子图带 `interrupt()` 与 Postgres checkpointer（多轮追问挂起/续跑）。

## 决策

统一入口 `POST /agent`（`app/agent/entry.py` 的 `dispatch`）采用**入口层网关路由**：

1. `route()` 拿意图；
2. `chat` → 直接 `chat_graph.ainvoke`；
3. `triage` → 直接 `triage_graph.ainvoke`（首次用 `initial_triage_state`，带 `thread_id` 用 `Command(resume=...)`）；
4. `insight` / `report` → `stub_reply` 占位。

父图 `build_agent_graph` 保持 chat + stub，仅供非流式 `/chat`（遗留）使用；`/chat/stream`、`/triage` 端点保留。

## 备选方案

- 父图真嵌套 triage 子图：**被否**。实测子图 `interrupt` 虽能冒泡到顶层，但顶层 `Command(resume=...)`
  要求父图也带 checkpointer，且父图状态需与 `TriageState` 的 11 个字段强耦合；复杂度显著高于收益，违背 KISS。
- 把 `/chat` 直接升级为统一入口：**被否**。`/chat` 响应模型只有 intent/confidence/reply，
  与 triage 的 question/advice 结构不兼容；且前端追问走独立 `/triage` 面板，无需改动。

## 后果

- 多一处 `dispatch` 编排（入口层），但各子图保持解耦，各自 checkpointer 职责清晰。
- 后续 `insight` / `report` 子图接入时，只需在 `dispatch` 里加两个分支，不碰 chat / triage。
- 遗留 `/chat` 仍把 triage 回 stub（文档已标注），实际统一入口是 `/agent`。
