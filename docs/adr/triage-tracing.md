# ADR：triage 子图用通用 span 层记录节点与 LLM 输入输出

日期：2026-10-22
状态：已接受

## 背景

Langfuse 之前只有 `/triage` 根 span，真实 LLM 的输入输出、结构化结果（如 `answer_verdict`）、
子图节点流向都没有记录。出现「体重下降/黑便」这类错误时，只能靠复现定位，无法直接归因。

## 决策

1. `observability.py` 提供通用能力：`llm_span`（generation、input/output/model）、
   `traced`（异步节点 span）、`set_attr` / `set_output`、Pydantic JSON 序列化与 4000 字截断。
2. triage 埋两个节点 span：`graph.triage.extract`、`graph.triage.advise`；三个 LLM generation：
   `extract_symptoms` / `rephrase_question` / `write_advice`。
3. 业务属性手工补：text/pending_question/symptoms/subject/answer_verdict/red_flag/
   collected/ruled_out/candidate_count/top_tier_count/candidates/next_question，以及
   level/conditions/suggestion。
4. 多轮对话用 `langfuse.session.id = thread_id` 关联。

## 备选方案

- 只保留根 span：被否，信息不足，无法归因。
- 全量自动采集（Langfuse CallbackHandler）：能覆盖 LLM 输入输出，但不认识业务语义
  （verdict、候选数、降级原因），仍需手工补属性；且多一个依赖与版本调试成本。
- 记录所有原始 prompt/response 不做截断：被否，存储膨胀、信噪比低、敏感面扩大。

## 后果

- 结构化输出值、子图节点流向、候选集变化都能在 Langfuse 直接查看。
- 通用 span 层与业务无关，后续父图接线时 triage 内部埋点原样复用，父图只需加一层薄 span。
- 需维护属性命名约定与截断策略；敏感字段脱敏与保留期是后续项。
