# ADR：统一评估 harness + CI 红线阻断

日期：2026-10-22
状态：已接受

## 背景

第 5 周要建评估体系：确定性断言（pytest）+ 语义评估（judge 打分）。若两者各写各的
散命令，后续 insight / report 子图评估会再复制一套；且「红旗规则不能被无意改坏」
需要一个自动防线，不能靠人每次手动跑测试。

## 决策

1. 建统一评估入口 `eval/harness.py`：`deterministic` 子命令复用 pytest，`semantic`
   子命令跑 golden 集 + judge，两类评估共用一个入口。
2. 语义评估用 DeepEval G-Eval（judge 固定 `temperature=0`），对最终建议质量打 0~1 分；
   golden 集 30 例入 repo 版本化（`eval/golden/advice_quality.jsonl`）。
3. CI 分两个 job：deterministic（全量 pytest，红线阻断合并）+ semantic（PR/main 抽样
   5 例、每周定时全量，非阻断）。
4. 红线 = 确定性红旗用例：人为改坏规则 → CI 红并阻断合并。

## 备选方案

- 不建 harness，散命令各跑各的：被否，后续子图评估难复用、CI 难统一。
- RAGAS + DeepEval 双框架：本周被否（RAGAS 0.4.3 与当前依赖冲突，且无检索可测），
  RAGAS 留到第 10 周 RAG 落地后接 faithfulness。
- 每次 PR 全量跑语义评估：被否，慢且费 token，改为抽样门禁。
- 用 LLM 当红旗判断而非确定性规则：被否，安全关键路径不可依赖概率系统。

## 后果

- 新子图评估只需往 `eval/golden/` 加数据、复用同一入口与 judge。
- 红旗改动有自动防线，面试可讲「故意改坏红旗 → CI 红 → 改回 → 绿」。
- 语义评估当前只覆盖一轮出结论的三类（emergency / accompany / no_symptom），
  多轮追问质量仍由确定性测试覆盖。
