# 评估体系（第 5 周）

## 一句话定位

一个总入口跑两类评估：**确定性检查**（结果固定、不看 AI）和**语义检查**（由 judge 打分）。

## 怎么跑

```bash
# 确定性检查（等价全量 pytest，不需要联网）
.venv/bin/python -m eval.harness deterministic

# 语义评估（golden 集 + judge，第 3/4 步接入）
.venv/bin/python -m eval.harness semantic
```

## 为什么要有统一入口

后续 insight / report 子图也要评估，但它们的评估逻辑和 triage 一样是
「确定性断言 + 语义打分」。收敛到这一个入口，新子图只要往 `eval/golden/`
加数据、复用同一套 judge 即可，不用各自复制一遍。

## golden 集契约（冻结）

位置：`eval/golden/*.jsonl`，入 repo 版本化。每行一个 JSON 对象，至少含：

- `id`：用例唯一编号
- `input`：输入的原始文本（症状描述等）
- `expected`：期望结果（级别 / 红旗性质 / 安全边界要点）

语义评估采用**抽样门禁**：PR 上抽样少量用例控制成本与耗时，全量在 main 分支或定时任务跑。
judge 固定 `temperature=0`，保证同一输入打分可复现。
