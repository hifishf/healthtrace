"""统一评估入口：确定性评估与语义评估走同一个命令行。

用法（在仓库根目录）：
    .venv/bin/python -m eval.harness deterministic   # 确定性检查（等价全量 pytest）
    .venv/bin/python -m eval.harness semantic        # 语义评估（golden 集 + judge，后续步接入）

设计意图：确定性检查（pytest）与语义检查（judge 打分）本质都是「评估」，
只是判定方式不同；收敛到同一个入口后，后续 insight / report 子图复用同一套 harness，
不用各自另起炉灶。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

from deepeval.test_case import LLMTestCase

from eval.metrics import build_advice_quality_metric
from eval.runner import run_triage_batch

# 仓库根（eval/ 的上一级）：从任意工作目录运行都能定位 tests 与 golden
REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = REPO_ROOT / "eval" / "golden"


def iter_golden(directory: Path = GOLDEN_DIR) -> Iterator[dict]:
    """按行读取 golden 目录下所有 jsonl，逐条 yield（dict）。

    golden 契约（冻结）：每行一个 JSON 对象，至少含 id / input / expected 三个字段。
    空行忽略；文件按名称排序，保证遍历顺序稳定。
    """
    for path in sorted(directory.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def run_deterministic(args: argparse.Namespace) -> int:
    """确定性评估：复用 pytest 跑指定目标，并把退出码透传给调用方。"""
    target = args.target or "tests"
    cmd = [sys.executable, "-m", "pytest", "-q", str(target)]
    print(f"▶ 确定性评估：{' '.join(cmd)}")
    # 用 sys.executable（即当前 venv 的 python），保证 pytest 与依赖来自同一环境
    return subprocess.call(cmd, cwd=REPO_ROOT)


def _sample(cases: list[dict], n: int | None) -> list[dict]:
    """抽样：n 为空或超过总量则全量；否则按固定种子抽 n 条（结果可复现）。"""
    if n is None or n >= len(cases):
        return cases
    return random.Random(0).sample(cases, n)


def run_semantic(args: argparse.Namespace) -> int:
    """语义评估：读 golden -> 跑真实分诊 -> judge 打分 -> 汇总。"""
    cases = list(iter_golden())
    if not cases:
        print(f"golden 目录为空：{GOLDEN_DIR}，请先放用例。")
        return 0

    sampled = _sample(cases, args.sample)
    advices = asyncio.run(run_triage_batch([case["input"] for case in sampled]))

    metric = build_advice_quality_metric()
    scores: list[float] = []
    level_hit = 0
    for case, advice in zip(sampled, advices):
        level = advice.get("level") or "unknown"
        suggestion = advice.get("suggestion") or ""
        conditions = advice.get("conditions") or []
        expected_level = case.get("expected_level")
        if expected_level == level:
            level_hit += 1

        test_case = LLMTestCase(
            input=case["input"],
            actual_output=suggestion,
            context=[
                f"建议级别：{level}（实际分级）",
                f"候选方向：{'、'.join(conditions) if conditions else '无'}",
            ],
            retrieval_context=conditions or ["无候选方向"],
        )
        metric.measure(test_case)
        scores.append(metric.score)
        print(f"{case['id']}  级别={level}  分数={metric.score:.2f}")

    mean = sum(scores) / len(scores)
    print(f"抽样 {len(sampled)} 条；平均分 {mean:.3f}；级别命中 {level_hit}/{len(sampled)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """解析子命令并分发到对应处理函数。"""
    parser = argparse.ArgumentParser(
        prog="eval.harness",
        description="HealthTrace 统一评估入口（确定性 + 语义）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    det = sub.add_parser("deterministic", help="跑确定性检查（等价全量 pytest）")
    det.add_argument("--target", default="tests", help="pytest 目标路径，默认 tests")
    det.set_defaults(func=run_deterministic)

    sem = sub.add_parser("semantic", help="跑语义评估（golden 集 + judge）")
    sem.add_argument("--sample", type=int, default=None, help="抽样条数，默认全量")
    sem.set_defaults(func=run_semantic)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
