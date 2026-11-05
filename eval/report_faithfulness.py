"""RAGAS Faithfulness 专项：检查报告建议是否只基于实际引用来源。"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import warnings
from dataclasses import dataclass
from pathlib import Path

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig

with warnings.catch_warnings():
    # RAGAS 0.4 的 legacy evaluate 仍接收旧 Metric；工程已固定 ragas<0.5。
    warnings.filterwarnings("ignore", message="Importing Faithfulness.*", category=DeprecationWarning)
    from ragas.metrics import Faithfulness

from app.agent.llm import get_llm
from app.config import settings
from eval.report_runner import run_report_batch

CASE_MIN_SCORE = 0.80
MEAN_MIN_SCORE = 0.85
GOLDEN_PATH = (
    Path(__file__).resolve().parents[1]
    / "eval"
    / "golden"
    / "report_faithfulness.jsonl"
)


@dataclass(frozen=True)
class ThresholdResult:
    """阈值判定结果。"""

    passed: bool
    mean: float
    failures: tuple[str, ...]


def load_cases(path: Path = GOLDEN_PATH) -> list[dict]:
    """读取报告 faithfulness golden 集。"""
    cases: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                cases.append(json.loads(line))
    return cases


def _sample(cases: list[dict], n: int | None) -> list[dict]:
    """按固定种子抽样，保证评估可复现。"""
    if n is None or n >= len(cases):
        return cases
    return random.Random(0).sample(cases, n)


def validate_source_binding(case: dict, report: dict) -> list[str]:
    """检查建议引用完整、来源存在且覆盖该用例期望的检索方向。"""
    errors: list[str] = []
    sources = report.get("sources") or []
    source_by_key = {source.get("knowledge_key"): source for source in sources}
    advice = report.get("advice") or []
    if not advice:
        errors.append("没有报告建议")
    for index, item in enumerate(advice):
        for key in item.get("source_keys") or []:
            if key not in source_by_key:
                errors.append(f"建议 {index + 1} 引用了不存在的来源 {key}")
    prefix = case.get("expected_source_prefix")
    if prefix and not any(str(key).startswith(prefix) for key in source_by_key):
        errors.append(f"来源未覆盖期望前缀 {prefix}")
    return errors


def build_faithfulness_sample(case: dict, report: dict) -> SingleTurnSample:
    """把建议对应的报告事实与实际引用知识作为 RAGAS 上下文。"""
    findings = {
        finding.get("finding_id"): finding for finding in report.get("findings") or []
    }
    sources = {
        source.get("knowledge_key"): source for source in report.get("sources") or []
    }
    contexts: list[str] = []
    for item in report.get("advice") or []:
        finding = findings.get(item.get("finding_id"))
        if finding:
            context = "报告事实：" + json.dumps(
                finding,
                ensure_ascii=False,
                sort_keys=True,
            )
            if context not in contexts:
                contexts.append(context)
        for key in item.get("source_keys") or []:
            source = sources.get(key)
            if not source:
                continue
            context = f"{source.get('title', '')}\n{source.get('content', '')}".strip()
            if context and context not in contexts:
                contexts.append(context)
    return SingleTurnSample(
        user_input=case["input"],
        response="\n".join(item.get("text", "") for item in report.get("advice") or []),
        retrieved_contexts=contexts,
    )


def evaluate_thresholds(cases: list[dict], scores: list[float]) -> ThresholdResult:
    """检查单例和均值阈值，NaN/缺失分数按失败处理。"""
    if len(cases) != len(scores):
        raise ValueError("cases 与 scores 数量不一致")
    failures: list[str] = []
    normalized: list[float] = []
    for case, raw_score in zip(cases, scores):
        score = float(raw_score)
        if math.isnan(score):
            failures.append(f"{case['id']} 分数为 NaN")
            score = 0.0
        elif score < CASE_MIN_SCORE:
            failures.append(
                f"{case['id']} {score:.3f} < {CASE_MIN_SCORE:.2f}"
            )
        normalized.append(score)
    mean = sum(normalized) / len(normalized) if normalized else 0.0
    if mean < MEAN_MIN_SCORE:
        failures.append(f"均值 {mean:.3f} < {MEAN_MIN_SCORE:.2f}")
    return ThresholdResult(not failures, mean, tuple(failures))


def score_faithfulness(samples: list[SingleTurnSample]) -> list[float]:
    """调用 RAGAS Faithfulness 逐例打分。"""
    judge = LangchainLLMWrapper(get_llm(temperature=0.0))
    metric = Faithfulness(llm=judge)
    result = evaluate(
        EvaluationDataset(samples=samples),
        metrics=[metric],
        raise_exceptions=False,
        show_progress=False,
        batch_size=1,
        run_config=RunConfig(
            timeout=int(settings.llm_timeout_seconds),
            max_retries=settings.llm_max_retries,
            max_workers=1,
        ),
    )
    scores = [row.get("faithfulness", 0.0) for row in result.scores]
    return [float(score) if score is not None else float("nan") for score in scores]


def run_faithfulness(cases: list[dict], reports: list[dict]) -> int:
    """执行来源校验、RAGAS 打分和阈值门禁，返回命令退出码。"""
    all_errors: list[str] = []
    for case, report in zip(cases, reports):
        errors = validate_source_binding(case, report)
        all_errors.extend(f"{case['id']}：{error}" for error in errors)
    if all_errors:
        for error in all_errors:
            print(f"FAIL {error}")
        return 1

    samples = [
        build_faithfulness_sample(case, report)
        for case, report in zip(cases, reports)
    ]
    scores = score_faithfulness(samples)
    for case, score in zip(cases, scores):
        print(f"{case['id']} faithfulness={score:.3f}")
    threshold = evaluate_thresholds(cases, scores)
    print(f"mean faithfulness={threshold.mean:.3f}")
    if not threshold.passed:
        for failure in threshold.failures:
            print(f"FAIL {failure}")
        return 1
    print("PASS 报告建议全部达到 faithfulness 阈值")
    return 0


def main(argv: list[str] | None = None) -> int:
    """命令行入口：运行报告 golden 集 faithfulness。"""
    parser = argparse.ArgumentParser(description="HealthTrace 报告 faithfulness 评估")
    parser.add_argument("--sample", type=int, default=None, help="抽样条数，默认全量")
    args = parser.parse_args(argv)
    if args.sample is not None and args.sample < 1:
        parser.error("--sample 必须大于 0")
    if not settings.llm_api_key:
        print("FAIL 未配置 LLM_API_KEY")
        return 2

    cases = _sample(load_cases(), args.sample)
    reports = asyncio.run(run_report_batch([case["input"] for case in cases]))
    return run_faithfulness(cases, reports)


if __name__ == "__main__":
    raise SystemExit(main())
