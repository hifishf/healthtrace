"""报告 faithfulness 评估契约：golden 完整性、来源映射与阈值门禁。"""
from eval.report_faithfulness import (
    build_faithfulness_sample,
    evaluate_thresholds,
    load_cases,
    validate_source_binding,
)


def _report() -> dict:
    """构造一份带两条来源、但只引用其中一条的报告结果。"""
    return {
        "findings": [
            {
                "finding_id": "metric:ldl",
                "kind": "metric_trend",
                "text": "LDL 偏高。",
            }
        ],
        "advice": [
            {
                "finding_id": "metric:ldl",
                "text": "LDL 偏高，建议携带报告咨询医生。",
                "source_keys": ["lab:b"],
            }
        ],
        "sources": [
            {
                "knowledge_key": "lab:a",
                "title": "知识 A",
                "content": "正文 A",
            },
            {
                "knowledge_key": "lab:b",
                "title": "知识 B",
                "content": "正文 B",
            },
        ],
    }


def test_report_golden_has_ten_unique_cases() -> None:
    """报告专项至少 10 条，ID 唯一且每条冻结期望来源前缀。"""
    cases = load_cases()

    assert len(cases) >= 10
    assert len({case["id"] for case in cases}) == len(cases)
    assert all(case["input"] for case in cases)
    assert all(case["expected_source_prefix"].startswith("lab:") for case in cases)


def test_faithfulness_contexts_only_include_cited_sources() -> None:
    """未被建议引用的检索结果不能进入 RAGAS 上下文。"""
    case = {"id": "case-1", "input": "帮我看报告", "expected_source_prefix": "lab:"}
    sample = build_faithfulness_sample(case, _report())

    assert sample.user_input == "帮我看报告"
    assert sample.response == "LDL 偏高，建议携带报告咨询医生。"
    assert len(sample.retrieved_contexts) == 2
    assert sample.retrieved_contexts[0].startswith("报告事实：")
    assert "\"metric:ldl\"" in sample.retrieved_contexts[0]
    assert sample.retrieved_contexts[1] == "知识 B\n正文 B"
    assert validate_source_binding(case, _report()) == []


def test_source_binding_rejects_missing_reference_and_expected_direction() -> None:
    """建议引用不存在来源或缺少期望来源时，门禁给出明确错误。"""
    report = _report()
    report["advice"][0]["source_keys"].append("lab:missing")
    case = {
        "id": "case-2",
        "input": "帮我看报告",
        "expected_source_prefix": "drug:",
    }

    errors = validate_source_binding(case, report)

    assert any("不存在" in error for error in errors)
    assert any("期望前缀" in error for error in errors)


def test_thresholds_reject_case_or_mean_below_frozen_values() -> None:
    """单例低于 0.80 或均值低于 0.85 都会失败。"""
    cases = [{"id": "a"}, {"id": "b"}]

    passed = evaluate_thresholds(cases, [0.90, 0.86])
    bad_case = evaluate_thresholds(cases, [0.79, 0.99])
    bad_mean = evaluate_thresholds(cases, [0.84, 0.84])

    assert passed.passed is True
    assert bad_case.passed is False
    assert bad_mean.passed is False
