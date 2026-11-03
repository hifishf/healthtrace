"""第 10 周 RAG 种子确定性断言：数量、唯一键、分类与来源完整性。"""
from collections import Counter

from scripts.rag_seed_data import build_knowledge_entries


def test_knowledge_seed_has_at_least_500_unique_entries() -> None:
    """知识库达到 500+，且 knowledge_key、标题和正文均不重复。"""
    entries = build_knowledge_entries()

    assert len(entries) >= 500
    assert len({entry["knowledge_key"] for entry in entries}) == len(entries)
    assert len({entry["title"] for entry in entries}) == len(entries)
    assert len({entry["content"] for entry in entries}) == len(entries)


def test_knowledge_seed_source_and_content_are_complete() -> None:
    """每条知识都有正文、可核验 URL、版本和正确长度的内容哈希。"""
    entries = build_knowledge_entries()

    for entry in entries:
        assert entry["title"].strip()
        assert entry["content"].strip()
        assert entry["source_name"].strip()
        assert entry["source_url"].startswith("https://")
        assert entry["source_version"]
        assert len(entry["content_hash"]) == 64
        assert int(entry["content_hash"], 16) >= 0


def test_knowledge_seed_covers_required_categories() -> None:
    """六类知识均存在，报告指标知识绑定 metric_codes。"""
    entries = build_knowledge_entries()
    counts = Counter(entry["category"] for entry in entries)

    assert set(counts) == {
        "symptom",
        "disease",
        "association",
        "lab",
        "drug",
        "safety",
    }
    assert counts["lab"] == 18
    assert all(entry["metric_codes"] for entry in entries if entry["category"] == "lab")
