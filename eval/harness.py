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
import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

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


def run_semantic(args: argparse.Namespace) -> int:
    """语义评估（占位）：先能发现 golden 用例，judge 打分在第 3、4 步接入。"""
    cases = list(iter_golden())
    if not cases:
        print("语义评估尚未接入：第 3 步加 judge，第 4 步加 golden 集。")
        return 0
    print(f"发现 golden 用例 {len(cases)} 条（目录：{GOLDEN_DIR}）；judge 打分逻辑在第 3 步接入。")
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
    sem.set_defaults(func=run_semantic)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
