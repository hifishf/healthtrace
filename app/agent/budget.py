"""LLM 调用成本控制：按会话累计 token 估算，超限强制收口不再调用 LLM。"""
from dataclasses import dataclass


@dataclass
class TokenBudget:
    """单会话的 token 预算计数器。

    估算口径：中英文混合文本按「字符数 / 4」粗估 token，够用于成本上限控制；
    不追求与供应商计费精确一致（精确计费以 Langfuse 的 usage 为准）。
    """

    limit: int
    used: int = 0

    def estimate(self, text: str) -> int:
        """把文本长度粗估为 token 数（至少 1）。"""
        return max(1, len(text or "") // 4)

    def exhausted(self) -> bool:
        """预算是否已耗尽。"""
        return self.used >= self.limit

    def consume(self, tokens: int) -> None:
        """累计已用 token。"""
        self.used += tokens
