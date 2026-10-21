"""LangGraph Postgres checkpointer：创建 + 建断点表（async 上下文管理器）。"""
from contextlib import asynccontextmanager
from typing import AsyncIterator

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.config import settings


@asynccontextmanager
async def open_checkpointer() -> AsyncIterator[AsyncPostgresSaver]:
    """打开并初始化 Postgres checkpointer，退出时关闭底层连接池。

    用法：async with open_checkpointer() as saver: await saver.setup()（内部已做）。
    interrupt 挂起的断点落在该 checkpointer 的 PostgreSQL 里，恢复必须带同一 thread_id。
    """
    async with AsyncPostgresSaver.from_conn_string(
        settings.checkpoint_database_url
    ) as saver:
        await saver.setup()
        yield saver
