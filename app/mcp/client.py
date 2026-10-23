"""进程内 FastMCP 客户端：Agent 层经此访问数据，不直接写 SQL。"""
from typing import Any

from fastmcp import Client

from app.mcp.server import mcp


async def call_tool(name: str, arguments: dict[str, Any]) -> Any:
    """调用进程内 MCP 工具并返回结构化数据（工具的 dict 返回值）。"""
    async with Client(mcp) as client:
        result = await client.call_tool(name, arguments)
        return result.data
