"""Helpers for interacting with MCP servers."""

from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional, Tuple

import mcp
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.exceptions import McpError

from ..config import logger


def ensure_streamable_http_url(url: str) -> str:
    """Ensure the MCP URL contains the FastMCP streamable HTTP path."""
    from urllib.parse import urlparse, urlunparse

    parsed = urlparse(url)
    path = parsed.path or ""
    if not path or path == "/":
        path = "/mcp"
    return urlunparse(parsed._replace(path=path))


async def discover_tools(mcp_url: str | None = None) -> Tuple[List[Dict[str, Any]], str]:
    """Discover available tools from the MCP server."""
    mcp_url = mcp_url or os.getenv("MCP_SERVER_URL", "http://127.0.0.1:9082")
    resolved_url = ensure_streamable_http_url(mcp_url)
    try:
        async with streamablehttp_client(resolved_url) as (read_stream, write_stream, _):
            async with mcp.ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.list_tools()
                logger.debug("Discovered %d tools via %s", len(result.tools), resolved_url)
                return [tool.model_dump() for tool in result.tools], resolved_url
    except Exception as exc:
        logger.exception("Failed to discover tools via %s: %s", resolved_url, exc)
        return [], resolved_url


async def call_mcp_tool(
    tool_name: str,
    arguments: Optional[Dict[str, Any]],
    mcp_url: str,
) -> Dict[str, Any]:
    """Invoke an MCP tool and return the dictionary payload."""
    normalized_args: Dict[str, Any]
    if not arguments:
        normalized_args = {"payload": {}}
    elif "payload" in arguments:
        normalized_args = arguments  # type: ignore[assignment]
    else:
        normalized_args = {"payload": arguments}

    logger.debug("Calling MCP tool '%s' with args %s", tool_name, normalized_args)
    async with streamablehttp_client(mcp_url) as (read_stream, write_stream, _):
        async with mcp.ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            try:
                call_result = await session.call_tool(tool_name, normalized_args)
            except McpError as exc:
                err_data = getattr(exc, "data", None)
                logger.error(
                    "MCP tool '%s' failed: %s | data=%s",
                    tool_name,
                    exc,
                    err_data,
                )
                raise
            payload = call_result.model_dump()
            payload.setdefault("tool", tool_name)
            return payload


def extract_text_content(tool_result: Dict[str, Any]) -> List[str]:
    texts: List[str] = []
    for block in tool_result.get("content", []) or []:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if text:
                texts.append(text)
    return texts


def parse_first_json_block(tool_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for text in extract_text_content(tool_result):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            continue
    return None


async def fetch_item_info(uri: str, mcp_url: str) -> Optional[Dict[str, Any]]:
    if not uri:
        return None
    try:
        result = await call_mcp_tool("SpotifyGetInfo", {"item_uri": uri}, mcp_url)
    except Exception:
        return None
    data = parse_first_json_block(result)
    if not data:
        return None
    item = data.get("item") if isinstance(data, dict) else None
    if isinstance(item, dict):
        return item
    return data if isinstance(data, dict) else None


__all__ = [
    "ensure_streamable_http_url",
    "discover_tools",
    "call_mcp_tool",
    "extract_text_content",
    "parse_first_json_block",
    "fetch_item_info",
]
