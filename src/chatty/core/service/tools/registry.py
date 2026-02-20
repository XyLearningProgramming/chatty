"""Tool registry for LangGraph agents.

Tools are built from ``persona.tools`` declarations.  Each declaration
references sources by id.
"""

import asyncio
import functools
import logging
from datetime import timedelta
from typing import Annotated

from fastapi import Depends
from langchain_core.tools import BaseTool

from chatty.configs.config import AppConfig, get_app_config
from chatty.configs.persona import (
    KnowledgeSource,
    ToolDeclaration,
)
from chatty.configs.system import PromptConfig

from .url_tool import URLDispatcherTool

logger = logging.getLogger(__name__)


def _apply_timeout(tool: BaseTool, timeout: timedelta) -> BaseTool:
    """Wrap a tool's ``_arun`` with a per-invocation timeout."""
    seconds = timeout.total_seconds()
    original_arun = tool._arun

    @functools.wraps(original_arun)
    async def _timed_arun(*args, **kwargs):  # type: ignore[no-untyped-def]
        async with asyncio.timeout(seconds):
            return await original_arun(*args, **kwargs)

    tool._arun = _timed_arun  # type: ignore[method-assign]
    return tool


class ToolRegistry:
    """Registry that builds tools from persona tool declarations."""

    def __init__(
        self,
        tools: list[ToolDeclaration],
        sources: dict[str, KnowledgeSource],
        prompt: PromptConfig,
        tool_timeout: timedelta,
    ) -> None:
        self._tools = self._build_tools(tools, sources, prompt)
        for tool in self._tools:
            _apply_timeout(tool, tool_timeout)

    def get_tools(self) -> list[BaseTool]:
        """Return the built tools."""
        return list(self._tools)

    # ------------------------------------------------------------------

    @staticmethod
    def _build_tools(
        declarations: list[ToolDeclaration],
        sources: dict[str, KnowledgeSource],
        prompt: PromptConfig,
    ) -> list[BaseTool]:
        return [
            URLDispatcherTool.from_declaration(decl, sources, prompt)
            for decl in declarations
        ]


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


def get_tool_registry(
    config: Annotated[AppConfig, Depends(get_app_config)],
) -> ToolRegistry:
    """Build a ``ToolRegistry`` from the latest config."""
    return ToolRegistry(
        tools=config.persona.tools,
        sources=config.persona.sources,
        prompt=config.prompt,
        tool_timeout=config.chat.tool_timeout,
    )
