"""Tool registry: builds tools from persona config and dispatches calls.

Tools are built from ``persona.tools`` declarations.  Each declaration
references knowledge sources by id.
"""

import asyncio
import logging
from datetime import timedelta
from typing import Annotated

from fastapi import Depends

from chatty.configs.config import AppConfig, get_app_config
from chatty.configs.persona import KnowledgeSource, ToolDeclaration
from chatty.configs.system import PromptConfig

from .model import ToolDefinition
from .search_tool import SearchTool

logger = logging.getLogger(__name__)


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
        self._tools_by_name: dict[str, SearchTool] = {t.name: t for t in self._tools}
        self._timeout_seconds = tool_timeout.total_seconds()

    def get_tools(self) -> list[ToolDefinition]:
        """Return OpenAI tool definitions for all registered tools."""
        return [t.to_tool_definition() for t in self._tools]

    async def execute(self, name: str, arguments: dict[str, str]) -> str:
        """Dispatch a tool call by name with a per-invocation timeout.

        Raises ``KeyError`` if *name* is not a registered tool.
        """
        tool = self._tools_by_name[name]
        async with asyncio.timeout(self._timeout_seconds):
            return await tool.execute(**arguments)

    # ------------------------------------------------------------------

    @staticmethod
    def _build_tools(
        declarations: list[ToolDeclaration],
        sources: dict[str, KnowledgeSource],
        prompt: PromptConfig,
    ) -> list[SearchTool]:
        return [
            SearchTool.from_declaration(decl, sources, prompt) for decl in declarations
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
