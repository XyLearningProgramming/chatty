"""Tool registry for LangGraph agents.

Tools are built from ``persona.tools`` declarations.  Each declaration
references sources by id and specifies an optional action-level
processor chain.  The registry merges source-level and action-level
processors, then delegates to the matching ``ToolBuilder``.
"""

import asyncio
import functools
import logging
from datetime import timedelta
from typing import Annotated, Any, Type

from fastapi import Depends
from langchain_core.tools import BaseTool

from chatty.configs.config import AppConfig, get_app_config
from chatty.configs.persona import (
    KnowledgeSource,
    ToolDeclaration,
)

from .model import ToolBuilder
from .processors import Processor, resolve_processors
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
    """Registry that builds tools from persona tool declarations.

    Receives the full ``PersonaConfig`` (or tools + sources) and
    ``tool_timeout`` at construction time.  Tests can pass these
    directly.
    """

    _known_tools: dict[str, Type[ToolBuilder]] = {
        cls.tool_type: cls for cls in [URLDispatcherTool]
    }

    def __init__(
        self,
        tools: list[ToolDeclaration],
        sources: dict[str, KnowledgeSource],
        tool_timeout: timedelta,
    ) -> None:
        self._tools = self._build_tools(tools, sources)
        for tool in self._tools:
            _apply_timeout(tool, tool_timeout)

    def get_tools(self) -> list[BaseTool]:
        """Return the built tools."""
        return list(self._tools)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_tools(
        self,
        declarations: list[ToolDeclaration],
        sources: dict[str, KnowledgeSource],
    ) -> list[BaseTool]:
        """Build tools from persona tool declarations."""
        tools: list[BaseTool] = []

        for decl in declarations:
            tool_builder = self._known_tools.get(decl.type)
            if not tool_builder:
                raise NotImplementedError(
                    f"Tool type '{decl.type}' is not supported."
                )

            merged_processors = self._merge_processors(decl, sources)

            tool = tool_builder.from_declaration(
                decl,
                sources,
                processors=(
                    merged_processors if merged_processors else None
                ),
            )
            tools.append(tool)

        return tools

    @staticmethod
    def _merge_processors(
        decl: ToolDeclaration,
        sources: dict[str, KnowledgeSource],
    ) -> dict[str, list[Any]]:
        """Merge source-level and action-level processors per source id.

        Returns a mapping of source_id -> list of resolved Processor
        instances (source processors first, then action processors).
        """
        action_procs: list[Processor] = []
        if decl.processors:
            action_procs = resolve_processors(decl.processors)

        merged: dict[str, list[Any]] = {}
        for source_id in decl.sources:
            source = sources[source_id]
            chain: list[Processor] = []

            if source.processors:
                chain.extend(resolve_processors(source.processors))

            chain.extend(action_procs)

            if chain:
                merged[source_id] = chain

        return merged


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


def get_tool_registry(
    config: Annotated[AppConfig, Depends(get_app_config)],
) -> ToolRegistry:
    """Build a ``ToolRegistry`` from the latest config.

    Config is injected via ``Depends(get_app_config)`` so there are
    no hidden lookups and the dependency can be overridden in tests.
    """
    return ToolRegistry(
        tools=config.persona.tools,
        sources=config.persona.sources,
        tool_timeout=config.chat.tool_timeout,
    )
