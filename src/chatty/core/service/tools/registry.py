"""Tool registry for LangGraph agents.

Tools are **derived from ``persona.sections``** by convention.  Sections
with a ``source_url`` become URL tools automatically.  The registry
re-reads the persona config on every ``get_tools()`` call so that
ConfigMap updates are picked up without a redeploy.
"""

import asyncio
import functools
import logging
from datetime import timedelta
from typing import Any, Type

from langchain_core.tools import BaseTool

from chatty.configs.config import get_app_config
from chatty.configs.persona import KnowledgeSection
from chatty.infra import singleton

from .model import ToolBuilder
from .processors import HtmlHeadTitleMeta, Processor
from .url_tool import URLDispatcherTool

logger = logging.getLogger(__name__)


def _apply_timeout(tool: BaseTool, timeout: timedelta) -> BaseTool:
    """Wrap a tool's ``_arun`` and ``_run`` with a per-invocation timeout.

    The wrapper is monkey-patched onto the *instance*, so it sits on top
    of any previously applied processors.
    """
    seconds = timeout.total_seconds()
    original_arun = tool._arun

    @functools.wraps(original_arun)
    async def _timed_arun(*args, **kwargs):  # type: ignore[no-untyped-def]
        async with asyncio.timeout(seconds):
            return await original_arun(*args, **kwargs)

    tool._arun = _timed_arun  # type: ignore[method-assign]
    return tool


class ToolRegistry:
    """Registry that derives tools from ``persona.sections``.

    Sections with ``source_url`` are treated as URL tools.  Each builder
    receives the matching sections and produces a single dispatcher tool.

    When constructed with explicit *sections* (useful in tests), tools
    are eagerly built and cached.  Otherwise ``get_tools()`` re-reads
    config via ``get_app_config()`` on every call so that ConfigMap
    updates are reflected immediately.
    """

    _known_tools: dict[str, Type[ToolBuilder]] = {
        cls.tool_type: cls for cls in [URLDispatcherTool]
    }
    _known_processors: dict[str, Type[Processor]] = {
        cls.processor_name: cls for cls in [HtmlHeadTitleMeta]
    }

    def __init__(
        self, sections: list[KnowledgeSection] | None = None
    ) -> None:
        """Create a registry, optionally with fixed sections for testing."""
        self._tools: list[BaseTool] | None = None
        if sections is not None:
            self._tools = self._build_tools(sections)

    def get_tools(self) -> list[BaseTool]:
        """Build and return tools from the latest persona config.

        Each tool is wrapped with the global ``tool_timeout`` from
        ``ChatConfig`` so that no single invocation can run forever.
        """
        if self._tools is not None:
            return list(self._tools)

        app_config = get_app_config()
        sections = app_config.persona.sections
        tools = self._build_tools(sections)
        for tool in tools:
            _apply_timeout(tool, app_config.chat.tool_timeout)
        return tools

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sections_with_url(
        sections: list[KnowledgeSection],
    ) -> list[KnowledgeSection]:
        """Filter sections that have a ``source_url`` (→ URL tools)."""
        return [s for s in sections if s.source_url]

    def _build_tools(
        self, sections: list[KnowledgeSection]
    ) -> list[BaseTool]:
        """Build tools from persona knowledge sections.

        Currently only the ``url`` tool type is supported.  All
        sections with ``source_url`` are grouped into a single URL
        dispatcher.
        """
        url_sections = self._sections_with_url(sections)
        if not url_sections:
            return []

        tool_builder = self._known_tools.get("url")
        if not tool_builder:
            raise NotImplementedError("Tool type 'url' is not supported.")

        # Resolve processors per section title
        resolved_processors: dict[str, list[Any]] = {}
        for section in url_sections:
            if section.processors:
                resolved_processors[section.title] = (
                    self._resolve_processors(section.processors)
                )

        tool = tool_builder.from_sections(
            url_sections,
            processors=resolved_processors if resolved_processors else None,
        )
        return [tool]

    def _resolve_processors(
        self, processor_names: list[str]
    ) -> list[Processor]:
        """Map processor names to instantiated ``Processor`` objects."""
        processors: list[Processor] = []
        for name in processor_names:
            processor_class = self._known_processors.get(name)
            if not processor_class:
                raise NotImplementedError(
                    f"Processor '{name}' is not supported."
                )
            processors.append(processor_class())
        return processors


@singleton
def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry singleton.

    The registry itself is a singleton, but it reads fresh sections
    via ``get_app_config()`` on every ``get_tools()`` call.
    """
    return ToolRegistry()
