"""Tool registry for LangGraph agents.

The registry is a singleton that holds **no tool definitions in memory**.
Each call to ``get_tools()`` delegates to ``get_app_config().tools``
which re-reads YAML config from disk, so ConfigMap updates are picked
up without a redeploy.
"""

import asyncio
import functools
import logging
from collections import defaultdict
from datetime import timedelta
from typing import Any, Type

from langchain_core.tools import BaseTool

from chatty.configs.config import get_app_config
from chatty.configs.tools import ToolConfig
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
    """Registry for managing tool builders and creating tool instances.

    Groups YAML tool configs by ``tool_type`` and calls each builder's
    ``from_configs`` once per group to produce a single dispatcher tool
    per type.

    When constructed with explicit *configs* (useful in tests), tools are
    eagerly built and cached.  Otherwise ``get_tools()`` re-reads config
    via ``get_app_config()`` on every call so that ConfigMap updates are
    reflected immediately.
    """

    _known_tools: dict[str, Type[ToolBuilder]] = {
        cls.tool_type: cls for cls in [URLDispatcherTool]
    }
    _known_processors: dict[str, Type[Processor]] = {
        cls.processor_name: cls for cls in [HtmlHeadTitleMeta]
    }

    def __init__(self, configs: list[ToolConfig] | None = None):
        """Create a registry, optionally with fixed configs for testing."""
        self._tools: list[BaseTool] | None = None
        if configs is not None:
            # Eagerly build (and validate) when static configs are provided
            self._tools = self._build_tools(configs)

    def get_tools(self) -> list[BaseTool]:
        """Build and return tools from the latest config on disk.

        Each tool is wrapped with the global ``tool_timeout`` from
        ``ChatConfig`` so that no single invocation can run forever.
        """
        if self._tools is not None:
            return list(self._tools)  # return a fresh list (same objects)

        app_config = get_app_config()
        tools = self._build_tools(app_config.tools)
        for tool in tools:
            _apply_timeout(tool, app_config.chat.tool_timeout)
        return tools

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_tools(self, configs: list[ToolConfig]) -> list[BaseTool]:
        """Group configs by ``tool_type`` and create one dispatcher per group."""
        groups: dict[str, list[ToolConfig]] = defaultdict(list)
        for config in configs:
            groups[config.tool_type].append(config)

        tools: list[BaseTool] = []
        for tool_type, group_configs in groups.items():
            tool_builder = self._known_tools.get(tool_type)
            if not tool_builder:
                raise NotImplementedError(
                    f"Tool type '{tool_type}' is not supported."
                )

            # Resolve processors per config name
            resolved_processors: dict[str, list[Any]] = {}
            for cfg in group_configs:
                if cfg.processors:
                    resolved_processors[cfg.name] = self._resolve_processors(
                        cfg.processors
                    )

            tool = tool_builder.from_configs(
                group_configs,
                processors=resolved_processors if resolved_processors else None,
            )
            tools.append(tool)

        return tools

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

    The registry itself is a singleton, but it reads fresh tool definitions
    via ``get_app_config()`` on every ``get_tools()`` call.
    """
    return ToolRegistry()
