"""Tool registry for LangGraph agents.

The registry is a singleton that holds **no tool definitions in memory**.
Each call to ``get_tools()`` delegates to ``get_app_config().tools``
which re-reads YAML config from disk, so ConfigMap updates are picked
up without a redeploy.
"""

import logging
from typing import Type

from langchain_core.tools import BaseTool

from chatty.configs.config import get_app_config
from chatty.configs.tools import ToolConfig
from chatty.infra import singleton

from .model import ToolBuilder
from .processors import HtmlHeadTitleMeta, Processor, with_processors
from .url_tool import FixedURLTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for managing tool builders and creating tool instances.

    Holds no tool definitions in memory.  ``get_tools()`` re-reads config
    via ``get_app_config()`` on every call so that ConfigMap updates are
    reflected immediately.
    """

    _known_tools: dict[str, Type[ToolBuilder]] = {
        cls.tool_type: cls for cls in [FixedURLTool]
    }
    _known_processors: dict[str, Type[Processor]] = {
        cls.processor_name: cls for cls in [HtmlHeadTitleMeta]
    }

    def get_tools(self) -> list[BaseTool]:
        """Build and return tools from the latest config on disk."""
        configs = get_app_config().tools
        return self._build_tools(configs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_tools(self, configs: list[ToolConfig]) -> list[BaseTool]:
        """Construct tool instances from config objects."""
        tools: list[BaseTool] = []
        for config in configs:
            tool_builder = self._known_tools.get(config.tool_type)
            if not tool_builder:
                logger.warning(
                    "Tool type '%s' is not supported, skipping.", config.tool_type
                )
                continue

            tool = tool_builder.from_config(config)

            if config.processors:
                tool = self._apply_processors(tool, config.processors)

            tools.append(tool)
        return tools

    def _apply_processors(
        self, tool: BaseTool, processor_names: list[str]
    ) -> BaseTool:
        """Apply processors to a tool instance."""
        processors = []
        for processor_name in processor_names:
            processor_class = self._known_processors.get(processor_name)
            if not processor_class:
                logger.warning(
                    "Processor '%s' is not supported, skipping.", processor_name
                )
                continue
            processors.append(processor_class())
        return with_processors(*processors)(tool)


@singleton
def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry singleton.

    The registry itself is a singleton, but it reads fresh tool definitions
    via ``get_app_config()`` on every ``get_tools()`` call.
    """
    return ToolRegistry()
