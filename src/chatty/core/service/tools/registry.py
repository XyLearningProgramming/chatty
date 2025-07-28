"""Modern tool registry for langchain agents."""

from typing import Annotated, List, Type

from fastapi import Depends
from langchain.tools import BaseTool

from chatty.configs.config import get_app_config
from chatty.configs.persona import PersonaToolConfig
from chatty.infra import singleton

from .model import ToolBuilder
from .processors import HtmlHeadTitleMeta, Processor, with_processors
from .url_tool import FixedURLTool


class ToolRegistry:
    """Registry for managing tool builders and creating tool instances."""

    _known_tools: dict[str, Type[ToolBuilder]] = {
        cls.tool_type: cls for cls in [FixedURLTool]
    }
    _known_processors: dict[str, Type[Processor]] = {
        cls.processor_name: cls for cls in [HtmlHeadTitleMeta]
    }

    def __init__(self, configs: list[PersonaToolConfig]):
        """Initialize registry and create tools from configuration."""
        self._tools: List[BaseTool] = []

        for config in configs:
            # Find the appropriate tool builder
            tool_builder = self._known_tools.get(config.tool_type)
            if not tool_builder:
                raise NotImplementedError(
                    f"Tool type '{config.tool_type}' is not supported."
                )

            # Create the tool instance
            tool = tool_builder.from_config(config)

            # Apply processors if specified
            if config.processors:
                tool = self._apply_processors(tool, config.processors)

            self._tools.append(tool)

    def _apply_processors(self, tool: BaseTool, processor_names: list[str]) -> BaseTool:
        """Apply processors to a tool instance."""
        processors = []
        for processor_name in processor_names:
            processor_class = self._known_processors.get(processor_name)
            if not processor_class:
                raise NotImplementedError(
                    f"Processor '{processor_name}' is not supported."
                )
            processors.append(processor_class())
        return with_processors(*processors)(tool)

    def get_tools(self) -> List[BaseTool]:
        """Get loaded tools."""
        return self._tools[:]


# Global registry instance
@singleton
def get_tool_registry(
    config: Annotated[
        list[PersonaToolConfig], Depends(lambda: get_app_config().persona.tools)
    ],
) -> ToolRegistry:
    """Get the global tool registry instance."""
    return ToolRegistry(config)
