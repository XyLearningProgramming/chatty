"""Modern tool registry for langchain agents."""

from abc import abstractmethod
from typing import Annotated, List, Protocol, Self, Type

from fastapi import Depends
from langchain.tools import BaseTool

from chatty.configs.config import get_app_config
from chatty.configs.persona import PersonaToolConfig
from chatty.infra import singleton

from .processors import HtmlHeadTitleMeta, Processor
from .url_tool import FixedURLTool


class ToolBuilder(Protocol):
    """Abstract base class for building tools."""

    @property
    def tool_type(self) -> str:
        """Return the tool type this builder handles."""
        return ""

    @abstractmethod
    def from_config(self, config: PersonaToolConfig) -> Self:
        """Build a tool from configuration."""
        pass


class ToolRegistry:
    """Registry for managing tool builders and creating tool instances."""

    _known_tools: dict[str, Type[FixedURLTool]] = {"url": FixedURLTool}
    _known_processors: dict[str, Type[Processor]] = {
        "html_head_title_meta": HtmlHeadTitleMeta
    }

    def __init__(self, config: list[PersonaToolConfig]):
        """Initialize registry and create tools from configuration."""
        self._tools: List[BaseTool] = []
        self._load_tools_from_config(config)

    def _load_tools_from_config(self, configs: list[PersonaToolConfig]) -> None:
        """Load tools from configuration with processor support."""
        for config in configs:
            try:
                # Find the appropriate tool builder
                tool_builder = self._known_tools.get(config.tool_type)
                if not tool_builder:
                    print(
                        f"Warning: Unknown tool type '{config.tool_type}' for tool '{config.name}'"
                    )
                    continue

                # Create the tool instance
                tool = tool_builder.from_config(config)

                # Apply processors if specified
                if config.processors:
                    tool = self._apply_processors(tool, config.processors)

                self._tools.append(tool)

            except Exception as e:
                print(f"Warning: Failed to create tool '{config.name}': {e}")

    def _apply_processors(self, tool: BaseTool, processor_names: list[str]) -> BaseTool:
        """Apply processors to a tool instance."""
        processors = []

        for processor_name in processor_names:
            processor_class = self._known_processors.get(processor_name)
            if processor_class:
                processors.append(processor_class())
            else:
                print(f"Warning: Unknown processor '{processor_name}'")

        if processors:
            # Apply processors by wrapping the tool's _run and _arun methods
            original_run = tool._run
            original_arun = tool._arun

            def _run_with_processors(*args, **kwargs) -> str:
                """Wrapped _run method that applies processors to the result."""
                result = original_run(*args, **kwargs)
                for processor in processors:
                    result = processor.process(result)
                return result

            async def _arun_with_processors(*args, **kwargs) -> str:
                """Wrapped _arun method that applies processors to the result."""
                result = await original_arun(*args, **kwargs)
                for processor in processors:
                    result = processor.process(result)
                return result

            # Replace the methods on the instance
            tool._run = _run_with_processors
            tool._arun = _arun_with_processors

        return tool

    def get_tools(self) -> List[BaseTool]:
        """Get loaded tools."""
        return self._tools[:]

    def register_tool_builder(self, tool_type: str, builder: ToolBuilder) -> None:
        """Register a new tool builder."""
        self._known_tools[tool_type] = builder

    def register_processor(
        self, processor_name: str, processor_class: Type[Processor]
    ) -> None:
        """Register a new processor class."""
        self._known_processors[processor_name] = processor_class


# Global registry instance
@singleton
def get_tool_registry(
    config: Annotated[
        list[PersonaToolConfig], Depends(lambda: get_app_config().persona.tools)
    ],
) -> ToolRegistry:
    """Get the global tool registry instance."""
    return ToolRegistry(config)
