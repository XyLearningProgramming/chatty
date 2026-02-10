"""Base classes and protocols for tools."""

from abc import abstractmethod
from typing import Protocol, Self

from chatty.configs.tools import ToolConfig


class ToolBuilder(Protocol):
    """Protocol for tool builders that can create tools from config."""

    @property
    def tool_type(self) -> str:
        """Return the tool type this builder handles."""
        return ""

    @classmethod
    @abstractmethod
    def from_config(cls, config: ToolConfig) -> Self:
        """Build a tool from configuration."""
        ...
