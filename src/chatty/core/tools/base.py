"""Base classes and protocols for tools."""

from abc import abstractmethod
from typing import Protocol, Self

from chatty.configs.persona import PersonaToolConfig


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