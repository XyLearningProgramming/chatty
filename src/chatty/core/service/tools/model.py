"""Base classes and protocols for tools."""

from abc import abstractmethod
from typing import Any, Protocol, Self

from chatty.configs.tools import ToolConfig


class ToolBuilder(Protocol):
    """Protocol for tool builders that can create tools from config.

    Each concrete builder handles one ``tool_type`` (e.g. ``'url'``).
    The registry groups all YAML entries by type and calls
    ``from_configs`` once per group to produce a **single dispatcher
    tool** that the model sees.
    """

    @property
    def tool_type(self) -> str:
        """Return the tool type this builder handles."""
        return ""

    @classmethod
    @abstractmethod
    def from_configs(
        cls,
        configs: list[ToolConfig],
        *,
        processors: dict[str, list[Any]] | None = None,
    ) -> Self:
        """Build a single dispatcher tool from a group of configs.

        Parameters
        ----------
        configs:
            All ``ToolConfig`` entries sharing the same ``tool_type``.
        processors:
            Mapping of config name → resolved ``Processor`` instances.
        """
        ...
