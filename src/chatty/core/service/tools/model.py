"""Base classes and protocols for tools."""

from abc import abstractmethod
from typing import Any, Protocol, Self

from chatty.configs.persona import KnowledgeSource, ToolDeclaration


class ToolBuilder(Protocol):
    """Protocol for tool builders that create tools from config.

    Each concrete builder handles one ``tool_type`` (e.g. ``'url'``).
    The registry calls ``from_declaration`` once per tool entry to
    produce a single dispatcher tool that the model sees.
    """

    @property
    def tool_type(self) -> str:
        """Return the tool type this builder handles."""
        return ""

    @classmethod
    @abstractmethod
    def from_declaration(
        cls,
        declaration: ToolDeclaration,
        sources: dict[str, KnowledgeSource],
        *,
        processors: dict[str, list[Any]] | None = None,
    ) -> Self:
        """Build a single dispatcher tool from a tool declaration.

        Parameters
        ----------
        declaration:
            The ``ToolDeclaration`` entry from persona config.
        sources:
            Full ``persona.sources`` dict for resolving source ids.
        processors:
            Mapping of source id to resolved ``Processor`` instances
            (source-level + action-level merged).
        """
        ...
