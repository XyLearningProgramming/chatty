"""Base classes and protocols for tools."""

from abc import abstractmethod
from typing import Any, Protocol, Self

from chatty.configs.persona import KnowledgeSection


class ToolBuilder(Protocol):
    """Protocol for tool builders that can create tools from config.

    Each concrete builder handles one ``tool_type`` (e.g. ``'url'``).
    The registry filters ``persona.sections`` by convention and calls
    ``from_sections`` once per group to produce a **single dispatcher
    tool** that the model sees.
    """

    @property
    def tool_type(self) -> str:
        """Return the tool type this builder handles."""
        return ""

    @classmethod
    @abstractmethod
    def from_sections(
        cls,
        sections: list[KnowledgeSection],
        *,
        processors: dict[str, list[Any]] | None = None,
    ) -> Self:
        """Build a single dispatcher tool from knowledge sections.

        Parameters
        ----------
        sections:
            ``KnowledgeSection`` entries matching this builder's type
            (e.g. sections with ``source_url`` for the URL builder).
        processors:
            Mapping of section title to resolved ``Processor`` instances.
        """
        ...
