"""Base classes and protocols for tools."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol, Self

from chatty.configs.persona import KnowledgeSource, ToolDeclaration

if TYPE_CHECKING:
    from chatty.configs.system import PromptConfig


class ToolBuilder(Protocol):
    """Protocol for tool builders that create tools from config.

    Each concrete builder handles one ``tool_type`` (e.g. ``'url_dispatcher'``).
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
        prompt: PromptConfig,
    ) -> Self:
        """Build a single dispatcher tool from a tool declaration.

        Parameters
        ----------
        declaration:
            The ``ToolDeclaration`` entry from persona config.
        sources:
            Full ``persona.sources`` dict for resolving source ids.
        prompt:
            Prompt templates for tool descriptions and error messages.
        """
        ...
