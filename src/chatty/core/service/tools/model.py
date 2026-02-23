"""Tool definition models and builder protocol.

``ToolDefinition`` / ``FunctionDefinition`` mirror the OpenAI
chat-completion tool spec (and the ``ChatCompletionTool`` TypedDict
from llama-cpp-python) as proper Pydantic models.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Literal, Protocol, Self

from pydantic import BaseModel, Field

from chatty.configs.persona import KnowledgeSource, ToolDeclaration

if TYPE_CHECKING:
    from chatty.configs.system import PromptConfig


# ---------------------------------------------------------------------------
# JSON Schema sub-models for function parameters
# ---------------------------------------------------------------------------


class PropertyDefinition(BaseModel):
    """Single property inside a JSON Schema ``properties`` block."""

    type: str
    description: str = ""
    enum: list[str] | None = None


class ParametersDefinition(BaseModel):
    """Top-level ``parameters`` object for a function definition.

    Represents a JSON Schema of ``type: "object"`` with named
    ``properties`` and a ``required`` list.
    """

    type: Literal["object"] = "object"
    properties: dict[str, PropertyDefinition] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# OpenAI-compatible tool definition models
# ---------------------------------------------------------------------------


class FunctionDefinition(BaseModel):
    """Function definition nested inside a ``ToolDefinition``."""

    name: str
    description: str = ""
    parameters: ParametersDefinition = Field(
        default_factory=ParametersDefinition,
    )


class ToolDefinition(BaseModel):
    """OpenAI-compatible tool definition.

    Maps 1-to-1 with ``ChatCompletionTool`` from llama-cpp-python::

        { "type": "function", "function": { "name": ..., ... } }
    """

    type: Literal["function"] = "function"
    function: FunctionDefinition


# ---------------------------------------------------------------------------
# Builder protocol
# ---------------------------------------------------------------------------


class ToolBuilder(Protocol):
    """Protocol for objects that can be built from a ``ToolDeclaration``
    and produce an OpenAI tool definition + execute tool calls.
    """

    @classmethod
    @abstractmethod
    def from_declaration(
        cls,
        declaration: ToolDeclaration,
        sources: dict[str, KnowledgeSource],
        prompt: PromptConfig,
    ) -> Self:
        """Build a tool from a persona config declaration."""
        ...

    def to_tool_definition(self) -> ToolDefinition:
        """Return the OpenAI-compatible tool definition."""
        ...

    async def execute(self, **kwargs: str) -> str:
        """Execute the tool and return a plain-text result."""
        ...
