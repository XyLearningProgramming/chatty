"""Tool and embed action declarations for persona config."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .persona_processors import ProcessorRef


def _get_processors(refs: list[ProcessorRef]) -> list:
    """Turn a list of ``ProcessorRef`` values into instantiated processors."""
    if not refs:
        return []
    from chatty.infra.processor_utils import get_processor

    return [
        get_processor(p) if isinstance(p, str)
        else get_processor(p.name, **p.model_dump(exclude={"name"}, exclude_none=True))
        for p in refs
    ]


class ToolDeclaration(BaseModel):
    """A tool exposed to the LLM agent.

    References sources by id.  The ``type`` field selects which
    ``ToolBuilder`` implementation creates the LangChain tool.
    """

    name: str = Field(description="Tool name seen by the model")
    type: str = Field(
        default="url_dispatcher",
        description="ToolBuilder type key",
    )
    sources: list[str] = Field(
        description="Source ids this tool can access",
    )
    processors: list[ProcessorRef] = Field(
        default_factory=list,
        description="Action-level processors applied after source processors",
    )
    description: str = Field(
        default="",
        description="Tool description for the model",
    )

    def get_processors(self) -> list:
        return _get_processors(self.processors)


class EmbedDeclaration(BaseModel):
    """An embed action for RAG retrieval.

    ``match_hints`` are short, query-like phrases that get embedded in
    pgvector for cosine similarity matching.  When a user query matches,
    the full processed content of the referenced source is injected
    into the system prompt.
    """

    source: str = Field(description="Source id to embed")
    match_hints: list[str] = Field(
        default_factory=list,
        description="Short phrases embedded for query matching. "
        "If empty, the source description is used as fallback.",
    )
    processors: list[ProcessorRef] = Field(
        default_factory=list,
        description="Action-level processors for content before "
        "prompt injection",
    )

    def get_processors(self) -> list:
        return _get_processors(self.processors)
