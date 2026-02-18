"""Tool and embed action declarations for persona config."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .persona_processors import ProcessorRef


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
