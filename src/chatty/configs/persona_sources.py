"""Knowledge source model for persona config.

Defines where content comes from (content_url or inline content).
"""

from __future__ import annotations

from datetime import timedelta

from pydantic import BaseModel, Field, field_validator, model_validator

from .persona_processors import ProcessorRef


class KnowledgeSource(BaseModel):
    """A single knowledge source.

    Defines WHERE content comes from.  Referenced by id (dict key in
    ``PersonaConfig.sources``) from tool and embed declarations.

    Exactly one of ``content_url`` or ``content`` must be set.
    """

    description: str = Field(
        default="",
        description="Human-readable description (used in tool hints)",
    )
    content_url: str = Field(
        default="",
        description="URL to fetch content from",
    )
    content: str = Field(
        default="",
        description="Inline text content",
    )
    timeout: timedelta = Field(
        default_factory=lambda: timedelta(seconds=30),
        description="HTTP request timeout (only for URL sources). "
        "YAML may use seconds as int, e.g. timeout: 30.",
    )
    processors: list[ProcessorRef] = Field(
        default_factory=list,
        description="Source-level processors applied on every fetch",
    )

    @field_validator("timeout", mode="before")
    @classmethod
    def _timeout_seconds(cls, v: int | timedelta) -> timedelta:
        """Accept int as seconds for YAML convenience."""
        if isinstance(v, int):
            return timedelta(seconds=v)
        return v

    @model_validator(mode="after")
    def _check_content_or_url(self) -> KnowledgeSource:
        has_url = bool(self.content_url)
        has_content = bool(self.content)
        if has_url and has_content:
            raise ValueError(
                "Source must have content_url OR content, not both"
            )
        if not has_url and not has_content:
            raise ValueError(
                "Source must have either content_url or content"
            )
        return self
