"""Knowledge source model for persona config.

Defines where content comes from (content_url or inline content).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import timedelta

from pydantic import BaseModel, Field, field_validator, model_validator

from .persona_processors import ProcessorRef

HttpGet = Callable[[str, float], Awaitable[str]]
"""``async (url, timeout) -> text`` — matches ``HttpClient.get``."""


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

    def get_processors(self) -> list:
        """Resolve ``self.processors`` refs into instantiated processors."""
        if not self.processors:
            return []
        from chatty.infra.processor_utils import get_processor

        return [
            get_processor(p) if isinstance(p, str)
            else get_processor(p.name, **p.model_dump(exclude={"name"}, exclude_none=True))
            for p in self.processors
        ]

    async def get_content(
        self,
        http_get: HttpGet,
        extra_processors: list | None = None,
    ) -> str:
        """Return processed content — inline or fetched via *http_get*.

        Applies source-level processors first, then *extra_processors*.
        Callers decide whether to cache the result.
        """
        if self.content:
            text = self.content
        else:
            text = await http_get(
                self.content_url, self.timeout.total_seconds()
            )
        for p in self.get_processors() + (extra_processors or []):
            text = p.process(text)
        return text
