"""Knowledge source model for persona config.

Defines where content comes from (content_url or inline content).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import timedelta

from pydantic import BaseModel, Field, model_validator

from .persona_processors import ProcessorRef

logger = logging.getLogger(__name__)

HttpGet = Callable[[str, float], Awaitable[str]]
"""``async (url, timeout) -> text`` — matches ``HttpClient.get``."""

# Module-level TTL cache for URL-fetched content.
# Key: (content_url, processor_fingerprint) → (raw_text, monotonic_timestamp)
_content_cache: dict[tuple[str, str], tuple[str, float]] = {}
_cache_lock = asyncio.Lock()


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
    cache_ttl: timedelta = Field(
        default_factory=lambda: timedelta(hours=6),
        description="TTL for in-process content cache (URL sources only). "
        "Set to 0 to disable. YAML may use seconds as int.",
    )
    processors: list[ProcessorRef] = Field(
        default_factory=list,
        description="Source-level processors applied on every fetch",
    )

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

    def _cache_key(self, extra_processors: list | None) -> tuple[str, str]:
        proc_names = [
            p.name if hasattr(p, "name") else type(p).__name__
            for p in self.get_processors() + (extra_processors or [])
        ]
        return (self.content_url, "|".join(proc_names))

    async def _get_content(
        self,
        http_get: HttpGet,
        extra_processors: list | None,
    ) -> str:
        """Fetch URL content, returning a cached copy when within *cache_ttl*."""
        ttl = self.cache_ttl.total_seconds()
        if ttl <= 0:
            return await http_get(
                self.content_url, self.timeout.total_seconds()
            )

        key = self._cache_key(extra_processors)
        now = time.monotonic()

        async with _cache_lock:
            entry = _content_cache.get(key)
            if entry is not None:
                cached_text, stored_at = entry
                if now - stored_at < ttl:
                    logger.debug("Content cache hit: %s", key[0])
                    return cached_text
                del _content_cache[key] # Lazy eviction.

        text = await http_get(
            self.content_url, self.timeout.total_seconds()
        )

        async with _cache_lock:
            _content_cache[key] = (text, time.monotonic())

        logger.debug("Content cache miss: %s", key[0])
        return text

    async def get_content(
        self,
        http_get: HttpGet,
        extra_processors: list | None = None,
    ) -> str:
        """Return processed content — inline or fetched via *http_get*.

        URL sources are served from an in-process TTL cache (see
        ``cache_ttl``).  Inline content bypasses the cache entirely.
        """
        if self.content:
            text = self.content
        else:
            text = await self._get_content(http_get, extra_processors)
        for p in self.get_processors() + (extra_processors or []):
            text = p.process(text)
        return text
