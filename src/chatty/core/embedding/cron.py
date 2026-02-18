"""Background cron that opportunistically embeds persona sources.

Each tick:

1. Reads ``persona.embed`` declarations from the (hot-reloadable) config.
2. For URL sources, fetches and caches the resolved content (applying
   source-level + embed-level processors).
3. For each embed declaration, builds ``match_hints`` text and checks
   the DB for missing embeddings.
4. For each missing embedding, races for the ``ModelSemaphore`` via
   ``try_embed`` (instant-timeout).  If the slot is busy, the entry
   is skipped until the next tick.

``build_cron`` is a lifespan dependency that creates, starts,
and stops the cron automatically via ``yield``.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI

from chatty.configs.config import AppConfig, get_app_config
from chatty.configs.persona import EmbedDeclaration, KnowledgeSource
from chatty.core.service.tools.processors import resolve_processors
from chatty.core.service.tools.url_tool import (
    _PDF_CONTENT_TYPES,
    _extract_text_from_pdf,
)
from chatty.infra.concurrency.semaphore import build_semaphore
from chatty.infra.db.engine import build_db
from chatty.infra.lifespan import get_app

from .client import EmbeddingClient
from .repository import EmbeddingRepository, text_hash

logger = logging.getLogger(__name__)

_resolved_content: dict[str, str] = {}
"""In-memory cache of fully processed content keyed by source id."""


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


async def _fetch_source_content(
    source_id: str,
    source: KnowledgeSource,
    embed_decl: EmbedDeclaration,
) -> str | None:
    """Fetch content from a source's ``content_url``.

    Applies source-level processors then embed-level processors.
    Returns ``None`` on failure (logged, not raised).
    """
    url = source.content_url
    timeout_seconds = source.timeout.total_seconds()

    try:
        async with httpx.AsyncClient(
            timeout=timeout_seconds
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

            ct = response.headers.get("content-type", "")
            if any(pdf in ct for pdf in _PDF_CONTENT_TYPES):
                content = _extract_text_from_pdf(response.content)
            else:
                content = response.text

            # Source-level processors
            if source.processors:
                for proc in resolve_processors(source.processors):
                    content = proc.process(content)

            # Embed action-level processors
            if embed_decl.processors:
                for proc in resolve_processors(embed_decl.processors):
                    content = proc.process(content)

            return content
    except Exception:
        logger.warning(
            "Cron: failed to fetch content for source '%s' "
            "from %s",
            source_id,
            url,
            exc_info=True,
        )
        return None


def _source_text(
    source_id: str,
    source: KnowledgeSource,
) -> str | None:
    """Return the embeddable text for a source (resolved or inline)."""
    if source.content_url:
        return _resolved_content.get(source_id)
    return source.content or None


def _match_hints_text(decl: EmbedDeclaration) -> str:
    """Build the text to embed for query matching from match_hints."""
    return "\n".join(decl.match_hints)


# ------------------------------------------------------------------
# EmbeddingCron
# ------------------------------------------------------------------


class EmbeddingCron:
    """Manages the embedding background loop lifecycle.

    Owns the ``EmbeddingClient`` (exposed via ``.client`` for DI)
    and an ``asyncio.Task`` that ticks at a configurable interval.
    """

    def __init__(
        self,
        client: EmbeddingClient,
        interval: int,
    ) -> None:
        self.client = client
        self._interval = interval
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(
            self._loop(), name="embedding-cron"
        )
        logger.info(
            "Embedding cron started (interval=%ds)",
            self._interval,
        )

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        logger.info("Embedding cron stopped.")

    # -- internal ----------------------------------------------------

    async def _loop(self) -> None:
        while True:
            try:
                await self._tick()
            except asyncio.CancelledError:
                logger.info(
                    "Embedding cron cancelled, shutting down."
                )
                return
            except Exception:
                logger.exception("Embedding cron tick failed")
            await asyncio.sleep(self._interval)

    async def _tick(self) -> None:
        config = get_app_config()
        sources = config.persona.sources
        embed_decls = config.persona.embed

        # Step 1: Resolve URL content for embed sources
        for decl in embed_decls:
            source = sources.get(decl.source)
            if source is None:
                continue
            if not source.content_url:
                continue
            if decl.source in _resolved_content:
                continue

            content = await _fetch_source_content(
                decl.source, source, decl
            )
            if content:
                _resolved_content[decl.source] = content
                logger.info(
                    "Cron: resolved content for source '%s' "
                    "(%d chars)",
                    decl.source,
                    len(content),
                )

        # Step 2: Find match_hints that need embedding
        hints_to_embed: list[tuple[EmbedDeclaration, str]] = []
        for decl in embed_decls:
            hints_text = _match_hints_text(decl)
            if not hints_text:
                continue
            cached = await self.client.get_cached(hints_text)
            if cached is None:
                hints_to_embed.append((decl, hints_text))

        if hints_to_embed:
            logger.info(
                "Cron: %d embed entry/entries need embedding",
                len(hints_to_embed),
            )

        # Step 3: Try to embed each missing hints text
        for decl, hints_text in hints_to_embed:
            result = await self.client.try_embed(hints_text)
            if result is not None:
                logger.info(
                    "Cron: embedded match_hints for source '%s' "
                    "(%d dims)",
                    decl.source,
                    len(result),
                )
            else:
                logger.debug(
                    "Cron: semaphore busy, skipping source '%s'",
                    decl.source,
                )


# ------------------------------------------------------------------
# Lifespan dependency
# ------------------------------------------------------------------


async def build_cron(
    app: Annotated[FastAPI, Depends(get_app)],
    config: Annotated[AppConfig, Depends(get_app_config)],
    _db: Annotated[None, Depends(build_db)],
    _sem: Annotated[None, Depends(build_semaphore)],
) -> AsyncGenerator[None, None]:
    """Create, start, and expose the embedding cron on ``app.state``."""
    repo = EmbeddingRepository(app.state.session_factory)
    client = EmbeddingClient(
        config=config.embedding,
        repository=repo,
        semaphore=app.state.semaphore,
    )
    cron = EmbeddingCron(
        client=client,
        interval=config.rag.cron_interval,
    )
    app.state.embedding_client = cron.client
    app.state.embedding_cron = cron
    await cron.start()
    yield
    await cron.stop()
