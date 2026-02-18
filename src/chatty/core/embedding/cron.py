"""Background cron that embeds persona source match_hints.

Each tick:

1. Reads ``persona.embed`` declarations from the (hot-reloadable) config.
2. For each individual hint that is not yet in the DB, embeds it
   (gated) and upserts the vector.  Each hint gets its own row.

Content resolution is NOT this module's job — whoever needs a source's
content resolves it ad hoc at request time.

``build_cron`` is a lifespan dependency that creates, starts,
and stops the cron automatically via ``yield``.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends, FastAPI, Request

from chatty.configs.config import AppConfig, get_app_config
from chatty.core.service.metrics import (
    EMBEDDING_CRON_HINTS_TOTAL,
    EMBEDDING_CRON_RUNS_TOTAL,
)
from chatty.infra.concurrency.base import AcquireTimeout
from chatty.infra.concurrency.semaphore import build_semaphore
from chatty.infra.db.embedding import EmbeddingRepository
from chatty.infra.db.engine import build_db
from chatty.infra.lifespan import get_app
from chatty.infra.telemetry import (
    ATTR_CRON_EMBEDDED,
    ATTR_CRON_TOTAL_PENDING,
    SPAN_EMBEDDING_CRON_TICK,
    tracer,
)

from .gated import GatedEmbedModel

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# EmbeddingCron
# ------------------------------------------------------------------


class EmbeddingCron:
    """Embeds individual match_hints for persona sources.

    Each hint is embedded as a separate row.  On every tick the cron
    checks for any missing ``(source_id, text, model_name)`` tuples
    and fills them in.
    """

    def __init__(
        self,
        embedder: GatedEmbedModel,
        repository: EmbeddingRepository,
        interval: int,
    ) -> None:
        self.embedder = embedder
        self.repository = repository
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
        with tracer.start_as_current_span(SPAN_EMBEDDING_CRON_TICK) as span:
            config = get_app_config()
            model_name = self.embedder.model_name

            existing = await self.repository.all_existing_texts(model_name)

            embedded = 0
            total_pending = 0
            for decl in config.persona.embed:
                for hint in decl.match_hints:
                    if not hint or (decl.source, hint) in existing:
                        continue
                    total_pending += 1

                    try:
                        vec = await self.embedder.embed(hint)
                        await self.repository.upsert(
                            decl.source, hint, vec, model_name
                        )
                        embedded += 1
                        EMBEDDING_CRON_HINTS_TOTAL.inc()
                        logger.info(
                            "Cron: embedded hint '%s' for source '%s'",
                            hint,
                            decl.source,
                        )
                    except AcquireTimeout:
                        EMBEDDING_CRON_RUNS_TOTAL.labels(status="skipped").inc()
                        logger.debug(
                            "Cron: semaphore busy, skipping hint '%s'",
                            hint,
                        )
                    except Exception:
                        EMBEDDING_CRON_RUNS_TOTAL.labels(status="error").inc()
                        logger.warning(
                            "Cron: failed to embed hint '%s' for source '%s'",
                            hint,
                            decl.source,
                            exc_info=True,
                        )

            span.set_attribute(ATTR_CRON_EMBEDDED, embedded)
            span.set_attribute(ATTR_CRON_TOTAL_PENDING, total_pending)
            EMBEDDING_CRON_RUNS_TOTAL.labels(status="ok").inc()
            if total_pending > 0:
                logger.info(
                    "Cron tick: embedded %d/%d hints", embedded, total_pending
                )


# ------------------------------------------------------------------
# FastAPI dependencies — built by build_cron, accessed via app.state
# ------------------------------------------------------------------


def get_embedder(request: Request) -> GatedEmbedModel:
    """FastAPI dependency — reads from ``app.state``."""
    return request.app.state.embedder


def get_embedding_repository(request: Request) -> EmbeddingRepository:
    """FastAPI dependency — reads from ``app.state``."""
    return request.app.state.embedding_repository


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
    embedder = GatedEmbedModel(
        config=config.embedding,
        semaphore=app.state.semaphore,
    )
    cron = EmbeddingCron(
        embedder=embedder,
        repository=repo,
        interval=config.rag.cron_interval,
    )
    app.state.embedder = embedder
    app.state.embedding_repository = repo
    app.state.embedding_cron = cron
    await cron.start()
    yield
    await cron.stop()
