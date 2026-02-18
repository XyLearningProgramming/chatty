"""Background cron that embeds persona source match_hints.

Each tick:

1. Reads ``persona.embed`` declarations from the (hot-reloadable) config.
2. For each declaration whose ``source_id`` is not yet in the DB,
   embeds the ``match_hints`` text (gated) and upserts the vector.

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
from chatty.configs.persona import EmbedDeclaration
from chatty.infra.concurrency.base import AcquireTimeout
from chatty.infra.concurrency.semaphore import build_semaphore
from chatty.infra.db.embedding import EmbeddingRepository
from chatty.infra.db.engine import build_db
from chatty.infra.lifespan import get_app

from .gated import GatedEmbedModel

logger = logging.getLogger(__name__)


def _match_hints_text(decl: EmbedDeclaration) -> str:
    """Build the text to embed from match_hints."""
    return "\n".join(decl.match_hints)


# ------------------------------------------------------------------
# EmbeddingCron
# ------------------------------------------------------------------


class EmbeddingCron:
    """Embeds match_hints for persona sources that are not yet in the DB.

    Uses ``EmbeddingRepository`` for DB ops and ``GatedEmbedModel``
    only for the gated embed call.
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
        config = get_app_config()
        embed_decls = config.persona.embed
        model_name = self.embedder.model_name

        for decl in embed_decls:
            if await self.repository.exists(decl.source, model_name):
                continue

            hints_text = _match_hints_text(decl)
            if not hints_text:
                continue

            try:
                vec = await self.embedder.embed(hints_text)
                await self.repository.upsert(
                    decl.source, vec, model_name
                )
                logger.info(
                    "Cron: embedded match_hints for source '%s'",
                    decl.source,
                )
            except AcquireTimeout:
                logger.debug(
                    "Cron: semaphore busy, skipping source '%s'",
                    decl.source,
                )
            except Exception:
                logger.warning(
                    "Cron: failed to embed/upsert source '%s'",
                    decl.source,
                    exc_info=True,
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
