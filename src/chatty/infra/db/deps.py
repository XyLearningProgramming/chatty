"""Higher-level per-request dependency factories for the db package.

These factories need intra-package imports (history, embedding, cache)
and therefore live inside ``db/``.  The low-level engine + session
plumbing lives in the sibling leaf module ``chatty.infra.db_engine``
to avoid circular imports with ``telemetry``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from chatty.infra.db_engine import get_session_factory


def get_chat_message_history_factory(
    sf: Annotated[
        async_sessionmaker[AsyncSession],
        Depends(get_session_factory),
    ],
):
    """Return a factory that creates PgChatMessageHistory per conversation/trace."""
    from .history import PgChatMessageHistory

    def factory(
        conversation_id: str,
        trace_id: str | None = None,
        max_messages: int | None = None,
    ) -> PgChatMessageHistory:
        return PgChatMessageHistory(
            sf, conversation_id, trace_id=trace_id, max_messages=max_messages
        )

    return factory


def get_embedding_repository(
    sf: Annotated[
        async_sessionmaker[AsyncSession],
        Depends(get_session_factory),
    ],
):
    """Return the embedding repository (exists, search, upsert) for this app."""
    from .embedding import EmbeddingRepository

    return EmbeddingRepository(sf)


def get_cache_repository(
    sf: Annotated[
        async_sessionmaker[AsyncSession],
        Depends(get_session_factory),
    ],
):
    """Return the cache repository (search) for this app."""
    from .cache import CacheRepository

    return CacheRepository(sf)
