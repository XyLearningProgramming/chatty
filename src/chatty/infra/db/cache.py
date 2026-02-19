"""Semantic response cache backed by the chat_messages table.

First-turn human messages carry a ``query_embedding`` (set via the
converter pair in ``converters.py``).  A subsequent similar query
finds the original human message and returns the corresponding AI
response via a lateral join.

TTL is enforced at read time against ``created_at``.

``CacheRepository`` wraps session lifecycle and exposes search.
"""

from __future__ import annotations

import logging
from datetime import timedelta

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from chatty.infra.telemetry import (
    ATTR_RAG_CACHE_HIT,
    SPAN_RAG_CACHE_CHECK,
    tracer,
)

from .models import ROLE_AI, ROLE_HUMAN

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (no magic strings below)
# ---------------------------------------------------------------------------

TABLE_CHAT_MESSAGES = "chat_messages"
COL_CONVERSATION_ID = "conversation_id"
COL_CREATED_AT = "created_at"
COL_ROLE = "role"
COL_CONTENT = "content"
COL_QUERY_EMBEDDING = "query_embedding"

PARAM_QUERY_VEC = "query_vec"
PARAM_THRESHOLD = "threshold"
PARAM_TTL_INTERVAL = "ttl_interval"

# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

SQL_SEARCH_CACHED_RESPONSE = f"""
    SELECT ai.{COL_CONTENT} AS response_text, human.similarity
    FROM (
        SELECT {COL_CONVERSATION_ID}, {COL_CREATED_AT},
               1 - ({COL_QUERY_EMBEDDING} <=> :{PARAM_QUERY_VEC}::vector) AS similarity
        FROM {TABLE_CHAT_MESSAGES}
        WHERE {COL_ROLE} = '{ROLE_HUMAN}'
          AND {COL_QUERY_EMBEDDING} IS NOT NULL
          AND {COL_CREATED_AT} >= now() - :{PARAM_TTL_INTERVAL}::interval
          AND 1 - ({COL_QUERY_EMBEDDING} <=> :{PARAM_QUERY_VEC}::vector) >= :{PARAM_THRESHOLD}
        ORDER BY {COL_QUERY_EMBEDDING} <=> :{PARAM_QUERY_VEC}::vector
        LIMIT 1
    ) AS human
    JOIN LATERAL (
        SELECT {COL_CONTENT}
        FROM {TABLE_CHAT_MESSAGES}
        WHERE {COL_CONVERSATION_ID} = human.{COL_CONVERSATION_ID}
          AND {COL_ROLE} = '{ROLE_AI}'
          AND {COL_CREATED_AT} > human.{COL_CREATED_AT}
        ORDER BY {COL_CREATED_AT}
        LIMIT 1
    ) AS ai ON true
"""


# ---------------------------------------------------------------------------
# Low-level function (session provided by caller)
# ---------------------------------------------------------------------------


def _vec_literal(embedding: list[float]) -> str:
    return "[" + ",".join(str(float(x)) for x in embedding) + "]"


async def search_cached_response(
    session: AsyncSession,
    query_embedding: list[float],
    similarity_threshold: float,
    ttl: timedelta,
) -> str | None:
    """Find a cached AI response for a semantically similar first-turn query.

    Returns the AI response text if a match is found within TTL,
    or ``None`` on cache miss.
    """
    result = await session.execute(
        text(SQL_SEARCH_CACHED_RESPONSE),
        {
            PARAM_QUERY_VEC: _vec_literal(query_embedding),
            PARAM_THRESHOLD: similarity_threshold,
            PARAM_TTL_INTERVAL: f"{int(ttl.total_seconds())} seconds",
        },
    )
    row = result.first()
    if row is None:
        return None

    logger.info("Cache hit (similarity=%.3f)", float(row.similarity))
    return str(row.response_text)


# ---------------------------------------------------------------------------
# Repository — wraps session_factory and exposes search
# ---------------------------------------------------------------------------


class CacheRepository:
    """Async repository for the semantic response cache.

    Hides session lifecycle and SQL; callers use high-level methods only.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._session_factory = session_factory

    async def search(
        self,
        query_embedding: list[float],
        similarity_threshold: float,
        ttl: timedelta,
    ) -> str | None:
        """Find a cached AI response for a semantically similar query.

        Returns the AI response text on cache hit, ``None`` on miss.
        """
        with tracer.start_as_current_span(SPAN_RAG_CACHE_CHECK) as span:
            async with self._session_factory() as session:
                cached = await search_cached_response(
                    session,
                    query_embedding,
                    similarity_threshold,
                    ttl,
                )
            is_hit = cached is not None
            span.set_attribute(ATTR_RAG_CACHE_HIT, is_hit)
            return cached
