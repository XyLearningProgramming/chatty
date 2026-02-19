"""Semantic response cache backed by the chat_messages table.

First-turn human messages can have their ``query_embedding`` stamped.
A subsequent similar query finds the original human message and returns
the corresponding AI response via a lateral join.

TTL is enforced at read time against ``created_at``.
"""

from __future__ import annotations

import logging
from datetime import timedelta

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

SQL_SEARCH_CACHED_RESPONSE = """
    SELECT ai.content AS response_text, human.similarity
    FROM (
        SELECT conversation_id, created_at,
               1 - (query_embedding <=> :query_vec::vector) AS similarity
        FROM chat_messages
        WHERE role = 'human'
          AND query_embedding IS NOT NULL
          AND created_at >= now() - :ttl_interval::interval
          AND 1 - (query_embedding <=> :query_vec::vector) >= :threshold
        ORDER BY query_embedding <=> :query_vec::vector
        LIMIT 1
    ) AS human
    JOIN LATERAL (
        SELECT content
        FROM chat_messages
        WHERE conversation_id = human.conversation_id
          AND role = 'ai'
          AND created_at > human.created_at
        ORDER BY created_at
        LIMIT 1
    ) AS ai ON true
"""

SQL_STAMP_EMBEDDING = """
    UPDATE chat_messages
    SET query_embedding = :embedding::vector
    WHERE conversation_id = :conversation_id
      AND role = 'human'
      AND query_embedding IS NULL
    ORDER BY created_at DESC
    LIMIT 1
"""


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


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
    query_vec_str = (
        "[" + ",".join(str(float(x)) for x in query_embedding) + "]"
    )
    ttl_interval = f"{int(ttl.total_seconds())} seconds"

    result = await session.execute(
        text(SQL_SEARCH_CACHED_RESPONSE),
        {
            "query_vec": query_vec_str,
            "threshold": similarity_threshold,
            "ttl_interval": ttl_interval,
        },
    )
    row = result.first()
    if row is None:
        return None

    logger.info("Cache hit (similarity=%.3f)", float(row.similarity))
    return str(row.response_text)


async def stamp_query_embedding(
    session: AsyncSession,
    conversation_id: str,
    query_embedding: list[float],
) -> None:
    """Stamp the query embedding onto the most recent human message
    in the given conversation (for future cache lookups).
    """
    embedding_str = (
        "[" + ",".join(str(float(x)) for x in query_embedding) + "]"
    )
    await session.execute(
        text(SQL_STAMP_EMBEDDING),
        {
            "embedding": embedding_str,
            "conversation_id": conversation_id,
        },
    )
    await session.commit()
