"""Source embeddings DB access: repository + low-level helpers.

``EmbeddingRepository`` wraps session lifecycle and exposes
all_existing_texts, search, upsert.  All SQL and implementation
details are internal.
"""

from __future__ import annotations

import logging
import time

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from chatty.infra.telemetry import (
    ATTR_EMBEDDING_RESULT_COUNT,
    ATTR_EMBEDDING_THRESHOLD,
    ATTR_EMBEDDING_TOP_K,
    SPAN_EMBEDDING_SEARCH,
    tracer,
)

from .models import SourceEmbedding

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (no magic strings below)
# ---------------------------------------------------------------------------

TABLE_SOURCE_EMBEDDINGS = "source_embeddings"
COL_SOURCE_ID = "source_id"
COL_TEXT = "text"
COL_MODEL_NAME = "model_name"
COL_EMBEDDING = "embedding"

INDEX_ELEMENTS = (COL_SOURCE_ID, COL_TEXT, COL_MODEL_NAME)

PARAM_QUERY_VEC = "query_vec"
PARAM_MODEL_NAME = "model_name"
PARAM_THRESHOLD = "threshold"
PARAM_LIMIT = "limit"

SQL_SEARCH = f"""
    SELECT {COL_SOURCE_ID}, similarity
    FROM (
        SELECT DISTINCT ON ({COL_SOURCE_ID})
            {COL_SOURCE_ID},
            1 - ({COL_EMBEDDING} <=> :{PARAM_QUERY_VEC}::vector) AS similarity,
            {COL_EMBEDDING} <=> :{PARAM_QUERY_VEC}::vector AS distance
        FROM {TABLE_SOURCE_EMBEDDINGS}
        WHERE {COL_MODEL_NAME} = :{PARAM_MODEL_NAME}
          AND 1 - ({COL_EMBEDDING} <=> :{PARAM_QUERY_VEC}::vector) >= :{PARAM_THRESHOLD}
        ORDER BY {COL_SOURCE_ID}, distance
    ) AS best_per_source
    ORDER BY similarity DESC
    LIMIT :{PARAM_LIMIT}
"""


async def all_existing_texts(
    session: AsyncSession, model_name: str
) -> set[tuple[str, str]]:
    """Return all (source_id, text) pairs already embedded for *model_name*."""
    rows = await session.execute(
        select(SourceEmbedding.source_id, SourceEmbedding.text).where(
            SourceEmbedding.model_name == model_name,
        )
    )
    return {(r.source_id, r.text) for r in rows.all()}


async def search(
    session: AsyncSession,
    query_embedding: list[float],
    model_name: str,
    similarity_threshold: float,
    top_k: int,
) -> list[tuple[str, float]]:
    """Search for similar source embeddings (pgvector cosine).

    Returns (source_id, similarity) tuples sorted by similarity, highest first.
    """
    query_vec_str = "[" + ",".join(str(float(x)) for x in query_embedding) + "]"
    result = await session.execute(
        text(SQL_SEARCH),
        {
            PARAM_QUERY_VEC: query_vec_str,
            PARAM_MODEL_NAME: model_name,
            PARAM_THRESHOLD: similarity_threshold,
            PARAM_LIMIT: top_k,
        },
    )
    return [(row.source_id, float(row.similarity)) for row in result.all()]


async def upsert(
    session: AsyncSession,
    source_id: str,
    hint_text: str,
    embedding: list[float],
    model_name: str,
) -> None:
    """Insert or update the embedding row for (source_id, text, model_name)."""
    stmt = (
        pg_insert(SourceEmbedding)
        .values(
            source_id=source_id,
            text=hint_text,
            embedding=embedding,
            model_name=model_name,
        )
        .on_conflict_do_update(
            index_elements=list(INDEX_ELEMENTS),
            set_={COL_EMBEDDING: embedding},
        )
    )
    await session.execute(stmt)


# ---------------------------------------------------------------------------
# Repository — wraps session_factory and exposes exists / search / upsert
# ---------------------------------------------------------------------------


class EmbeddingRepository:
    """Async repository for source_embeddings.

    Hides session lifecycle and SQL; callers use high-level methods only.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._session_factory = session_factory

    async def all_existing_texts(self, model_name: str) -> set[tuple[str, str]]:
        """Return all (source_id, text) pairs already embedded for *model_name*."""
        async with self._session_factory() as session:
            return await all_existing_texts(session, model_name)

    async def search(
        self,
        query_embedding: list[float],
        model_name: str,
        similarity_threshold: float,
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Search similar embeddings; returns (source_id, similarity) tuples."""
        from chatty.core.service.metrics import EMBEDDING_LATENCY_SECONDS

        with tracer.start_as_current_span(SPAN_EMBEDDING_SEARCH) as span:
            span.set_attribute(ATTR_EMBEDDING_TOP_K, top_k)
            span.set_attribute(ATTR_EMBEDDING_THRESHOLD, similarity_threshold)
            start = time.monotonic()
            async with self._session_factory() as session:
                results = await search(
                    session,
                    query_embedding,
                    model_name,
                    similarity_threshold,
                    top_k,
                )
            EMBEDDING_LATENCY_SECONDS.labels(operation="search").observe(
                time.monotonic() - start
            )
            span.set_attribute(ATTR_EMBEDDING_RESULT_COUNT, len(results))
            logger.debug(
                "Embedding search: %d results (top_k=%d, threshold=%.2f)",
                len(results),
                top_k,
                similarity_threshold,
            )
            return results

    async def upsert(
        self,
        source_id: str,
        hint_text: str,
        embedding: list[float],
        model_name: str,
    ) -> None:
        """Insert or update the embedding row; commits the session."""
        async with self._session_factory() as session:
            await upsert(session, source_id, hint_text, embedding, model_name)
            await session.commit()
