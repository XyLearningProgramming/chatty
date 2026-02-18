"""Source embeddings DB access: repository + low-level helpers.

``EmbeddingRepository`` wraps session lifecycle and exposes exists,
search, upsert. All SQL and implementation details are internal.
"""

from __future__ import annotations

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .models import SourceEmbedding

# ---------------------------------------------------------------------------
# Constants (no magic strings below)
# ---------------------------------------------------------------------------

TABLE_SOURCE_EMBEDDINGS = "source_embeddings"
COL_SOURCE_ID = "source_id"
COL_MODEL_NAME = "model_name"
COL_EMBEDDING = "embedding"

INDEX_ELEMENTS = (COL_SOURCE_ID, COL_MODEL_NAME)

PARAM_QUERY_VEC = "query_vec"
PARAM_MODEL_NAME = "model_name"
PARAM_THRESHOLD = "threshold"
PARAM_LIMIT = "limit"

SQL_SEARCH = f"""
    SELECT
        {COL_SOURCE_ID},
        1 - (embedding <=> :{PARAM_QUERY_VEC}::vector) AS similarity
    FROM {TABLE_SOURCE_EMBEDDINGS}
    WHERE {COL_MODEL_NAME} = :{PARAM_MODEL_NAME}
    AND 1 - (embedding <=> :{PARAM_QUERY_VEC}::vector) >= :{PARAM_THRESHOLD}
    ORDER BY embedding <=> :{PARAM_QUERY_VEC}::vector
    LIMIT :{PARAM_LIMIT}
"""


async def exists(
    session: AsyncSession, source_id: str, model_name: str
) -> bool:
    """Return whether an embedding row exists for (source_id, model_name)."""
    row = (
        await session.execute(
            select(SourceEmbedding.id).where(
                SourceEmbedding.source_id == source_id,
                SourceEmbedding.model_name == model_name,
            )
        )
    ).scalar_one_or_none()
    return row is not None


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
    query_vec_str = (
        "{" + ",".join(str(float(x)) for x in query_embedding) + "}"
    )
    result = await session.execute(
        text(SQL_SEARCH),
        {
            PARAM_QUERY_VEC: query_vec_str,
            PARAM_MODEL_NAME: model_name,
            PARAM_THRESHOLD: similarity_threshold,
            PARAM_LIMIT: top_k,
        },
    )
    return [
        (row.source_id, float(row.similarity))
        for row in result.all()
    ]


async def upsert(
    session: AsyncSession,
    source_id: str,
    embedding: list[float],
    model_name: str,
) -> None:
    """Insert or update the embedding row for (source_id, model_name)."""
    stmt = (
        pg_insert(SourceEmbedding)
        .values(
            source_id=source_id,
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
    """Async repository for source_embeddings: exists, search, upsert.

    Hides session lifecycle and SQL; callers use high-level methods only.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._session_factory = session_factory

    async def exists(self, source_id: str, model_name: str) -> bool:
        """Return whether an embedding row exists for (source_id, model_name)."""
        async with self._session_factory() as session:
            return await exists(session, source_id, model_name)

    async def search(
        self,
        query_embedding: list[float],
        model_name: str,
        similarity_threshold: float,
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Search similar embeddings; returns (source_id, similarity) tuples."""
        async with self._session_factory() as session:
            return await search(
                session,
                query_embedding,
                model_name,
                similarity_threshold,
                top_k,
            )

    async def upsert(
        self,
        source_id: str,
        embedding: list[float],
        model_name: str,
    ) -> None:
        """Insert or update the embedding row; commits the session."""
        async with self._session_factory() as session:
            await upsert(session, source_id, embedding, model_name)
            await session.commit()
