"""Database helpers for the text_embeddings table."""

import hashlib
import logging

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import async_sessionmaker

from chatty.infra.db.models import TextEmbedding

logger = logging.getLogger(__name__)


def text_hash(text: str) -> str:
    """Return the SHA-256 hex digest of *text*."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingRepository:
    """Thin async wrapper around ``text_embeddings`` DB operations."""

    def __init__(self, session_factory: async_sessionmaker) -> None:
        self._session_factory = session_factory

    async def get_by_hash(
        self, hash_value: str, model_name: str
    ) -> list[float] | None:
        """Return the cached embedding vector or ``None``."""
        async with self._session_factory() as session:
            row = (
                await session.execute(
                    select(TextEmbedding.embedding).where(
                        TextEmbedding.text_hash == hash_value,
                        TextEmbedding.model_name == model_name,
                    )
                )
            ).scalar_one_or_none()
            return row  # type: ignore[return-value]

    async def store(
        self,
        text: str,
        embedding: list[float],
        model_name: str,
    ) -> None:
        """Insert an embedding (upsert -- ignore if duplicate)."""
        hash_value = text_hash(text)
        stmt = (
            pg_insert(TextEmbedding)
            .values(
                text_hash=hash_value,
                text_content=text,
                embedding=embedding,
                model_name=model_name,
            )
            .on_conflict_do_nothing(
                index_elements=["text_hash", "model_name"]
            )
        )
        try:
            async with self._session_factory() as session:
                await session.execute(stmt)
                await session.commit()
        except Exception:
            logger.warning(
                "Failed to store embedding for hash %s",
                hash_value,
                exc_info=True,
            )

    async def get_batch(
        self, hash_values: list[str], model_name: str
    ) -> dict[str, list[float]]:
        """Return a mapping ``{text_hash: embedding}`` for found rows."""
        if not hash_values:
            return {}
        async with self._session_factory() as session:
            rows = (
                await session.execute(
                    select(
                        TextEmbedding.text_hash,
                        TextEmbedding.embedding,
                    ).where(
                        TextEmbedding.text_hash.in_(hash_values),
                        TextEmbedding.model_name == model_name,
                    )
                )
            ).all()
            return {row.text_hash: row.embedding for row in rows}

    async def search_similar(
        self,
        query_embedding: list[float],
        model_name: str,
        similarity_threshold: float,
        top_k: int,
        text_hashes: list[str] | None = None,
    ) -> list[tuple[str, str, float]]:
        """Search for similar embeddings using pgvector cosine distance.

        Returns a list of (text_hash, text_content, similarity) tuples,
        sorted by similarity (highest first). Similarity is cosine similarity
        (1 - cosine_distance).

        Args:
            query_embedding: The query vector to search for.
            model_name: Filter by model name.
            similarity_threshold: Minimum similarity score (0-1).
            top_k: Maximum number of results to return.
            text_hashes: Optional list of text_hashes to restrict search to.
                If None, searches all embeddings for the model.

        Returns:
            List of (text_hash, text_content, similarity) tuples.
        """
        # Format query embedding as PostgreSQL array literal for vector casting
        # Use array format {0.1,0.2,...} which can be cast to vector
        query_vec_str = "{" + ",".join(str(float(x)) for x in query_embedding) + "}"

        # Build the SQL query using pgvector's <=> operator (cosine distance)
        # pgvector's <=> returns cosine distance (0 = identical, 2 = opposite)
        # Cosine similarity = 1 - (cosine_distance / 2)
        # So: similarity = 1 - (embedding <=> query) / 2
        # But actually, <=> returns 1 - cosine_similarity, so:
        # similarity = 1 - (embedding <=> query)
        if text_hashes:
            sql = text("""
                SELECT 
                    text_hash,
                    text_content,
                    1 - (embedding <=> :query_vec::vector) AS similarity
                FROM text_embeddings
                WHERE model_name = :model_name
                AND text_hash = ANY(:text_hashes)
                AND 1 - (embedding <=> :query_vec::vector) >= :threshold
                ORDER BY embedding <=> :query_vec::vector
                LIMIT :limit
            """)
            params = {
                "query_vec": query_vec_str,
                "model_name": model_name,
                "text_hashes": text_hashes,
                "threshold": similarity_threshold,
                "limit": top_k,
            }
        else:
            sql = text("""
                SELECT 
                    text_hash,
                    text_content,
                    1 - (embedding <=> :query_vec::vector) AS similarity
                FROM text_embeddings
                WHERE model_name = :model_name
                AND 1 - (embedding <=> :query_vec::vector) >= :threshold
                ORDER BY embedding <=> :query_vec::vector
                LIMIT :limit
            """)
            params = {
                "query_vec": query_vec_str,
                "model_name": model_name,
                "threshold": similarity_threshold,
                "limit": top_k,
            }

        async with self._session_factory() as session:
            result = await session.execute(sql, params)
            rows = result.all()
            return [(row.text_hash, row.text_content, float(row.similarity)) for row in rows]
