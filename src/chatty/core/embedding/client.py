"""EmbeddingClient -- semaphore-gated, DB-cached OpenAI embeddings."""

import logging
from typing import Sequence

import openai

from chatty.configs.system import EmbeddingConfig
from chatty.infra.concurrency.semaphore import ModelSemaphore

from .repository import EmbeddingRepository, text_hash

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """OpenAI-compatible embedding client with DB cache and semaphore.

    Public API
    ----------
    ``embed(text)``
        Hot-path: acquires the semaphore (blocking), checks DB cache,
        calls the API on miss, stores the result.

    ``try_embed(text)``
        Cron-path: uses ``semaphore.try_slot()`` (instant timeout),
        returns ``None`` when busy.

    ``get_cached(text)``
        Read-only DB lookup (no API call, no semaphore).

    ``get_cached_batch(texts)``
        Batch read-only DB lookup.
    """

    def __init__(
        self,
        config: EmbeddingConfig,
        repository: EmbeddingRepository,
        semaphore: ModelSemaphore,
    ) -> None:
        self._config = config
        self._repo = repository
        self._semaphore = semaphore
        self._openai = openai.AsyncOpenAI(
            base_url=config.endpoint,
            api_key=config.api_key or "unused",
        )

    # ------------------------------------------------------------------
    # Hot path (blocking semaphore)
    # ------------------------------------------------------------------

    async def embed(self, text: str) -> list[float]:
        """Return the embedding for *text*, using DB cache first.

        Acquires the model semaphore (blocking) if an API call is needed.
        """
        cached = await self._repo.get_by_hash(
            text_hash(text), self._config.model_name
        )
        if cached is not None:
            return cached

        async with self._semaphore.slot():
            return await self._call_and_store(text)

    # ------------------------------------------------------------------
    # Cron path (non-blocking)
    # ------------------------------------------------------------------

    async def try_embed(self, text: str) -> list[float] | None:
        """Try to embed *text* without blocking.

        Returns the embedding on success, ``None`` if the semaphore is
        busy or if the text is already cached.
        """
        cached = await self._repo.get_by_hash(
            text_hash(text), self._config.model_name
        )
        if cached is not None:
            return cached

        async with self._semaphore.try_slot() as acquired:
            if not acquired:
                return None
            return await self._call_and_store(text)

    # ------------------------------------------------------------------
    # Read-only DB lookups
    # ------------------------------------------------------------------

    async def get_cached(self, text: str) -> list[float] | None:
        """Return the cached embedding or ``None`` (no API call)."""
        return await self._repo.get_by_hash(
            text_hash(text), self._config.model_name
        )

    async def get_cached_batch(
        self, texts: Sequence[str]
    ) -> dict[str, list[float] | None]:
        """Return ``{text: embedding | None}`` for each input text.

        Only performs a DB lookup -- never calls the embedding API.
        """
        hashes = {text_hash(t): t for t in texts}
        found = await self._repo.get_batch(
            list(hashes.keys()), self._config.model_name
        )
        return {
            original_text: found.get(h)
            for h, original_text in hashes.items()
        }

    async def search_similar(
        self,
        query_embedding: list[float],
        similarity_threshold: float,
        top_k: int,
        text_hashes: list[str] | None = None,
    ) -> list[tuple[str, str, float]]:
        """Search for similar embeddings using pgvector.

        Returns a list of (text_hash, text_content, similarity) tuples,
        sorted by similarity (highest first).

        Args:
            query_embedding: The query vector to search for.
            similarity_threshold: Minimum similarity score (0-1).
            top_k: Maximum number of results to return.
            text_hashes: Optional list of text_hashes to restrict search to.

        Returns:
            List of (text_hash, text_content, similarity) tuples.
        """
        return await self._repo.search_similar(
            query_embedding=query_embedding,
            model_name=self._config.model_name,
            similarity_threshold=similarity_threshold,
            top_k=top_k,
            text_hashes=text_hashes,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _call_and_store(self, text: str) -> list[float]:
        """Call the embedding API and persist the result."""
        response = await self._openai.embeddings.create(
            input=text,
            model=self._config.model_name,
        )
        embedding = response.data[0].embedding
        await self._repo.store(text, embedding, self._config.model_name)
        return embedding
