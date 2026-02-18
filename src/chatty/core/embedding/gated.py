"""GatedEmbedModel — semaphore-gated OpenAI embedding API call.

Single method: ``embed(text) -> list[float]``.  Everything else
(exists, upsert, search) is the caller's job via EmbeddingRepository.
"""

from __future__ import annotations

import openai

from chatty.configs.system import EmbeddingConfig
from chatty.infra.concurrency.semaphore import ModelSemaphore


class GatedEmbedModel:
    """Gates the embedding API call behind a semaphore."""

    def __init__(
        self,
        config: EmbeddingConfig,
        semaphore: ModelSemaphore,
    ) -> None:
        self._config = config
        self._semaphore = semaphore
        self._openai = openai.AsyncOpenAI(
            base_url=config.endpoint,
            api_key=config.api_key or "unused",
        )

    @property
    def model_name(self) -> str:
        return self._config.model_name

    async def embed(self, text: str) -> list[float]:
        """Return the embedding vector for *text* (gated)."""
        async with self._semaphore.slot():
            response = await self._openai.embeddings.create(
                input=text,
                model=self._config.model_name,
            )
            return response.data[0].embedding
