"""GatedEmbedModel — semaphore-gated OpenAI embedding API call.

Single method: ``embed(text) -> list[float]``.  Everything else
(exists, upsert, search) is the caller's job via EmbeddingRepository.
"""

from __future__ import annotations

import logging
import time

import openai

from chatty.configs.system import EmbeddingConfig
from chatty.core.service.metrics import (
    EMBEDDING_CALLS_IN_FLIGHT,
    EMBEDDING_INPUT_TOKENS,
    EMBEDDING_LATENCY_SECONDS,
)
from chatty.infra.concurrency.semaphore import ModelSemaphore
from chatty.infra.telemetry import (
    ATTR_EMBEDDING_MODEL,
    ATTR_EMBEDDING_TEXT_LEN,
    SPAN_EMBEDDING_EMBED,
    tracer,
)
from chatty.infra.tokens import estimate_tokens, truncate_to_tokens

logger = logging.getLogger(__name__)


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
        """Return the embedding vector for *text* (gated).

        Input is truncated to ``max_input_tokens`` before the API call.
        """
        text = truncate_to_tokens(text, self._config.max_input_tokens)
        EMBEDDING_INPUT_TOKENS.labels(model_name=self._config.model_name).observe(
            estimate_tokens(text)
        )
        with tracer.start_as_current_span(SPAN_EMBEDDING_EMBED) as span:
            span.set_attribute(ATTR_EMBEDDING_MODEL, self._config.model_name)
            span.set_attribute(ATTR_EMBEDDING_TEXT_LEN, len(text))
            logger.debug(
                "Embedding text (model=%s, len=%d)",
                self._config.model_name,
                len(text),
            )
            start = time.monotonic()
            async with self._semaphore.slot():
                EMBEDDING_CALLS_IN_FLIGHT.labels(
                    model_name=self._config.model_name
                ).inc()
                try:
                    response = await self._openai.embeddings.create(
                        input=text,
                        model=self._config.model_name,
                    )
                finally:
                    EMBEDDING_CALLS_IN_FLIGHT.labels(
                        model_name=self._config.model_name
                    ).dec()
            EMBEDDING_LATENCY_SECONDS.labels(operation="embed").observe(
                time.monotonic() - start
            )
            embedding = response.data[0].embedding
            # Some local model servers (e.g. vLLM) wrap the vector in an
            # extra list, returning [[…]] instead of the standard flat [… ].
            if embedding and isinstance(embedding[0], list):
                embedding = embedding[0]
            return embedding
