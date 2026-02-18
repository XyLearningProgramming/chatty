"""RAG chat service -- retrieval-augmented generation without tool calls.

The model receives an enriched system prompt that includes the top-K
most relevant knowledge sources (by cosine similarity of user query
against ``match_hints`` embeddings).  Uses a simple LangGraph agent
without tools.

Graceful degradation: sources whose match_hints embeddings have not
yet been computed by the background cron are silently skipped.
"""

from __future__ import annotations

import logging

from langchain.agents import create_agent
from langchain_core.language_models import BaseLanguageModel

from chatty.configs.config import AppConfig
from chatty.core.embedding.gated import GatedEmbedModel
from chatty.infra.db.embedding import EmbeddingRepository
from chatty.infra.http_utils import HttpClient

from .base import GraphChatService, PgCallbackFactory
from .models import ChatContext

logger = logging.getLogger(__name__)


class RagChatService(GraphChatService):
    """RAG chat service: embed query, retrieve top-K, enrich prompt.

    Flow:
    1. Embed the user query (blocking semaphore acquire).
    2. Single pgvector search returns (source_id, similarity) tuples.
    3. Resolve each source_id to full content via persona config.
    4. Build an enriched system prompt with retrieved context.
    5. Create a simple agent graph (no tools) with the enriched prompt.
    """

    chat_service_name = "rag"

    def __init__(
        self,
        llm: BaseLanguageModel,
        config: AppConfig,
        embedder: GatedEmbedModel,
        embedding_repository: EmbeddingRepository,
        pg_callback_factory: PgCallbackFactory,
    ) -> None:
        super().__init__(llm, config, pg_callback_factory)
        self._embedder = embedder
        self._embedding_repository = embedding_repository
        self._rag_config = config.rag
        config.prompt.render_rag_prompt(base="", content="")

    # ------------------------------------------------------------------
    # Top-K retrieval
    # ------------------------------------------------------------------

    async def _retrieve_top_k(
        self,
        query_emb: list[float],
    ) -> list[tuple[str, str, float]]:
        """Return (source_id, full_content, similarity) tuples.

        Content is resolved ad hoc per source (inline or URL fetch)
        with source-level + embed-level processors applied.
        """
        persona = self._config.persona
        sources = persona.sources
        embed_by_source = {d.source: d for d in persona.embed}

        results = await self._embedding_repository.search(
            query_emb,
            self._embedder.model_name,
            self._rag_config.similarity_threshold,
            self._rag_config.top_k,
        )

        top_results: list[tuple[str, str, float]] = []
        for source_id, similarity in results:
            source = sources.get(source_id)
            if source is None:
                continue
            try:
                decl = embed_by_source.get(source_id)
                content = await source.get_content(
                    HttpClient.get,
                    extra_processors=decl.get_processors() if decl else None,
                )
            except Exception:
                logger.warning(
                    "Failed to resolve content for source '%s'",
                    source_id,
                    exc_info=True,
                )
                continue
            top_results.append((source_id, content, similarity))

        return top_results

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_rag_prompt(
        self,
        top_results: list[tuple[str, str, float]],
    ) -> str:
        """Render the RAG system prompt template with base + content."""
        context_parts: list[str] = []
        for source_id, content, sim in top_results:
            context_parts.append(
                f"### {source_id} (relevance: {sim:.2f})\n{content}"
            )

        return self._config.prompt.render_rag_prompt(
            base=self._system_prompt,
            content="\n\n".join(context_parts),
        )

    # ------------------------------------------------------------------
    # Graph creation
    # ------------------------------------------------------------------

    async def _create_graph(self, ctx: ChatContext):
        """Create a simple LangGraph agent with enriched RAG prompt."""
        query_emb = await self._embedder.embed(ctx.query)

        top_results = await self._retrieve_top_k(query_emb)

        enriched_prompt = self._build_rag_prompt(top_results)

        return create_agent(
            model=self._llm,
            tools=[],
            system_prompt=enriched_prompt,
        )
