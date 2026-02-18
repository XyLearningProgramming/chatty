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
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from chatty.configs.config import AppConfig
from chatty.configs.persona import EmbedDeclaration
from chatty.core.embedding.client import EmbeddingClient
from chatty.core.embedding.cron import _match_hints_text, _source_text
from chatty.core.embedding.repository import text_hash

from .base import GraphChatService
from .models import ChatContext

logger = logging.getLogger(__name__)


class RagChatService(GraphChatService):
    """RAG chat service: embed query, retrieve top-K, enrich prompt.

    Flow:
    1. Embed the user query (blocking semaphore acquire).
    2. Search for similar match_hints embeddings using pgvector.
    3. Map results back to source ids and retrieve full content.
    4. Build an enriched system prompt with retrieved context.
    5. Create a simple agent graph (no tools) with the enriched prompt.
    """

    chat_service_name = "rag"

    def __init__(
        self,
        llm: BaseLanguageModel,
        config: AppConfig,
        embedding_client: EmbeddingClient,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        super().__init__(llm, config, session_factory)
        self._embedding_client = embedding_client
        self._rag_config = config.rag

    # ------------------------------------------------------------------
    # Top-K retrieval (match_hints based)
    # ------------------------------------------------------------------

    async def _retrieve_top_k(
        self,
        query_emb: list[float],
        embed_decls: list[EmbedDeclaration],
    ) -> list[tuple[str, str, float]]:
        """Return (source_id, full_content, similarity) tuples.

        Matches the user query embedding against match_hints
        embeddings in pgvector, then resolves the source_id to
        full content for prompt injection.
        """
        sources = self._config.persona.sources

        # Build mapping from hints_hash -> (source_id, full_content)
        hints_map: dict[str, tuple[str, str]] = {}
        text_hashes: list[str] = []

        for decl in embed_decls:
            source = sources.get(decl.source)
            if source is None:
                continue

            hints_text = _match_hints_text(decl)
            if not hints_text:
                continue

            full_content = _source_text(decl.source, source)
            if full_content is None:
                continue

            hash_value = text_hash(hints_text)
            hints_map[hash_value] = (decl.source, full_content)
            text_hashes.append(hash_value)

        if not text_hashes:
            return []

        results = await self._embedding_client.search_similar(
            query_embedding=query_emb,
            similarity_threshold=self._rag_config.similarity_threshold,
            top_k=self._rag_config.top_k,
            text_hashes=text_hashes,
        )

        top_results: list[tuple[str, str, float]] = []
        for hints_hash, _text_content, similarity in results:
            if hints_hash in hints_map:
                source_id, full_content = hints_map[hints_hash]
                top_results.append(
                    (source_id, full_content, similarity)
                )

        return top_results

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_rag_prompt(
        self,
        top_results: list[tuple[str, str, float]],
    ) -> str:
        """Build an enriched system prompt with retrieved context."""
        base = self._system_prompt

        if not top_results:
            return base

        context_parts: list[str] = []
        for source_id, content, sim in top_results:
            context_parts.append(
                f"### {source_id} (relevance: {sim:.2f})\n{content}"
            )

        context_block = "\n\n".join(context_parts)
        return (
            f"{base}\n\n"
            "Below is relevant context retrieved from your "
            "knowledge base. Use it to inform your answer:\n\n"
            f"{context_block}"
        )

    # ------------------------------------------------------------------
    # Graph creation
    # ------------------------------------------------------------------

    async def _create_graph(self, ctx: ChatContext):
        """Create a simple LangGraph agent with enriched RAG prompt."""
        query_emb = await self._embedding_client.embed(ctx.query)

        embed_decls = self._config.persona.embed
        top_results = await self._retrieve_top_k(
            query_emb, embed_decls
        )

        enriched_prompt = self._build_rag_prompt(top_results)

        return create_agent(
            model=self._llm,
            tools=[],
            system_prompt=enriched_prompt,
        )
