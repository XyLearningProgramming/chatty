"""RAG chat service -- retrieval-augmented generation without tool calls.

The model receives an enriched system prompt that includes the top-K
most relevant knowledge sections (by cosine similarity to the user
query). Uses a simple LangGraph agent without tools.

Graceful degradation: sections whose embeddings have not yet been
computed by the background cron are silently skipped.
"""

import logging

from langchain.agents import create_agent
from langchain_core.language_models import BaseLanguageModel

from chatty.configs.config import AppConfig
from chatty.configs.persona import KnowledgeSection
from chatty.core.embedding.client import EmbeddingClient
from chatty.core.embedding.cron import _section_text
from chatty.core.embedding.repository import text_hash

from .base import GraphChatService
from .models import ChatContext

logger = logging.getLogger(__name__)


class RagChatService(GraphChatService):
    """RAG chat service: embed query, retrieve top-K, enrich prompt.

    Uses a simple LangGraph agent (without tools) with an enriched system
    prompt that includes retrieved context. The graph is created after
    retrieval so the prompt can be built with the retrieved sections.

    Flow:
    1. Embed the user query (blocking semaphore acquire).
    2. Search for similar section embeddings using pgvector (no semaphore).
    3. Map results back to KnowledgeSection objects.
    4. Build an enriched system prompt with retrieved context.
    5. Create a simple agent graph (no tools) with the enriched prompt.
    6. Execute the graph (handled by base class).
    """

    chat_service_name = "rag"

    def __init__(
        self,
        llm: BaseLanguageModel,
        config: AppConfig,
        embedding_client: EmbeddingClient,
    ) -> None:
        super().__init__(llm, config)
        self._embedding_client = embedding_client
        self._rag_config = config.rag

    # ------------------------------------------------------------------
    # Top-K retrieval
    # ------------------------------------------------------------------

    async def _retrieve_top_k(
        self,
        query_emb: list[float],
        sections: list[KnowledgeSection],
    ) -> list[tuple[KnowledgeSection, str, float]]:
        """Return (section, text, similarity) tuples sorted by score.

        Uses pgvector similarity search in the database. Sections without
        cached embeddings are silently skipped.
        """
        # Build mapping from text_hash to (section, text) for lookup
        section_map: dict[str, tuple[KnowledgeSection, str]] = {}
        text_hashes: list[str] = []
        for section in sections:
            text = _section_text(section)
            if text is None:
                continue
            hash_value = text_hash(text)
            section_map[hash_value] = (section, text)
            text_hashes.append(hash_value)

        if not text_hashes:
            return []

        # Search for similar embeddings using pgvector
        results = await self._embedding_client.search_similar(
            query_embedding=query_emb,
            similarity_threshold=self._rag_config.similarity_threshold,
            top_k=self._rag_config.top_k,
            text_hashes=text_hashes,
        )

        # Map results back to KnowledgeSection objects
        top_sections: list[tuple[KnowledgeSection, str, float]] = []
        for text_hash_val, text_content, similarity in results:
            if text_hash_val in section_map:
                section, _ = section_map[text_hash_val]
                top_sections.append((section, text_content, similarity))

        return top_sections

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_rag_prompt(
        self,
        top_sections: list[tuple[KnowledgeSection, str, float]],
    ) -> str:
        """Build an enriched system prompt with retrieved context."""
        base = self._system_prompt

        if not top_sections:
            return base

        context_parts: list[str] = []
        for section, text, sim in top_sections:
            context_parts.append(
                f"### {section.title} (relevance: {sim:.2f})\n{text}"
            )

        context_block = "\n\n".join(context_parts)
        return (
            f"{base}\n\n"
            "Below is relevant context retrieved from your knowledge base. "
            "Use it to inform your answer:\n\n"
            f"{context_block}"
        )

    # ------------------------------------------------------------------
    # Graph creation (with RAG prompt enrichment)
    # ------------------------------------------------------------------

    async def _create_graph(self, ctx: ChatContext):
        """Create a simple LangGraph agent (no tools) with enriched RAG prompt.

        Retrieves top-K sections, builds enriched prompt, then creates
        the graph with that prompt.
        """
        # 1. Embed query (blocking acquire -- races with LLM)
        query_emb = await self._embedding_client.embed(ctx.query)

        # 2. Search for similar section embeddings using pgvector (read-only, no semaphore)
        sections = self._config.persona.sections
        top_sections = await self._retrieve_top_k(query_emb, sections)

        # 3. Build enriched system prompt with retrieved context
        enriched_prompt = self._build_rag_prompt(top_sections)

        # 4. Create simple agent graph (no tools) with enriched prompt
        return create_agent(
            model=self._llm,
            tools=[],  # No tools for RAG
            system_prompt=enriched_prompt,
        )
