"""RAG chat service -- LangGraph StateGraph with semantic caching.

Pipeline nodes:
    embed_query  → cache_check ─┬─ (hit)  → return_cached → END
                                └─ (miss) → retrieve_topk → build_prompt
                                           → generate → cache_write → END

First-turn queries (no conversation history) are eligible for semantic
caching.  The cache is backed by the ``query_embedding`` column on
``chat_messages``: a similar past human message is found and its
corresponding AI response is returned.  TTL is checked at read time
against ``created_at`` using ``CacheConfig.ttl``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator
from typing import TypedDict

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from chatty.configs.config import AppConfig
from chatty.core.embedding.gated import GatedEmbedModel
from chatty.infra.db.cache import search_cached_response, stamp_query_embedding
from chatty.infra.db.embedding import EmbeddingRepository
from chatty.infra.http_utils import HttpClient
from chatty.infra.telemetry import (
    ATTR_RAG_CACHE_HIT,
    ATTR_RAG_QUERY_LEN,
    ATTR_RAG_RESULT_COUNT,
    ATTR_RAG_THRESHOLD,
    ATTR_RAG_TOP_K,
    SPAN_RAG_CACHE_CHECK,
    SPAN_RAG_CACHE_WRITE,
    SPAN_RAG_PIPELINE,
    SPAN_RAG_RETRIEVE,
    tracer,
)

from .callback import PgCallbackFactory
from .metrics import (
    RAG_CACHE_LOOKUPS_TOTAL,
    RAG_RETRIEVAL_LATENCY_SECONDS,
    RAG_SOURCES_RETURNED,
    observe_stream_response,
)
from .models import ChatContext, ChatService, ContentEvent, StreamEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class RagState(TypedDict, total=False):
    """Typed state threaded through every node in the RAG graph."""

    query: str
    history: list[BaseMessage]
    conversation_id: str

    query_embedding: list[float]
    is_first_turn: bool

    cache_hit: str | None

    top_results: list[tuple[str, str, float]]
    enriched_prompt: str

    response_text: str


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class RagChatService(ChatService):
    """RAG chat service with LangGraph StateGraph and semantic caching.

    The graph is compiled once at construction.  Node functions are bound
    methods so they have full access to repos, embedder, and config.
    """

    chat_service_name = "rag"

    def __init__(
        self,
        llm: BaseLanguageModel,
        config: AppConfig,
        embedder: GatedEmbedModel,
        embedding_repository: EmbeddingRepository,
        pg_callback_factory: PgCallbackFactory,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._llm = llm
        self._config = config
        self._embedder = embedder
        self._embedding_repository = embedding_repository
        self._pg_callback_factory = pg_callback_factory
        self._session_factory = session_factory
        self._rag_config = config.rag
        self._cache_config = config.cache
        self._system_prompt = config.prompt.render_system_prompt(config.persona)
        config.prompt.render_rag_prompt(base="", content="")

        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self):
        builder: StateGraph = StateGraph(RagState)

        builder.add_node("embed_query", self._embed_query_node)
        builder.add_node("cache_check", self._cache_check_node)
        builder.add_node("return_cached", self._return_cached_node)
        builder.add_node("retrieve_topk", self._retrieve_topk_node)
        builder.add_node("build_prompt", self._build_prompt_node)
        builder.add_node("generate", self._generate_node)
        builder.add_node("cache_write", self._cache_write_node)

        builder.add_edge(START, "embed_query")
        builder.add_edge("embed_query", "cache_check")
        builder.add_conditional_edges(
            "cache_check",
            self._route_after_cache,
            {"hit": "return_cached", "miss": "retrieve_topk"},
        )
        builder.add_edge("return_cached", END)
        builder.add_edge("retrieve_topk", "build_prompt")
        builder.add_edge("build_prompt", "generate")
        builder.add_edge("generate", "cache_write")
        builder.add_edge("cache_write", END)

        return builder.compile()

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    @staticmethod
    def _route_after_cache(state: RagState) -> str:
        return "hit" if state.get("cache_hit") else "miss"

    # ------------------------------------------------------------------
    # Node: embed_query
    # ------------------------------------------------------------------

    async def _embed_query_node(self, state: RagState) -> dict:
        query_emb = await self._embedder.embed(state["query"])
        return {
            "query_embedding": query_emb,
            "is_first_turn": len(state.get("history") or []) == 0,
        }

    # ------------------------------------------------------------------
    # Node: cache_check
    # ------------------------------------------------------------------

    async def _cache_check_node(self, state: RagState) -> dict:
        with tracer.start_as_current_span(SPAN_RAG_CACHE_CHECK) as span:
            if not self._cache_config.enabled or not state.get("is_first_turn"):
                span.set_attribute(ATTR_RAG_CACHE_HIT, False)
                RAG_CACHE_LOOKUPS_TOTAL.labels(result="skip").inc()
                return {"cache_hit": None}

            async with self._session_factory() as session:
                cached = await search_cached_response(
                    session,
                    query_embedding=state["query_embedding"],
                    similarity_threshold=self._cache_config.similarity_threshold,
                    ttl=self._cache_config.ttl,
                )

            is_hit = cached is not None
            span.set_attribute(ATTR_RAG_CACHE_HIT, is_hit)
            RAG_CACHE_LOOKUPS_TOTAL.labels(
                result="hit" if is_hit else "miss"
            ).inc()
            return {"cache_hit": cached}

    # ------------------------------------------------------------------
    # Node: return_cached  (pass-through; content delivered via updates)
    # ------------------------------------------------------------------

    @staticmethod
    async def _return_cached_node(state: RagState) -> dict:
        return {}

    # ------------------------------------------------------------------
    # Node: retrieve_topk
    # ------------------------------------------------------------------

    async def _retrieve_topk_node(self, state: RagState) -> dict:
        with tracer.start_as_current_span(SPAN_RAG_RETRIEVE) as span:
            span.set_attribute(ATTR_RAG_THRESHOLD, self._rag_config.similarity_threshold)
            span.set_attribute(ATTR_RAG_TOP_K, self._rag_config.top_k)

            start = time.monotonic()
            results = await self._embedding_repository.search(
                state["query_embedding"],
                self._embedder.model_name,
                self._rag_config.similarity_threshold,
                self._rag_config.top_k,
            )

            persona = self._config.persona
            sources = persona.sources
            embed_by_source = {d.source: d for d in persona.embed}

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

            elapsed = time.monotonic() - start
            RAG_RETRIEVAL_LATENCY_SECONDS.observe(elapsed)
            RAG_SOURCES_RETURNED.observe(len(top_results))
            span.set_attribute(ATTR_RAG_RESULT_COUNT, len(top_results))
            logger.info(
                "RAG: retrieved %d sources (threshold=%.2f)",
                len(top_results),
                self._rag_config.similarity_threshold,
            )
            return {"top_results": top_results}

    # ------------------------------------------------------------------
    # Node: build_prompt
    # ------------------------------------------------------------------

    def _build_prompt_node(self, state: RagState) -> dict:
        context_parts: list[str] = []
        for source_id, content, sim in state.get("top_results", []):
            context_parts.append(
                f"### {source_id} (relevance: {sim:.2f})\n{content}"
            )

        enriched = self._config.prompt.render_rag_prompt(
            base=self._system_prompt,
            content="\n\n".join(context_parts),
        )
        return {"enriched_prompt": enriched}

    # ------------------------------------------------------------------
    # Node: generate  (LLM call — tokens streamed via stream_mode)
    # ------------------------------------------------------------------

    async def _generate_node(
        self, state: RagState, config: RunnableConfig
    ) -> dict:
        messages: list[BaseMessage] = [
            SystemMessage(content=state["enriched_prompt"]),
            *state.get("history", []),
            HumanMessage(content=state["query"]),
        ]
        response = await self._llm.ainvoke(messages, config=config)
        return {"response_text": response.content or ""}

    # ------------------------------------------------------------------
    # Node: cache_write  (stamp embedding on the human message row)
    # ------------------------------------------------------------------

    async def _cache_write_node(self, state: RagState) -> dict:
        if not state.get("is_first_turn") or not self._cache_config.enabled:
            return {}

        with tracer.start_as_current_span(SPAN_RAG_CACHE_WRITE):
            try:
                async with self._session_factory() as session:
                    await stamp_query_embedding(
                        session,
                        conversation_id=state["conversation_id"],
                        query_embedding=state["query_embedding"],
                    )
            except Exception:
                logger.warning("Failed to stamp query embedding", exc_info=True)

        return {}

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def stream_response(
        self, ctx: ChatContext
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream domain events, wrapping with metrics."""
        decorated = observe_stream_response(self.chat_service_name)(
            self._stream
        )
        async for event in decorated(ctx):
            yield event

    async def _stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[StreamEvent, None]:
        """Execute the RAG graph and yield domain events.

        Uses dual ``stream_mode=["messages", "updates"]``:
        - ``"messages"`` delivers LLM token chunks from the generate node.
        - ``"updates"`` delivers the cache_check result so cached
          responses can be emitted without an LLM call.
        """
        with tracer.start_as_current_span(SPAN_RAG_PIPELINE) as span:
            span.set_attribute(ATTR_RAG_QUERY_LEN, len(ctx.query))

            pg_callback = self._pg_callback_factory(
                ctx.conversation_id,
                ctx.trace_id,
                getattr(self._llm, "model_name", None),
            )

            graph_input: RagState = {
                "query": ctx.query,
                "history": list(ctx.history),
                "conversation_id": ctx.conversation_id,
            }

            async for mode, data in self._graph.astream(
                graph_input,
                stream_mode=["messages", "updates"],
                config={"callbacks": [pg_callback]},
            ):
                if mode == "messages":
                    chunk, _metadata = data
                    if isinstance(chunk, AIMessageChunk) and chunk.content:
                        yield ContentEvent(content=chunk.content)

                elif mode == "updates":
                    if not isinstance(data, dict):
                        continue
                    cache_update = data.get("cache_check")
                    if cache_update and cache_update.get("cache_hit"):
                        yield ContentEvent(
                            content=cache_update["cache_hit"]
                        )
