"""RAG chat service -- LangGraph StateGraph with semantic caching.

Pipeline nodes:
    embed_query → cache_check ─┬─ (hit)  → record_cached → END
                               └─ (miss) → classify_query
                                   ─┬─ (simple) → generate → END  (no_think)
                                    └─ (complex) → retrieve_topk → build_rag_prompt
                                                  → generate → END

Every query is embedded up front so the vector is always persisted
(including trivial no-think shortcuts and cache misses).  The cache
is checked next; hits short-circuit the whole pipeline.  On a miss
the classify node decides whether the query is trivial (use the
no-think model directly) or complex (full RAG retrieval).

First-turn queries (no conversation history) are eligible for
semantic caching.  The cache is backed by the ``query_embedding``
column on ``chat_messages``: a similar past human message is found
and its corresponding AI response is returned.  TTL is checked at
read time against ``created_at`` using ``CacheConfig.ttl``.

Embedding is piped through the normal callback write path via
``query_to_human_message`` (no separate UPDATE needed).  Cache hits
are recorded as regular human + AI message pairs with
``extra.cache_hit = true``.
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
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from chatty.configs.config import AppConfig
from chatty.core.embedding.gated import GatedEmbedModel
from chatty.infra.db.cache import CacheRepository
from chatty.infra.db.callback import PGMessageCallback
from chatty.infra.db.converters import (
    cached_response_to_ai_message,
    query_to_human_message,
)
from chatty.infra.db.embedding import EmbeddingRepository
from chatty.infra.db.history import ChatMessageHistoryFactory
from chatty.infra.http_utils import HttpClient
from chatty.infra.telemetry import (
    ATTR_RAG_QUERY_LEN,
    ATTR_RAG_RESULT_COUNT,
    ATTR_RAG_THRESHOLD,
    ATTR_RAG_TOP_K,
    SPAN_RAG_PIPELINE,
    SPAN_RAG_RETRIEVE,
    tracer,
)
from chatty.infra.tokens import estimate_tokens

from .metrics import (
    RAG_CACHE_LOOKUPS_TOTAL,
    RAG_RETRIEVAL_LATENCY_SECONDS,
    RAG_SOURCES_RETURNED,
)
from .models import ChatContext, ChatService, ContentEvent, StreamEvent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node / edge / stream-mode constants  (avoid magic strings)
# ---------------------------------------------------------------------------

NODE_CLASSIFY = "classify_query"
NODE_EMBED_QUERY = "embed_query"
NODE_CACHE_CHECK = "cache_check"
NODE_RECORD_CACHED = "record_cached"
NODE_RETRIEVE_TOPK = "retrieve_topk"
NODE_BUILD_RAG_PROMPT = "build_rag_prompt"
NODE_GENERATE = "generate"

ROUTE_SIMPLE = "simple"
ROUTE_COMPLEX = "complex"
ROUTE_HIT = "hit"
ROUTE_MISS = "miss"

CACHE_RESULT_SKIP = "skip"
CACHE_RESULT_HIT = "hit"
CACHE_RESULT_MISS = "miss"

STREAM_MODE_MESSAGES = "messages"
STREAM_MODE_UPDATES = "updates"

# State field keys
KEY_QUERY = "query"
KEY_HISTORY = "history"
KEY_CONVERSATION_ID = "conversation_id"
KEY_QUERY_EMBEDDING = "query_embedding"
KEY_IS_FIRST_TURN = "is_first_turn"
KEY_CACHE_HIT = "cache_hit"
KEY_TOP_RESULTS = "top_results"
KEY_ENRICHED_PROMPT = "enriched_prompt"
KEY_RESPONSE_TEXT = "response_text"
KEY_SKIP_THINKING = "skip_thinking"

# ---------------------------------------------------------------------------
# Prompt constants
# ---------------------------------------------------------------------------
PROMPT_RAG_LINE_BREAK = "\n\n"

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
    skip_thinking: bool

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
        llm_no_think: BaseLanguageModel,
        config: AppConfig,
        embedder: GatedEmbedModel,
        embedding_repository: EmbeddingRepository,
        history_factory: ChatMessageHistoryFactory,
        cache_repository: CacheRepository,
    ) -> None:
        self._llm = llm
        self._llm_no_think = llm_no_think
        self._config = config
        self._embedder = embedder
        self._embedding_repository = embedding_repository
        self._history_factory = history_factory
        self._cache_repository = cache_repository
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

        builder.add_node(NODE_CLASSIFY, self._classify_query_node)
        builder.add_node(NODE_EMBED_QUERY, self._embed_query_node)
        builder.add_node(NODE_CACHE_CHECK, self._cache_check_node)
        builder.add_node(NODE_RECORD_CACHED, self._record_cached_node)
        builder.add_node(NODE_RETRIEVE_TOPK, self._retrieve_topk_node)
        builder.add_node(NODE_BUILD_RAG_PROMPT, self._build_rag_prompt_node)
        builder.add_node(NODE_GENERATE, self._generate_node)

        builder.add_edge(START, NODE_EMBED_QUERY)
        builder.add_edge(NODE_EMBED_QUERY, NODE_CACHE_CHECK)
        builder.add_conditional_edges(
            NODE_CACHE_CHECK,
            self._route_after_cache,
            {ROUTE_HIT: NODE_RECORD_CACHED, ROUTE_MISS: NODE_CLASSIFY},
        )
        builder.add_edge(NODE_RECORD_CACHED, END)
        builder.add_conditional_edges(
            NODE_CLASSIFY,
            self._route_after_classify,
            {ROUTE_SIMPLE: NODE_GENERATE, ROUTE_COMPLEX: NODE_RETRIEVE_TOPK},
        )
        builder.add_edge(NODE_RETRIEVE_TOPK, NODE_BUILD_RAG_PROMPT)
        builder.add_edge(NODE_BUILD_RAG_PROMPT, NODE_GENERATE)
        builder.add_edge(NODE_GENERATE, END)

        return builder.compile()

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    @staticmethod
    def _route_after_cache(state: RagState) -> str:
        return ROUTE_HIT if state.get(KEY_CACHE_HIT) else ROUTE_MISS

    @staticmethod
    def _route_after_classify(state: RagState) -> str:
        return ROUTE_SIMPLE if state.get(KEY_SKIP_THINKING) else ROUTE_COMPLEX

    # ------------------------------------------------------------------
    # Node: classify_query
    # ------------------------------------------------------------------

    def _classify_query_node(self, state: RagState) -> dict:
        """Classify the query as trivial or complex.

        Trivial first-turn queries (short greetings, etc.) skip the full
        RAG pipeline and go straight to generate via the no-think model.
        """
        chat_cfg = self._config.chat
        if not chat_cfg.rag_no_think_enabled:
            return {KEY_SKIP_THINKING: False}

        is_first_turn = state.get(KEY_IS_FIRST_TURN, False)
        query_len = len(state[KEY_QUERY].strip())
        is_trivial = is_first_turn and query_len <= chat_cfg.rag_no_think_max_chars

        if is_trivial:
            logger.debug(
                "classify_query: trivial (%d chars, first_turn=%s) — skipping RAG",
                query_len,
                is_first_turn,
            )
            return {
                KEY_SKIP_THINKING: True,
                KEY_ENRICHED_PROMPT: self._system_prompt,
            }
        return {KEY_SKIP_THINKING: False}

    # ------------------------------------------------------------------
    # Node: embed_query
    # ------------------------------------------------------------------

    async def _embed_query_node(self, state: RagState) -> dict:
        query_emb = await self._embedder.embed(state[KEY_QUERY])
        return {
            KEY_QUERY_EMBEDDING: query_emb,
        }

    # ------------------------------------------------------------------
    # Node: cache_check
    # ------------------------------------------------------------------

    async def _cache_check_node(self, state: RagState) -> dict:
        if not self._cache_config.enabled or not state.get(KEY_IS_FIRST_TURN):
            RAG_CACHE_LOOKUPS_TOTAL.labels(result=CACHE_RESULT_SKIP).inc()
            return {KEY_CACHE_HIT: None}

        cached = await self._cache_repository.search(
            query_embedding=state[KEY_QUERY_EMBEDDING],
            similarity_threshold=self._cache_config.similarity_threshold,
            ttl=self._cache_config.ttl,
        )

        is_hit = cached is not None
        RAG_CACHE_LOOKUPS_TOTAL.labels(
            result=CACHE_RESULT_HIT if is_hit else CACHE_RESULT_MISS
        ).inc()
        return {KEY_CACHE_HIT: cached}

    # ------------------------------------------------------------------
    # Node: record_cached  (persist query + cached response to DB)
    # ------------------------------------------------------------------

    async def _record_cached_node(self, state: RagState) -> dict:
        """Record the cached query/response pair in chat_messages.

        Uses the converter pairs so the human message carries the
        embedding and the AI message is marked ``cache_hit = true``.
        """
        human = query_to_human_message(
            state[KEY_QUERY],
            embedding=state.get(KEY_QUERY_EMBEDDING),
        )
        ai = cached_response_to_ai_message(
            state[KEY_CACHE_HIT],
            model_name=self._config.llm.model_name,
        )
        try:
            await self._current_history.aadd_messages([human, ai])
        except Exception:
            logger.warning("Failed to record cached messages", exc_info=True)
        return {}

    # ------------------------------------------------------------------
    # Node: retrieve_topk
    # ------------------------------------------------------------------

    async def _retrieve_topk_node(self, state: RagState) -> dict:
        with tracer.start_as_current_span(SPAN_RAG_RETRIEVE) as span:
            span.set_attribute(
                ATTR_RAG_THRESHOLD, self._rag_config.similarity_threshold
            )
            span.set_attribute(ATTR_RAG_TOP_K, self._rag_config.top_k)

            start = time.monotonic()
            results = await self._embedding_repository.search(
                state[KEY_QUERY_EMBEDDING],
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
                    logger.error(
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
            return {KEY_TOP_RESULTS: top_results}

    # ------------------------------------------------------------------
    # Node: build_rag_prompt  (assemble RAG context into system prompt)
    # ------------------------------------------------------------------

    def _build_rag_prompt_node(self, state: RagState) -> dict:
        prompt_cfg = self._config.prompt
        llm_cfg = self._config.llm
        input_budget = llm_cfg.context_window - llm_cfg.max_tokens

        base_cost = estimate_tokens(self._system_prompt)
        query_cost = estimate_tokens(state.get(KEY_QUERY, ""))
        remaining = input_budget - base_cost - query_cost

        context_parts: list[str] = []
        for source_id, content, sim in state.get(KEY_TOP_RESULTS, []):
            section = prompt_cfg.render_rag_context_section(
                source_id=source_id,
                similarity=sim,
                content=content,
            )
            section_cost = estimate_tokens(section)
            if section_cost > remaining:
                logger.debug(
                    "RAG budget full — dropping source '%s' "
                    "(need %d tokens, %d remaining)",
                    source_id,
                    section_cost,
                    remaining,
                )
                continue
            context_parts.append(section)
            remaining -= section_cost

        enriched = prompt_cfg.render_rag_prompt(
            base=self._system_prompt,
            content=PROMPT_RAG_LINE_BREAK.join(context_parts),
        )
        return {KEY_ENRICHED_PROMPT: enriched}

    # ------------------------------------------------------------------
    # Node: generate  (LLM call — tokens streamed via stream_mode)
    # ------------------------------------------------------------------

    async def _generate_node(self, state: RagState, config: RunnableConfig) -> dict:
        human = query_to_human_message(
            state[KEY_QUERY], embedding=state.get(KEY_QUERY_EMBEDDING)
        )

        messages: list[BaseMessage] = [
            SystemMessage(content=state[KEY_ENRICHED_PROMPT]),
            *state.get(KEY_HISTORY, []),
            human,
        ]
        llm = self._llm_no_think if state.get(KEY_SKIP_THINKING) else self._llm
        response = await llm.ainvoke(messages, config=config)
        return {KEY_RESPONSE_TEXT: response.content or ""}

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def stream_response(
        self, ctx: ChatContext
    ) -> AsyncGenerator[StreamEvent, None]:
        """Execute the RAG graph and yield domain events.

        Uses dual ``stream_mode=["messages", "updates"]``:
        - ``"messages"`` delivers LLM token chunks from the generate node.
        - ``"updates"`` delivers the record_cached result so cached
          responses can be emitted without an LLM call.
        """
        with tracer.start_as_current_span(SPAN_RAG_PIPELINE) as span:
            span.set_attribute(ATTR_RAG_QUERY_LEN, len(ctx.query))

            history = self._history_factory(
                ctx.conversation_id,
                trace_id=ctx.trace_id,
            )
            self._current_history = history
            pg_callback = PGMessageCallback(
                history=history,
                model_name=self._config.llm.model_name,
            )

            graph_input: RagState = {
                KEY_QUERY: ctx.query,
                KEY_HISTORY: list(ctx.history),
                KEY_CONVERSATION_ID: ctx.conversation_id,
                KEY_IS_FIRST_TURN: not ctx.history,
            }

            async for mode, data in self._graph.astream(
                graph_input,
                stream_mode=[STREAM_MODE_MESSAGES, STREAM_MODE_UPDATES],
                config={"callbacks": [pg_callback]},
            ):
                if mode == STREAM_MODE_MESSAGES:
                    chunk, _metadata = data
                    if isinstance(chunk, AIMessageChunk) and chunk.content:
                        yield ContentEvent(content=chunk.content)

                elif mode == STREAM_MODE_UPDATES:
                    if not isinstance(data, dict):
                        continue
                    cache_update = data.get(NODE_CACHE_CHECK)
                    if cache_update and cache_update.get(KEY_CACHE_HIT):
                        yield ContentEvent(content=cache_update[KEY_CACHE_HIT])
