"""Base chat service with shared concerns.

Provides metrics tracking (via ``observe_stream_response``), PG callback
creation, and persona prompt building so that each concrete service only
implements ``_stream_response``.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import AsyncGenerator, Callable
from typing import Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, HumanMessage

from chatty.configs.config import AppConfig
from chatty.infra.db.callback import PGMessageCallback

from .metrics import observe_stream_response
from .models import (
    LANGGRAPH_CONFIG_KEY_CALLBACKS,
    LANGGRAPH_INPUT_KEY_MESSAGES,
    LANGGRAPH_STREAM_MODE_MESSAGES,
    ChatContext,
    ChatService,
    StreamEvent,
)
from .stream import map_langgraph_stream

PgCallbackFactory = Callable[[str, str, str | None], PGMessageCallback]


class BaseChatService(ChatService):
    """Concrete base with shared concerns.

    Subclasses set ``chat_service_name`` and implement
    ``_stream_response``.  ``stream_response`` wraps it with
    Prometheus metrics via ``observe_stream_response``.
    """

    chat_service_name: str = ""

    def __init__(
        self,
        llm: BaseLanguageModel,
        config: AppConfig,
        pg_callback_factory: PgCallbackFactory,
    ) -> None:
        self._llm = llm
        self._config = config
        self._pg_callback_factory = pg_callback_factory
        self._system_prompt = config.prompt.render_system_prompt(config.persona)

    # ------------------------------------------------------------------
    # PG callback helper
    # ------------------------------------------------------------------

    def _create_pg_callback(
        self, ctx: ChatContext
    ) -> PGMessageCallback:
        """Create a per-request PG callback for message recording."""
        return self._pg_callback_factory(
            ctx.conversation_id,
            ctx.trace_id,
            getattr(self._llm, "model_name", None),
        )

    # ------------------------------------------------------------------
    # Public streaming interface (metrics applied via decorator)
    # ------------------------------------------------------------------

    async def stream_response(
        self, ctx: ChatContext
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream domain events, wrapping with metrics."""
        decorated = observe_stream_response(self.chat_service_name)(
            self._stream_response
        )
        async for event in decorated(ctx):
            yield event

    # ------------------------------------------------------------------
    # Abstract: subclasses implement the actual streaming logic
    # ------------------------------------------------------------------

    @abstractmethod
    async def _stream_response(
        self, ctx: ChatContext
    ) -> AsyncGenerator[StreamEvent, None]:
        """Yield domain events for a user query."""
        ...


class GraphChatService(BaseChatService):
    """Base class for LangGraph-based chat services.

    Automatically handles:
    - PG callback binding
    - Graph execution via astream
    - Stream mapping to domain events

    Subclasses only need to implement ``_create_graph``.
    """

    @abstractmethod
    async def _create_graph(self, ctx: ChatContext) -> Any:
        """Create the LangGraph graph for this request."""
        ...

    def _prepare_graph_input(
        self, ctx: ChatContext
    ) -> dict[str, list[BaseMessage]]:
        """Prepare the input dictionary for graph.astream."""
        messages: list[BaseMessage] = list(ctx.history) + [
            HumanMessage(content=ctx.query)
        ]
        return {LANGGRAPH_INPUT_KEY_MESSAGES: messages}

    async def _stream_response(
        self, ctx: ChatContext
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream domain events by executing the graph."""
        pg_callback = self._create_pg_callback(ctx)
        graph = await self._create_graph(ctx)
        graph_input = self._prepare_graph_input(ctx)
        graph_config = {
            LANGGRAPH_CONFIG_KEY_CALLBACKS: [pg_callback]
        }

        raw_stream = graph.astream(
            graph_input,
            stream_mode=LANGGRAPH_STREAM_MODE_MESSAGES,
            config=graph_config,
        )

        async for event in map_langgraph_stream(raw_stream):
            yield event
