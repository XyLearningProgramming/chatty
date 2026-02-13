"""Base chat service with shared concerns.

Provides metrics tracking (via ``observe_stream_response``), PG callback
creation, and persona prompt building so that each concrete service only
implements ``_stream_response``.
"""

from abc import abstractmethod
from typing import AsyncGenerator, Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, HumanMessage

from chatty.configs.config import AppConfig
from chatty.infra.db.callback import PGMessageCallback

from .metrics import observe_stream_response
from .models import (
    ChatContext,
    ChatService,
    LANGGRAPH_CONFIG_KEY_CALLBACKS,
    LANGGRAPH_INPUT_KEY_MESSAGES,
    LANGGRAPH_STREAM_MODE_MESSAGES,
    StreamEvent,
)
from .stream import map_langgraph_stream


class BaseChatService(ChatService):
    """Concrete base with shared concerns.

    Subclasses set ``chat_service_name`` and implement ``_stream_response``.
    ``stream_response`` wraps ``_stream_response`` with Prometheus metrics
    via the existing ``observe_stream_response`` decorator.
    """

    chat_service_name: str = ""

    def __init__(self, llm: BaseLanguageModel, config: AppConfig) -> None:
        self._llm = llm
        self._config = config
        self._system_prompt = config.persona.build_system_prompt(
            config.prompt.system_prompt
        )

    # ------------------------------------------------------------------
    # PG callback helper
    # ------------------------------------------------------------------

    def _create_pg_callback(self, ctx: ChatContext) -> PGMessageCallback:
        """Create a per-request PG callback for message recording."""
        return PGMessageCallback(
            conversation_id=ctx.conversation_id,
            trace_id=ctx.trace_id,
            model_name=getattr(self._llm, "model_name", None),
        )

    # ------------------------------------------------------------------
    # Public streaming interface (metrics applied via decorator)
    # ------------------------------------------------------------------

    async def stream_response(
        self, ctx: ChatContext
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream domain events, wrapping ``_stream_response`` with metrics.
        
        Metrics are automatically applied via the ``observe_stream_response``
        decorator using ``self.chat_service_name``.
        """
        # Apply metrics decorator with service name at runtime
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
        """Yield domain events for a user query.

        Metrics and PG callback creation are handled by the base class.
        Subclasses focus on the core agent / retrieval logic.
        """
        ...


class GraphChatService(BaseChatService):
    """Base class for LangGraph-based chat services.

    Automatically handles:
    - PG callback binding
    - Graph execution via astream
    - Stream mapping to domain events

    Subclasses only need to implement ``_create_graph`` to customize the graph.
    """

    @abstractmethod
    async def _create_graph(self, ctx: ChatContext) -> Any:
        """Create the LangGraph graph for this request.

        Args:
            ctx: Per-request chat context.

        Returns:
            A compiled LangGraph graph (e.g., from ``create_agent``).

        Note:
            This method is async to allow subclasses to perform async operations
            (e.g., retrieval) before building the graph.
        """
        ...

    def _prepare_graph_input(self, ctx: ChatContext) -> dict[str, list[BaseMessage]]:
        """Prepare the input dictionary for graph.astream.

        Args:
            ctx: Per-request chat context.

        Returns:
            Input dict with messages key containing history + current query.
        """
        messages: list[BaseMessage] = list(ctx.history) + [
            HumanMessage(content=ctx.query)
        ]
        return {LANGGRAPH_INPUT_KEY_MESSAGES: messages}

    async def _stream_response(
        self, ctx: ChatContext
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream domain events by executing the graph automatically.

        This method handles:
        - Creating the PG callback
        - Creating the graph (via ``_create_graph``)
        - Executing astream with proper config
        - Mapping the stream to domain events
        """
        pg_callback = self._create_pg_callback(ctx)
        graph = await self._create_graph(ctx)
        graph_input = self._prepare_graph_input(ctx)
        graph_config = {LANGGRAPH_CONFIG_KEY_CALLBACKS: [pg_callback]}

        raw_stream = graph.astream(
            graph_input,
            stream_mode=LANGGRAPH_STREAM_MODE_MESSAGES,
            config=graph_config,
        )

        async for event in map_langgraph_stream(raw_stream):
            yield event
