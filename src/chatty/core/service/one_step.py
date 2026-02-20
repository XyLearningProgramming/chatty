"""Chat service using LangGraph agent (via langchain.agents).

Self-contained implementation — no base class inheritance beyond the
abstract ``ChatService`` interface.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from langchain.agents import create_agent
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage

from chatty.configs.config import AppConfig

from .callback import PgCallbackFactory
from .models import (
    LANGGRAPH_CONFIG_KEY_CALLBACKS,
    LANGGRAPH_INPUT_KEY_MESSAGES,
    LANGGRAPH_STREAM_MODE_MESSAGES,
    ChatContext,
    ChatService,
    StreamEvent,
)
from .stream import map_langgraph_stream
from .tools.registry import ToolRegistry


class OneStepChatService(ChatService):
    """Chat service powered by LangGraph ReAct agent.

    The agent graph is rebuilt on every request so that hot-reloaded tool
    definitions from the ConfigMap are picked up immediately.
    """

    chat_service_name = "one_step"

    def __init__(
        self,
        llm: BaseLanguageModel,
        tools_registry: ToolRegistry,
        config: AppConfig,
        pg_callback_factory: PgCallbackFactory,
    ):
        self._llm = llm
        self._tools_registry = tools_registry
        self._config = config
        self._pg_callback_factory = pg_callback_factory
        self._system_prompt = config.prompt.render_system_prompt(config.persona)

    async def stream_response(
        self, ctx: ChatContext
    ) -> AsyncGenerator[StreamEvent, None]:
        """Execute the LangGraph agent and yield domain events."""
        pg_callback = self._pg_callback_factory(
            ctx.conversation_id,
            ctx.trace_id,
            self._config.llm.model_name,
        )

        graph = create_agent(
            model=self._llm,
            tools=self._tools_registry.get_tools(),
            system_prompt=self._system_prompt,
        )

        messages = list(ctx.history) + [HumanMessage(content=ctx.query)]
        raw_stream = graph.astream(
            {LANGGRAPH_INPUT_KEY_MESSAGES: messages},
            stream_mode=LANGGRAPH_STREAM_MODE_MESSAGES,
            config={LANGGRAPH_CONFIG_KEY_CALLBACKS: [pg_callback]},
        )

        async for event in map_langgraph_stream(raw_stream):
            yield event
