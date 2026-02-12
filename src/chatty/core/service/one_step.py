"""Chat service using LangGraph agent (via langchain.agents)."""

from typing import AsyncGenerator

from langchain.agents import create_agent
from langchain_core.language_models import BaseLanguageModel

from chatty.configs.config import AppConfig
from chatty.infra.db.callback import PGMessageCallback

from .metrics import observe_stream_response
from .models import ChatContext, ChatService, StreamEvent
from .prompt import (
    PERSONA_CHARACTER_DEFAULT,
    PERSONA_EXPERTISE_DEFAULT,
    SYSTEM_PROMPT,
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
    ):
        """Store references for per-request agent creation."""
        self._llm = llm
        self._tools_registry = tools_registry

        persona = config.persona
        persona_character = (
            ", ".join(persona.character)
            if persona.character
            else PERSONA_CHARACTER_DEFAULT
        )
        persona_expertise = (
            ", ".join(persona.expertise)
            if persona.expertise
            else PERSONA_EXPERTISE_DEFAULT
        )

        self._system_prompt = SYSTEM_PROMPT.format(
            persona_name=persona.name,
            persona_character=persona_character,
            persona_expertise=persona_expertise,
        )

    @observe_stream_response(chat_service_name)
    async def stream_response(
        self, ctx: ChatContext
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream domain events for a user question.

        The LangGraph agent is created fresh each request so that the
        latest tool definitions from YAML files (including ConfigMap)
        are always used.

        A ``PGMessageCallback`` is attached to every run so that each
        LangChain message (system, human, AI, tool) is recorded to
        PostgreSQL as a fire-and-forget INSERT.
        """
        pg_callback = PGMessageCallback(
            conversation_id=ctx.conversation_id,
            trace_id=ctx.trace_id,
            model_name=getattr(self._llm, "model_name", None),
        )
        graph = create_agent(
            model=self._llm,
            tools=self._tools_registry.get_tools(),
            system_prompt=self._system_prompt,
        )
        raw_stream = graph.astream(
            {"messages": ctx.history + [("user", ctx.query)]},
            stream_mode="messages",
            config={"callbacks": [pg_callback]},
        )
        async for event in map_langgraph_stream(raw_stream):
            yield event
