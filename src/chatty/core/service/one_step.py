"""Chat service using LangGraph agent (via langchain.agents)."""

from typing import AsyncGenerator

from langchain.agents import create_agent
from langchain_core.language_models import BaseLanguageModel

from chatty.configs.config import AppConfig

from .metrics import observe_stream_response
from .models import ChatService, StreamEvent
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
        self, question: str
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream domain events for a user question.

        The LangGraph agent is created fresh each request so that the
        latest tool definitions from YAML files (including ConfigMap)
        are always used.
        """
        graph = create_agent(
            model=self._llm,
            tools=self._tools_registry.get_tools(),
            system_prompt=self._system_prompt,
        )
        raw_stream = graph.astream(
            {"messages": [("user", question)]},
            stream_mode="messages",
        )
        async for event in map_langgraph_stream(raw_stream):
            yield event
