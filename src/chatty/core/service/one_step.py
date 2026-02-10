"""Chat service using LangGraph agent (via langchain.agents)."""

from typing import AsyncGenerator

from langchain.agents import create_agent
from langchain_core.language_models import BaseLanguageModel

from chatty.configs.config import AppConfig

from .models import ChatService, StreamEvent
from .prompt import (
    PERSONA_CHARACTER_DEFAULT,
    PERSONA_EXPERTISE_DEFAULT,
    SYSTEM_PROMPT,
)
from .stream import map_langgraph_stream
from .tools.registry import ToolRegistry


class OneStepChatService(ChatService):
    """Chat service powered by LangGraph ReAct agent."""

    chat_service_name = "one_step"

    def __init__(
        self,
        llm: BaseLanguageModel,
        tools_registry: ToolRegistry,
        config: AppConfig,
    ):
        """Initialize chat service with LangGraph agent."""
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

        system_prompt = SYSTEM_PROMPT.format(
            persona_name=persona.name,
            persona_character=persona_character,
            persona_expertise=persona_expertise,
        )

        # LangGraph agent with first-class system_prompt support.
        self._graph = create_agent(
            model=llm,
            tools=tools_registry.get_tools(),
            system_prompt=system_prompt,
        )

    async def stream_response(
        self, question: str
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream domain events for a user question.

        Uses ``stream_mode="messages"`` which yields
        ``(BaseMessageChunk, metadata)`` tuples that the stream mapper
        converts into domain ``StreamEvent`` instances.
        """
        raw_stream = self._graph.astream(
            {"messages": [("user", question)]},
            stream_mode="messages",
        )
        async for event in map_langgraph_stream(raw_stream):
            yield event
