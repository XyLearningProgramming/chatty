"""Chat service with ReAct agent implementation."""

from typing import AsyncGenerator

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel

from chatty.configs.config import AppConfig

from .models import (
    ChatService,
    ServiceStreamEvent,
)
from .prompt import (
    PERSONA_CHARACTER_DEFAULT,
    PERSONA_EXPERTISE_DEFAULT,
    REACT_PROMPT_ONE_STEP,
)
from .tools.registry import ToolRegistry

# LangChain constants
LANGCHAIN_ASTREAM_VERSION = "v2"


class OneStepChatService(ChatService):
    """Async chat service powered by ReAct agent with streaming support."""

    chat_service_name = "one_step"

    def __init__(
        self,
        llm: BaseLanguageModel,
        tools_registry: ToolRegistry,
        config: AppConfig,
    ):
        """Initialize chat service with configuration."""
        # Format the prompt with persona information
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

        formatted_prompt = REACT_PROMPT_ONE_STEP.format(
            persona_name=persona.name,
            persona_character=persona_character,
            persona_expertise=persona_expertise,
            tools="{tools}",
            tool_names="{tool_names}",
            input="{input}",
            agent_scratchpad="{agent_scratchpad}",
        )

        # Create ReAct agent
        self.get_agent_executor = lambda: AgentExecutor(
            name=self.chat_service_name,
            agent=create_react_agent(
                llm=llm,
                tools=tools_registry.get_tools(),
                prompt=PromptTemplate.from_template(formatted_prompt),
            ),
            tools=tools_registry.get_tools(),
            verbose=True,
            max_iterations=3,
            early_stopping_method="force",
            # Allow fallbacks of parsing error .
            handle_parsing_errors=False,
        )

    async def stream_response(
        self, question: str
    ) -> AsyncGenerator["ServiceStreamEvent", None]:
        """Stream response tokens and structured data with real-time pattern detection.

        Args:
            question: The user's question

        Yields:
            SSE-compatible events: token events, structured data events, and end-of-stream

        Raises:
            asyncio.CancelledError: When the operation is cancelled
        """
        # Use StreamProcessor to handle LangChain events
        langchain_events = self.get_agent_executor().astream_events(
            {"input": question},
            config={"run_name": self.chat_service_name},
            version=LANGCHAIN_ASTREAM_VERSION,
        )
        async for event in langchain_events:
            # Yield unmodified structured data events for now
            yield event
            # # 1) Plain LLM tokens
            # if isinstance(evt, LLMResult) and evt.generations:
            #     # each generation has .text for the new token(s)
            #     for gen in evt.generations:
            #         # you may need to diff against previous to get only the new chunk
            #         yield ServiceTokenEvent(content=gen.text)

            # # 2) Function/tool calls
            # elif isinstance(evt, AgentAction):
            #     # if the agent is invoking your “emit_structured” tool:
            #     if evt.tool == "emit_structured":
            #         # evt.log or evt.tool_input is the JSON string/obj
            #         # parse it and emit a structured event
            #         data = evt.tool_input
            #         if isinstance(data, str):
            #             # sometimes it’s a JSON‐string
            #             data = json.loads(data)
            #         yield ServiceStructuredDataEvent(data=data)
            #     else:
            #         # you could also surface other tool results here,
            #         # or ignore them if you treat them as “internal”
            #         pass

            # # 3) Final finish
            # elif isinstance(evt, AgentFinish):
            #     yield ServiceEndOfStreamEvent()
            #     return
