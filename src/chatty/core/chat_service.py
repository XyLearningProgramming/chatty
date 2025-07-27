"""Chat service with ReAct agent implementation."""

from typing import Annotated, Any, AsyncGenerator, Dict

from fastapi import Depends
from langchain.agents import AgentExecutor, create_react_agent
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel

from chatty.core.llm import get_llm
from chatty.infra import singleton

from .prompt import REACT_PROMPT
from .tools.registry import ToolRegistry, get_tool_registry


class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming agent responses."""

    def __init__(self):
        self.tokens = []
        self.current_step = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Handle new token from LLM."""
        self.tokens.append(token)

    def on_agent_action(self, action, **kwargs) -> None:
        """Handle agent action."""
        self.current_step = f"Action: {action.tool} - {action.tool_input}"

    def on_tool_end(self, output: str, **kwargs) -> None:
        """Handle tool completion."""
        self.current_step = f"Observation: {output[:100]}..."


class ChatService:
    """Async chat service powered by ReAct agent with streaming support."""

    def __init__(self, llm: BaseLanguageModel, tools_registry: ToolRegistry):
        """Initialize chat service with configuration."""
        # Create ReAct agent
        self.get_agent_executor = lambda: AgentExecutor(
            agent=create_react_agent(
                llm=llm,
                tools=tools_registry.get_tools(),
                prompt=PromptTemplate.from_template(REACT_PROMPT),
            ),
            tools=tools_registry.get_tools(),
            verbose=True,
            max_iterations=8,
            early_stopping_method="generate",
            handle_parsing_errors=True,
        )

        # Create agent executor

    async def process(self, question: str) -> Dict[str, Any]:
        """Process a question using the ReAct agent (async).

        Args:
            question: The user's question

        Returns:
            Response dictionary from the agent executor
        """
        response = await self.get_agent_executor().ainvoke({"input": question})
        return response

    async def stream_response(
        self, question: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream response tokens and structured data.

        Args:
            question: The user's question

        Yields:
            SSE-compatible events: token events and structured data events
        """
        callback_handler = StreamingCallbackHandler()

        try:
            # Process with streaming callback
            response = await self.get_agent_executor().ainvoke(
                {"input": question}, {"callbacks": [callback_handler]}
            )

            # Stream tokens as they come in
            for token in callback_handler.tokens:
                yield {"type": "token", "data": {"token": token}}

            # Send final structured data
            yield {
                "type": "structured_data",
                "data": {
                    "final_answer": response.get("output", ""),
                    "intermediate_steps": response.get("intermediate_steps", []),
                },
            }

            # End of stream marker
            yield {"type": "end_of_stream", "data": {}}

        except Exception as e:
            # Error event
            yield {"type": "error", "data": {"error": str(e)}}


@singleton
def get_chat_service(
    llm: Annotated[BaseLanguageModel, Depends(get_llm)],
    tools_registry: Annotated[ToolRegistry, Depends(get_tool_registry)],
) -> ChatService:
    """Factory function to create a configured chat service."""
    return ChatService(llm, tools_registry)
