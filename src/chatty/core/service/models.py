from abc import ABC, abstractmethod
from typing import AsyncGenerator

from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables.schema import CustomStreamEvent, StandardStreamEvent

from chatty.configs.config import AppConfig

from .tools.registry import ToolRegistry

# Event type constants
EVENT_TYPE_TOKEN = "token"
EVENT_TYPE_STRUCTURED_DATA = "structured_data"
EVENT_TYPE_END_OF_STREAM = "end_of_stream"

# Structured data type constants
STRUCTURED_TYPE_AGENT_ACTION = "agent_action"
STRUCTURED_TYPE_AGENT_OBSERVATION = "agent_observation"
STRUCTURED_TYPE_AGENT_THOUGHT = "agent_thought"
STRUCTURED_TYPE_TOOL_CALL = "tool_call"
STRUCTURED_TYPE_TOOL_RESULT = "tool_result"
STRUCTURED_TYPE_JSON_OUTPUT = "json_output"
STRUCTURED_TYPE_FINAL_ANSWER = "final_answer"

# Field names for structured data events
FIELD_TYPE = "type"
FIELD_DATA_TYPE = "data_type"
FIELD_DATA = "data"
FIELD_CONTENT = "content"
FIELD_TOOL = "tool"
FIELD_INPUT = "input"
FIELD_OUTPUT = "output"
FIELD_STATUS = "status"


class ChatService(ABC):
    """Abstract base class for chat services."""

    @abstractmethod
    def __init__(
        self,
        llm: BaseLanguageModel,
        tools_registry: ToolRegistry,
        config: AppConfig,
    ) -> None:
        """Initialize the chat service with required dependencies."""
        super().__init__()

    @abstractmethod
    async def stream_response(
        self, question: str
    ) -> AsyncGenerator["ServiceStreamEvent", None]:
        """Stream response tokens and structured data.

        Args:
            question: The user's question

        Yields:
            SSE-compatible events: token events and structured data events
        """
        pass


# class ServiceTokenEvent(BaseModel):
#     """Token streaming event from service layer."""

#     type: Literal["token"] = EVENT_TYPE_TOKEN
#     content: str = Field(description="Individual token content")


# class ServiceEndOfStreamEvent(BaseModel):
#     """End of stream marker event from service layer."""

#     type: Literal["end_of_stream"] = EVENT_TYPE_END_OF_STREAM


# class ServiceStructuredDataEvent(BaseModel):
#     """Final answer structured data event."""

#     type: Literal["structured_data"] = EVENT_TYPE_STRUCTURED_DATA
#     data: dict[str, any] = Field(
#         default_factory=dict,
#         description="Structured data content, can be tool call, result, or final answer",
#     )


# Union type for all service layer streaming events
# ServiceStreamEvent = (
#     ServiceTokenEvent | ServiceStructuredDataEvent | ServiceEndOfStreamEvent
# )
ServiceStreamEvent = StandardStreamEvent | CustomStreamEvent
