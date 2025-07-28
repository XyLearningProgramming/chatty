from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Literal

from langchain_core.language_models import BaseLanguageModel
from pydantic import BaseModel, Field

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


class ServiceTokenEvent(BaseModel):
    """Token streaming event from service layer."""

    type: Literal["token"] = EVENT_TYPE_TOKEN
    content: str = Field(description="Individual token content")


# Typed data models for structured events
class ToolCallData(BaseModel):
    """Typed data for tool call events."""

    tool: str = Field(description="Name of the tool being called")
    input: str = Field(description="Input parameters for the tool")
    status: str = Field(description="Status of the tool call")


class ToolResultData(BaseModel):
    """Typed data for tool result events."""

    output: str = Field(description="Output from the tool execution")
    status: str = Field(description="Status of the tool execution")


class FinalAnswerData(BaseModel):
    """Typed data for final answer events."""

    content: str = Field(description="The final answer content")


# Specific structured data event types
class ServiceToolCallEvent(BaseModel):
    """Tool call structured data event."""

    type: Literal["structured_data"] = EVENT_TYPE_STRUCTURED_DATA
    data_type: Literal["tool_call"] = STRUCTURED_TYPE_TOOL_CALL
    data: ToolCallData


class ServiceToolResultEvent(BaseModel):
    """Tool result structured data event."""

    type: Literal["structured_data"] = EVENT_TYPE_STRUCTURED_DATA
    data_type: Literal["tool_result"] = STRUCTURED_TYPE_TOOL_RESULT
    data: ToolResultData


class ServiceJsonOutputEvent(BaseModel):
    """JSON output structured data event."""

    type: Literal["structured_data"] = EVENT_TYPE_STRUCTURED_DATA
    data_type: Literal["json_output"] = STRUCTURED_TYPE_JSON_OUTPUT
    data: dict[str, Any] = Field(description="Parsed JSON data from code blocks")


class ServiceFinalAnswerEvent(BaseModel):
    """Final answer structured data event."""

    type: Literal["structured_data"] = EVENT_TYPE_STRUCTURED_DATA
    data_type: Literal["final_answer"] = STRUCTURED_TYPE_FINAL_ANSWER
    data: FinalAnswerData


class ServiceEndOfStreamEvent(BaseModel):
    """End of stream marker event from service layer."""

    type: Literal["end_of_stream"] = EVENT_TYPE_END_OF_STREAM


# Union type for structured data events
ServiceStructuredDataEvent = (
    ServiceToolCallEvent
    | ServiceToolResultEvent
    | ServiceJsonOutputEvent
    | ServiceFinalAnswerEvent
)

# Union type for all service layer streaming events
ServiceStreamEvent = (
    ServiceTokenEvent | ServiceStructuredDataEvent | ServiceEndOfStreamEvent
)
