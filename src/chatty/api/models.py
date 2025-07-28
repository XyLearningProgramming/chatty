"""Pydantic models for the chat API."""

from traceback import format_exc
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from chatty.core.service.models import ServiceStreamEvent

# Maximum length for chat query input, only short queries are allowed
CHAT_QUERY_MAX_LENGTH = 1024


class ChatMessage(BaseModel):
    """A single message in the conversation."""

    role: Literal["user", "assistant"] = Field(description="Message sender role")
    content: str = Field(description="Message content")


class ChatRequest(BaseModel):
    """Request model for the chat endpoint."""

    query: str = Field(
        description="User query to process", max_length=CHAT_QUERY_MAX_LENGTH
    )
    conversation_history: list[ChatMessage] = Field(
        default_factory=list,
        description="Previous conversation messages for context",
    )


class TokenEvent(BaseModel):
    """Token streaming event."""

    type: Literal["token"] = "token"
    content: str = Field(description="Token content")


class StructuredDataEvent(BaseModel):
    """Structured data event."""

    type: Literal["structured_data"] = "structured_data"
    data: dict[str, Any] = Field(description="Structured data payload")


class EndOfStreamEvent(BaseModel):
    """End of stream marker event."""

    type: Literal["end_of_stream"] = "end_of_stream"


class ErrorEvent(BaseModel):
    """Error event."""

    type: Literal["error"] = "error"
    message: str = Field(description="Error message")
    code: str | None = Field(default=None, description="Error code")


# Union type for all possible streaming events
StreamEvent = TokenEvent | StructuredDataEvent | EndOfStreamEvent | ErrorEvent


def convert_service_event_to_api_event(service_event: "ServiceStreamEvent") -> str:
    """Convert service layer event to API layer SSE format efficiently.

    Service and API events now have matching field names, so conversion is direct.
    """
    from chatty.core.service.models import (
        ServiceEndOfStreamEvent,
        ServiceStructuredDataEvent,
        ServiceTokenEvent,
    )

    if isinstance(service_event, ServiceTokenEvent):
        api_event = TokenEvent(**service_event.model_dump())
    elif isinstance(service_event, ServiceStructuredDataEvent):
        api_event = StructuredDataEvent(**service_event.model_dump())
    elif isinstance(service_event, ServiceEndOfStreamEvent):
        api_event = EndOfStreamEvent(**service_event.model_dump())
    else:
        # Fallback for unknown event types
        api_event = ErrorEvent(
            message=f"Unknown event type: {type(service_event)}",
            code="UNKNOWN_EVENT_TYPE",
        )

    return f"data: {api_event.model_dump_json()}\n\n"


def convert_service_exc_to_api_error_event(exc: Exception) -> str:
    error_event = ErrorEvent(
        message=f"An error occurred during processing: {format_exc()}",
        code="PROCESSING_ERROR",
    )
    return f"data: {error_event.model_dump_json()}\n\n"


class ChatResponse(BaseModel):
    """Non-streaming response model (fallback)."""

    response: str = Field(description="Complete response text")
    structured_data: dict[str, Any] | None = Field(
        default=None, description="Optional structured data"
    )
