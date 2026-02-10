"""Pydantic models for the chat API."""

from traceback import format_exc
from typing import Literal

from pydantic import BaseModel, Field

from chatty.core.service.models import ErrorEvent, StreamEvent

# Maximum length for chat query input, only short queries are allowed
CHAT_QUERY_MAX_LENGTH = 1024

# Re-export for the API layer
__all__ = ["ChatMessage", "ChatRequest", "StreamEvent", "ErrorEvent"]


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


def format_sse(event: StreamEvent) -> str:
    """Serialize a domain StreamEvent to an SSE data line."""
    return f"data: {event.model_dump_json()}\n\n"


def format_error_sse(exc: Exception) -> str:
    """Serialize an exception to an SSE error event."""
    error_event = ErrorEvent(
        message=f"An error occurred during processing: {format_exc()}",
        code="PROCESSING_ERROR",
    )
    return f"data: {error_event.model_dump_json()}\n\n"
