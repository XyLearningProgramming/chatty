"""Pydantic models for the chat API."""

from traceback import format_exc

from pydantic import BaseModel, Field

from chatty.core.service.models import ErrorEvent, StreamEvent

# Maximum length for chat query input, only short queries are allowed
CHAT_QUERY_MAX_LENGTH = 1024

# Re-export for the API layer
__all__ = ["ChatRequest", "StreamEvent", "ErrorEvent"]


class ChatRequest(BaseModel):
    """Request model for the chat endpoint.

    Two modes of operation (ChatGPT-style):

    - **New conversation**: omit ``conversation_id`` — the server
      generates one and returns it in the ``X-Chatty-Conversation``
      response header.
    - **Continue conversation**: pass the ``conversation_id`` received
      from a previous turn — the server loads history from the DB.
    """

    query: str = Field(
        description="User query to process", max_length=CHAT_QUERY_MAX_LENGTH
    )
    conversation_id: str | None = Field(
        default=None,
        description="Existing conversation ID for follow-up turns. "
        "Omit to start a new conversation.",
    )
    nonce: str | None = Field(
        default=None,
        max_length=128,
        description="Client-generated unique ID for this request. "
        "If provided, the server rejects duplicate nonces within 60 s.",
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
