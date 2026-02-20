"""Pydantic models for the chat API."""

import logging
from traceback import format_exc

from pydantic import BaseModel, Field

from chatty.core.service.models import ErrorEvent, StreamEvent

logger = logging.getLogger(__name__)

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
        description="User query to process", max_length=512,
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


def format_error_sse(exc: Exception, *, send_traceback: bool = False) -> str:
    """Serialize an exception to an SSE error event."""
    if send_traceback:
        message = f"An error occurred during processing: {format_exc()}"
    else:
        message = "An internal error occurred."
        logger.error("Hidden error full stack trace: %s", format_exc())
    error_event = ErrorEvent(message=message, code="PROCESSING_ERROR")
    return f"data: {error_event.model_dump_json()}\n\n"
