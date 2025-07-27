"""Pydantic models for the chat API."""

from typing import Any, Literal

from pydantic import BaseModel, Field

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


class ChatResponse(BaseModel):
    """Non-streaming response model (fallback)."""

    response: str = Field(description="Complete response text")
    structured_data: dict[str, Any] | None = Field(
        default=None, description="Optional structured data"
    )
