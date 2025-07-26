"""Pydantic models for the chat API."""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single message in the conversation."""

    role: Literal["user", "assistant"] = Field(description="Message sender role")
    content: str = Field(description="Message content")


class ChatRequest(BaseModel):
    """Request model for the chat endpoint."""

    query: str = Field(description="User query to process")
    conversation_history: List[ChatMessage] = Field(
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
    data: Dict[str, Any] = Field(description="Structured data payload")


class EndOfStreamEvent(BaseModel):
    """End of stream marker event."""

    type: Literal["end_of_stream"] = "end_of_stream"


class ErrorEvent(BaseModel):
    """Error event."""

    type: Literal["error"] = "error"
    message: str = Field(description="Error message")
    code: Optional[str] = Field(default=None, description="Error code")


# Union type for all possible streaming events
StreamEvent = Union[TokenEvent, StructuredDataEvent, EndOfStreamEvent, ErrorEvent]


class ChatResponse(BaseModel):
    """Non-streaming response model (fallback)."""

    response: str = Field(description="Complete response text")
    structured_data: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional structured data"
    )
