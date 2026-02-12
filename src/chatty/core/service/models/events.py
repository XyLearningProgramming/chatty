"""Domain stream events emitted by the chat service."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class QueuedEvent(BaseModel):
    """Request was admitted into the concurrency gate inbox."""

    type: Literal["queued"] = "queued"
    position: int = Field(description="Current inbox occupancy after admission")
    message: str = Field(
        default="Request queued, waiting for available slot",
        description="Human-readable status message",
    )


class DequeuedEvent(BaseModel):
    """Concurrency slot acquired — agent run is starting.

    Infrastructure IDs (conversation_id, trace_id) are delivered via
    response headers (X-Chatty-Conversation, X-Chatty-Trace), not in
    the SSE stream — following the ChatGPT convention.
    """

    type: Literal["dequeued"] = "dequeued"
    message: str = Field(
        default="Slot acquired, processing request",
        description="Human-readable status message",
    )


class ThinkingEvent(BaseModel):
    """Agent internal reasoning (intermediate thoughts, scratchpad)."""

    type: Literal["thinking"] = "thinking"
    content: str = Field(description="Agent reasoning content")


class ContentEvent(BaseModel):
    """User-facing streamed text tokens (final answer)."""

    type: Literal["content"] = "content"
    content: str = Field(description="Text token content")
    message_id: str | None = Field(
        default=None,
        description="Provider message ID (e.g. OpenAI chatcmpl-xxx)",
    )


class ToolCallEvent(BaseModel):
    """Tool invocation lifecycle event."""

    type: Literal["tool_call"] = "tool_call"
    name: str = Field(description="Tool name, e.g. 'search_website'")
    status: Literal["started", "completed", "error"] = Field(
        description="Tool call lifecycle status"
    )
    arguments: dict[str, Any] | None = Field(
        default=None, description="Tool arguments (present when started)"
    )
    result: str | None = Field(
        default=None,
        description="Tool result (present when completed or error)",
    )
    message_id: str | None = Field(
        default=None,
        description="Tool call ID (e.g. OpenAI call_xxx)",
    )


class ErrorEvent(BaseModel):
    """Stream-level error event."""

    type: Literal["error"] = "error"
    message: str = Field(description="Error message")
    code: str | None = Field(default=None, description="Error code")


StreamEvent = (
    QueuedEvent
    | DequeuedEvent
    | ThinkingEvent
    | ContentEvent
    | ToolCallEvent
    | ErrorEvent
)
