"""Domain stream events emitted by the chat service."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class QueuedEvent(BaseModel):
    """First event on every SSE stream — sent right before streaming begins.

    Confirms that the request was admitted into the inbox and tells the
    client its current position.  Inbox admission (``inbox.enter()``)
    happens in the ``enforce_inbox`` dependency *before* the response
    starts; this event is emitted as soon as the SSE generator runs,
    making it the earliest signal the client can receive.
    """

    type: Literal["queued"] = "queued"
    position: int = Field(description="Current inbox occupancy after admission")
    message: str = Field(
        default="Request admitted, starting processing",
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


StreamEvent = QueuedEvent | ThinkingEvent | ContentEvent | ToolCallEvent | ErrorEvent
