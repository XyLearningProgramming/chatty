"""Domain models for the chat service layer."""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Literal

from langchain_core.language_models import BaseLanguageModel
from pydantic import BaseModel, Field

from chatty.configs.config import AppConfig
from chatty.core.service.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Event type constants — import these instead of duplicating strings.
# ---------------------------------------------------------------------------

EVENT_TYPE_QUEUED = "queued"
EVENT_TYPE_DEQUEUED = "dequeued"
EVENT_TYPE_THINKING = "thinking"
EVENT_TYPE_CONTENT = "content"
EVENT_TYPE_TOOL_CALL = "tool_call"
EVENT_TYPE_ERROR = "error"

VALID_EVENT_TYPES = frozenset({
    EVENT_TYPE_QUEUED,
    EVENT_TYPE_DEQUEUED,
    EVENT_TYPE_THINKING,
    EVENT_TYPE_CONTENT,
    EVENT_TYPE_TOOL_CALL,
    EVENT_TYPE_ERROR,
})

# Tool call lifecycle statuses
TOOL_STATUS_STARTED = "started"
TOOL_STATUS_COMPLETED = "completed"
TOOL_STATUS_ERROR = "error"

VALID_TOOL_STATUSES = frozenset({
    TOOL_STATUS_STARTED,
    TOOL_STATUS_COMPLETED,
    TOOL_STATUS_ERROR,
})


# ---------------------------------------------------------------------------
# Domain stream events
# ---------------------------------------------------------------------------


class QueuedEvent(BaseModel):
    """Request was admitted into the concurrency gate inbox."""

    type: Literal["queued"] = "queued"
    position: int = Field(description="Current inbox occupancy after admission")
    message: str = Field(
        default="Request queued, waiting for available slot",
        description="Human-readable status message",
    )


class DequeuedEvent(BaseModel):
    """Concurrency slot acquired — agent run is starting."""

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


# ---------------------------------------------------------------------------
# Abstract chat service
# ---------------------------------------------------------------------------


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
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream response as domain events.

        Args:
            question: The user's question

        Yields:
            StreamEvent instances (thinking, content, tool_call, error)
        """
        pass
