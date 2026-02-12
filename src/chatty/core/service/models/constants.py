"""Event type and tool status constants."""

# ---------------------------------------------------------------------------
# Event type constants — import these instead of duplicating strings.
# ---------------------------------------------------------------------------

EVENT_TYPE_QUEUED = "queued"
EVENT_TYPE_DEQUEUED = "dequeued"
EVENT_TYPE_THINKING = "thinking"
EVENT_TYPE_CONTENT = "content"
EVENT_TYPE_TOOL_CALL = "tool_call"
EVENT_TYPE_ERROR = "error"

VALID_EVENT_TYPES = frozenset(
    {
        EVENT_TYPE_QUEUED,
        EVENT_TYPE_DEQUEUED,
        EVENT_TYPE_THINKING,
        EVENT_TYPE_CONTENT,
        EVENT_TYPE_TOOL_CALL,
        EVENT_TYPE_ERROR,
    }
)

# Tool call lifecycle statuses
TOOL_STATUS_STARTED = "started"
TOOL_STATUS_COMPLETED = "completed"
TOOL_STATUS_ERROR = "error"

VALID_TOOL_STATUSES = frozenset(
    {
        TOOL_STATUS_STARTED,
        TOOL_STATUS_COMPLETED,
        TOOL_STATUS_ERROR,
    }
)
