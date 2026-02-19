"""Unit tests for domain stream event models."""

import json

import pytest
from pydantic import ValidationError

from chatty.core.service.models import (
    EVENT_TYPE_CONTENT,
    EVENT_TYPE_ERROR,
    EVENT_TYPE_THINKING,
    EVENT_TYPE_TOOL_CALL,
    TOOL_STATUS_COMPLETED,
    TOOL_STATUS_ERROR,
    TOOL_STATUS_STARTED,
    VALID_EVENT_TYPES,
    VALID_TOOL_STATUSES,
    ContentEvent,
    ErrorEvent,
    ThinkingEvent,
    ToolCallEvent,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify event type and tool status constants are consistent."""

    def test_valid_event_types(self):
        assert EVENT_TYPE_THINKING in VALID_EVENT_TYPES
        assert EVENT_TYPE_CONTENT in VALID_EVENT_TYPES
        assert EVENT_TYPE_TOOL_CALL in VALID_EVENT_TYPES
        assert EVENT_TYPE_ERROR in VALID_EVENT_TYPES
        assert len(VALID_EVENT_TYPES) == 4

    def test_valid_tool_statuses(self):
        assert TOOL_STATUS_STARTED in VALID_TOOL_STATUSES
        assert TOOL_STATUS_COMPLETED in VALID_TOOL_STATUSES
        assert TOOL_STATUS_ERROR in VALID_TOOL_STATUSES
        assert len(VALID_TOOL_STATUSES) == 3

    def test_event_models_match_constants(self):
        """Each model's default type field matches the constant."""
        assert ThinkingEvent(content="x").type == EVENT_TYPE_THINKING
        assert ContentEvent(content="x").type == EVENT_TYPE_CONTENT
        assert (
            ToolCallEvent(name="t", status=TOOL_STATUS_STARTED).type
            == EVENT_TYPE_TOOL_CALL
        )
        assert ErrorEvent(message="m").type == EVENT_TYPE_ERROR


# ---------------------------------------------------------------------------
# ThinkingEvent
# ---------------------------------------------------------------------------


class TestThinkingEvent:
    def test_serialization(self):
        event = ThinkingEvent(content="reasoning step")
        data = json.loads(event.model_dump_json())
        assert data == {"type": "thinking", "content": "reasoning step"}

    def test_type_is_literal(self):
        event = ThinkingEvent(content="abc")
        assert event.type == "thinking"

    def test_empty_content(self):
        event = ThinkingEvent(content="")
        assert event.content == ""

    def test_requires_content(self):
        with pytest.raises(ValidationError):
            ThinkingEvent()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# ContentEvent
# ---------------------------------------------------------------------------


class TestContentEvent:
    def test_serialization(self):
        event = ContentEvent(content="hello")
        data = json.loads(event.model_dump_json())
        assert data == {"type": "content", "content": "hello"}

    def test_type_is_literal(self):
        assert ContentEvent(content="x").type == "content"

    def test_requires_content(self):
        with pytest.raises(ValidationError):
            ContentEvent()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# ToolCallEvent
# ---------------------------------------------------------------------------


class TestToolCallEvent:
    def test_started_event(self):
        event = ToolCallEvent(
            name="search_web",
            status="started",
            arguments={"query": "LangGraph"},
        )
        data = json.loads(event.model_dump_json())
        assert data["type"] == "tool_call"
        assert data["name"] == "search_web"
        assert data["status"] == "started"
        assert data["arguments"] == {"query": "LangGraph"}
        assert data["result"] is None

    def test_completed_event(self):
        event = ToolCallEvent(
            name="search_web",
            status="completed",
            result="found 3 results",
        )
        data = json.loads(event.model_dump_json())
        assert data["status"] == "completed"
        assert data["result"] == "found 3 results"
        assert data["arguments"] is None

    def test_error_event(self):
        event = ToolCallEvent(
            name="search_web",
            status="error",
            result="connection timeout",
        )
        assert event.status == "error"
        assert event.result == "connection timeout"

    def test_defaults(self):
        event = ToolCallEvent(name="tool", status="started")
        assert event.arguments is None
        assert event.result is None

    def test_requires_name_and_status(self):
        with pytest.raises(ValidationError):
            ToolCallEvent()  # type: ignore[call-arg]

    def test_invalid_status_rejected(self):
        with pytest.raises(ValidationError):
            ToolCallEvent(name="t", status="running")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ErrorEvent
# ---------------------------------------------------------------------------


class TestErrorEvent:
    def test_serialization(self):
        event = ErrorEvent(message="boom", code="PROCESSING_ERROR")
        data = json.loads(event.model_dump_json())
        assert data == {
            "type": "error",
            "message": "boom",
            "code": "PROCESSING_ERROR",
        }

    def test_code_optional(self):
        event = ErrorEvent(message="oops")
        assert event.code is None

    def test_requires_message(self):
        with pytest.raises(ValidationError):
            ErrorEvent()  # type: ignore[call-arg]
