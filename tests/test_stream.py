"""Unit tests for the LangGraph stream mapper."""

import pytest
from langchain_core.messages import AIMessageChunk, ToolMessage

from chatty.core.service.models import (
    EVENT_TYPE_CONTENT,
    EVENT_TYPE_THINKING,
    EVENT_TYPE_TOOL_CALL,
    TOOL_STATUS_COMPLETED,
    TOOL_STATUS_ERROR,
    TOOL_STATUS_STARTED,
    ContentEvent,
    ThinkingEvent,
    ToolCallEvent,
)
from chatty.core.service.stream import _map_chunk, map_langgraph_stream

# ---------------------------------------------------------------------------
# _map_chunk — synchronous unit tests
# ---------------------------------------------------------------------------


class TestMapChunk:
    """Tests for the internal _map_chunk function."""

    def test_ai_content_from_agent_node_yields_content_event(self):
        chunk = AIMessageChunk(content="Hello")
        event = _map_chunk(chunk, {"langgraph_node": "agent"})
        assert isinstance(event, ContentEvent)
        assert event.type == EVENT_TYPE_CONTENT
        assert event.content == "Hello"

    def test_ai_content_from_other_node_yields_thinking_event(self):
        chunk = AIMessageChunk(content="Let me think...")
        event = _map_chunk(chunk, {"langgraph_node": "planner"})
        assert isinstance(event, ThinkingEvent)
        assert event.type == EVENT_TYPE_THINKING
        assert event.content == "Let me think..."

    def test_ai_content_from_missing_node_yields_thinking_event(self):
        chunk = AIMessageChunk(content="hmm")
        event = _map_chunk(chunk, {})
        assert isinstance(event, ThinkingEvent)
        assert event.content == "hmm"

    def test_ai_empty_content_is_skipped(self):
        chunk = AIMessageChunk(content="")
        assert _map_chunk(chunk, {"langgraph_node": "agent"}) is None

    def test_ai_tool_call_chunk_yields_started_event(self):
        chunk = AIMessageChunk(
            content="",
            tool_call_chunks=[
                {"name": "search_web", "args": '{"q": "hi"}', "id": "c1", "index": 0}
            ],
        )
        event = _map_chunk(chunk, {"langgraph_node": "agent"})
        assert isinstance(event, ToolCallEvent)
        assert event.type == EVENT_TYPE_TOOL_CALL
        assert event.status == TOOL_STATUS_STARTED
        assert event.name == "search_web"
        assert event.arguments == {"q": "hi"}

    def test_ai_tool_call_chunk_with_json_string_args(self):
        """Args arrive as a JSON string; mapper should parse to dict."""
        chunk = AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "name": "fetch_url",
                    "args": '{"url": "http://x"}',
                    "id": "c2",
                    "index": 0,
                }
            ],
        )
        event = _map_chunk(chunk, {"langgraph_node": "agent"})
        assert isinstance(event, ToolCallEvent)
        assert event.arguments == {"url": "http://x"}

    def test_ai_tool_call_chunk_without_name_is_skipped(self):
        """Incremental arg-only chunks (no name) should be skipped."""
        chunk = AIMessageChunk(
            content="",
            tool_call_chunks=[
                {"name": "", "args": '{"partial": true}', "id": "c3", "index": 0}
            ],
        )
        assert _map_chunk(chunk, {"langgraph_node": "agent"}) is None

    def test_ai_tool_call_chunk_with_invalid_json_args(self):
        chunk = AIMessageChunk(
            content="",
            tool_call_chunks=[
                {"name": "my_tool", "args": "not-json", "id": "c4", "index": 0}
            ],
        )
        event = _map_chunk(chunk, {"langgraph_node": "agent"})
        assert isinstance(event, ToolCallEvent)
        assert event.arguments is None  # graceful fallback

    def test_tool_message_completed(self):
        msg = ToolMessage(content="result data", name="search_web", tool_call_id="c1")
        event = _map_chunk(msg, {"langgraph_node": "tools"})
        assert isinstance(event, ToolCallEvent)
        assert event.status == TOOL_STATUS_COMPLETED
        assert event.name == "search_web"
        assert event.result == "result data"

    def test_tool_message_error(self):
        msg = ToolMessage(
            content="timeout", name="fetch_url", tool_call_id="c2", status="error"
        )
        event = _map_chunk(msg, {"langgraph_node": "tools"})
        assert isinstance(event, ToolCallEvent)
        assert event.status == TOOL_STATUS_ERROR
        assert event.result == "timeout"

    def test_tool_message_empty_content(self):
        msg = ToolMessage(content="", name="noop", tool_call_id="c3")
        event = _map_chunk(msg, {"langgraph_node": "tools"})
        assert isinstance(event, ToolCallEvent)
        assert event.result is None

    def test_unknown_chunk_type_is_skipped(self):
        """Non-AI, non-Tool messages should be silently ignored."""

        class FakeChunk:
            pass

        assert _map_chunk(FakeChunk(), {"langgraph_node": "agent"}) is None


# ---------------------------------------------------------------------------
# map_langgraph_stream — async integration tests
# ---------------------------------------------------------------------------


async def _async_iter(items):
    """Helper to create an async generator from a list."""
    for item in items:
        yield item


class TestMapLanggraphStream:
    """Tests for the full async stream mapper."""

    @pytest.mark.asyncio
    async def test_maps_mixed_stream(self):
        """A realistic stream with content and tool events."""
        raw = [
            (AIMessageChunk(content="Thinking..."), {"langgraph_node": "planner"}),
            (
                AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {"name": "search", "args": "{}", "id": "t1", "index": 0}
                    ],
                ),
                {"langgraph_node": "agent"},
            ),
            (
                ToolMessage(content="3 results", name="search", tool_call_id="t1"),
                {"langgraph_node": "tools"},
            ),
            (AIMessageChunk(content="Here is "), {"langgraph_node": "agent"}),
            (AIMessageChunk(content="the answer."), {"langgraph_node": "agent"}),
        ]

        events = [e async for e in map_langgraph_stream(_async_iter(raw))]

        assert len(events) == 5
        assert isinstance(events[0], ThinkingEvent)
        assert isinstance(events[1], ToolCallEvent)
        assert events[1].status == TOOL_STATUS_STARTED
        assert isinstance(events[2], ToolCallEvent)
        assert events[2].status == TOOL_STATUS_COMPLETED
        assert isinstance(events[3], ContentEvent)
        assert isinstance(events[4], ContentEvent)

        # Verify concatenated content
        content_text = "".join(
            e.content for e in events if isinstance(e, ContentEvent)
        )
        assert content_text == "Here is the answer."

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        events = [e async for e in map_langgraph_stream(_async_iter([]))]
        assert events == []

    @pytest.mark.asyncio
    async def test_all_empty_content_skipped(self):
        raw = [
            (AIMessageChunk(content=""), {"langgraph_node": "agent"}),
            (AIMessageChunk(content=""), {"langgraph_node": "agent"}),
        ]
        events = [e async for e in map_langgraph_stream(_async_iter(raw))]
        assert events == []

    @pytest.mark.asyncio
    async def test_only_tool_events(self):
        raw = [
            (
                AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {"name": "fetch", "args": '{"url":"x"}', "id": "t1", "index": 0}
                    ],
                ),
                {"langgraph_node": "agent"},
            ),
            (
                ToolMessage(content="page html", name="fetch", tool_call_id="t1"),
                {"langgraph_node": "tools"},
            ),
        ]
        events = [e async for e in map_langgraph_stream(_async_iter(raw))]
        assert len(events) == 2
        assert all(isinstance(e, ToolCallEvent) for e in events)
        assert events[0].status == TOOL_STATUS_STARTED
        assert events[1].status == TOOL_STATUS_COMPLETED
