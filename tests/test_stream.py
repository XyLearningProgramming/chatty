"""Unit tests for the LLM stream mapper (thin pass-through; upstream does reasoning/tool_calls)."""

import pytest
from langchain_core.messages import AIMessageChunk

from chatty.core.service.models import (
    EVENT_TYPE_CONTENT,
    EVENT_TYPE_THINKING,
    EVENT_TYPE_TOOL_CALL,
    TOOL_STATUS_STARTED,
    ContentEvent,
    ThinkingEvent,
    ToolCallEvent,
)
from chatty.core.service.stream import (
    chunk_to_thinking_and_content,
    map_llm_stream,
    normalize_tool_call,
)


async def _async_iter(items):
    for item in items:
        yield item


# ---------------------------------------------------------------------------
# normalize_tool_call
# ---------------------------------------------------------------------------


class TestNormalizeToolCall:
    def test_langchain_shape(self):
        n = normalize_tool_call({"name": "lookup", "args": {"source": "resume"}, "id": "c1"})
        assert n == {"name": "lookup", "args": {"source": "resume"}, "id": "c1"}

    def test_openai_slm_server_shape(self):
        n = normalize_tool_call({
            "id": "call_abc",
            "type": "function",
            "function": {"name": "search", "arguments": '{"q": "x"}'},
        })
        assert n["name"] == "search"
        assert n["args"] == {"q": "x"}
        assert n["id"] == "call_abc"

    def test_missing_name_returns_none(self):
        assert normalize_tool_call({"args": "{}", "id": "c1"}) is None
        assert normalize_tool_call({"function": {"arguments": "{}"}}) is None


# ---------------------------------------------------------------------------
# chunk_to_thinking_and_content (RAG: thinking + content only, no tool_call)
# ---------------------------------------------------------------------------


class TestChunkToThinkingAndContent:
    def test_content_only(self):
        chunk = AIMessageChunk(content="Hello")
        events = list(chunk_to_thinking_and_content(chunk))
        assert len(events) == 1
        assert isinstance(events[0], ContentEvent)
        assert events[0].content == "Hello"

    def test_reasoning_and_content(self):
        chunk = AIMessageChunk(
            content="Answer.",
            additional_kwargs={"reasoning_content": "Think."},
        )
        events = list(chunk_to_thinking_and_content(chunk))
        assert len(events) == 2
        assert isinstance(events[0], ThinkingEvent)
        assert events[0].content == "Think."
        assert isinstance(events[1], ContentEvent)
        assert events[1].content == "Answer."


# ---------------------------------------------------------------------------
# map_llm_stream
# ---------------------------------------------------------------------------


class TestMapLlmStream:
    @pytest.mark.asyncio
    async def test_maps_content_chunks(self):
        chunks = [
            AIMessageChunk(content="Hello "),
            AIMessageChunk(content="world"),
        ]
        events = [e async for e in map_llm_stream(_async_iter(chunks))]
        assert len(events) == 2
        assert all(isinstance(e, ContentEvent) for e in events)
        assert events[0].content == "Hello "
        assert events[1].content == "world"

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        events = [e async for e in map_llm_stream(_async_iter([]))]
        assert events == []

    @pytest.mark.asyncio
    async def test_empty_content_skipped(self):
        chunks = [
            AIMessageChunk(content=""),
            AIMessageChunk(content=""),
        ]
        events = [e async for e in map_llm_stream(_async_iter(chunks))]
        assert events == []

    @pytest.mark.asyncio
    async def test_reasoning_content_yields_thinking_event(self):
        chunks = [
            AIMessageChunk(
                content="",
                additional_kwargs={"reasoning_content": "Let me think."},
            ),
            AIMessageChunk(content="The answer is 42."),
        ]
        events = [e async for e in map_llm_stream(_async_iter(chunks))]
        assert len(events) == 2
        assert isinstance(events[0], ThinkingEvent)
        assert events[0].type == EVENT_TYPE_THINKING
        assert events[0].content == "Let me think."
        assert isinstance(events[1], ContentEvent)
        assert events[1].content == "The answer is 42."

    @pytest.mark.asyncio
    async def test_tool_call_chunk_yields_started_event(self):
        chunks = [
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": "lookup", "args": '{"source": "resume"}', "id": "c1", "index": 0}
                ],
            ),
        ]
        events = [e async for e in map_llm_stream(_async_iter(chunks))]
        assert len(events) == 1
        assert isinstance(events[0], ToolCallEvent)
        assert events[0].type == EVENT_TYPE_TOOL_CALL
        assert events[0].status == TOOL_STATUS_STARTED
        assert events[0].name == "lookup"
        assert events[0].arguments == {"source": "resume"}
        assert events[0].message_id == "c1"

    @pytest.mark.asyncio
    async def test_tool_call_message_id_set(self):
        """ToolCallEvent gets message_id from chunk (for linking completed/error)."""
        chunks = [
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": "search_web", "args": '{"query": "LangChain"}', "id": "call_xyz", "index": 0}
                ],
            ),
        ]
        events = [e async for e in map_llm_stream(_async_iter(chunks))]
        assert len(events) == 1
        assert events[0].name == "search_web"
        assert events[0].arguments == {"query": "LangChain"}
        assert events[0].message_id == "call_xyz"

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_one_chunk(self):
        chunks = [
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": "lookup", "args": "{}", "id": "t1", "index": 0},
                    {"name": "search", "args": '{"q": "x"}', "id": "t2", "index": 1},
                ],
            ),
        ]
        events = [e async for e in map_llm_stream(_async_iter(chunks))]
        assert len(events) == 2
        assert events[0].name == "lookup" and events[0].message_id == "t1"
        assert events[1].name == "search" and events[1].arguments == {"q": "x"} and events[1].message_id == "t2"

    @pytest.mark.asyncio
    async def test_tool_call_without_name_skipped(self):
        chunks = [
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": "", "args": '{"partial": true}', "id": "c3", "index": 0}
                ],
            ),
        ]
        events = [e async for e in map_llm_stream(_async_iter(chunks))]
        assert events == []

    @pytest.mark.asyncio
    async def test_mixed_content_and_tool_stream(self):
        chunks = [
            AIMessageChunk(content="Let me look that up."),
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": "lookup", "args": '{"source": "resume"}', "id": "t1", "index": 0}
                ],
            ),
        ]
        events = [e async for e in map_llm_stream(_async_iter(chunks))]
        assert len(events) == 2
        assert isinstance(events[0], ContentEvent)
        assert isinstance(events[1], ToolCallEvent)
        assert events[1].status == TOOL_STATUS_STARTED

    @pytest.mark.asyncio
    async def test_accumulator_receives_message(self):
        from chatty.core.service.stream import StreamAccumulator

        chunks = [
            AIMessageChunk(content="Hi"),
            AIMessageChunk(content=" there."),
        ]
        acc = StreamAccumulator()
        events = [e async for e in map_llm_stream(_async_iter(chunks), accumulator=acc)]
        assert len(events) == 2
        assert acc.message is not None
        assert (acc.message.content or "") == "Hi there."
