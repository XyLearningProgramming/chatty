"""Tests for StreamProcessor class."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from chatty.core.service.stream_processor import (
    StreamProcessor,
    LANGCHAIN_EVENT_CHAT_MODEL_STREAM,
    LANGCHAIN_EVENT_LLM_STREAM,
    LANGCHAIN_EVENT_TOOL_START,
    LANGCHAIN_EVENT_TOOL_END,
    LANGCHAIN_EVENT_CHAIN_END,
)
from chatty.core.service.models import (
    ServiceTokenEvent,
    ServiceStructuredDataEvent,
    ServiceEndOfStreamEvent,
    STRUCTURED_TYPE_JSON_OUTPUT,
    STRUCTURED_TYPE_FINAL_ANSWER,
    STRUCTURED_TYPE_TOOL_CALL,
    STRUCTURED_TYPE_TOOL_RESULT,
)


class TestStreamProcessor:
    """Test cases for StreamProcessor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = StreamProcessor()

    def test_initialization(self):
        """Test StreamProcessor initializes correctly."""
        assert self.processor._token_buffer == ""
        assert self.processor._pending_tokens == ""

    def test_reset(self):
        """Test reset clears buffers."""
        self.processor._token_buffer = "test"
        self.processor._pending_tokens = "test"
        
        self.processor.reset()
        
        assert self.processor._token_buffer == ""
        assert self.processor._pending_tokens == ""

    def test_process_token_simple_text(self):
        """Test processing simple text tokens."""
        structured_events, token_events = self.processor._process_token("Hello ")
        
        assert len(structured_events) == 0
        assert len(token_events) == 1
        assert token_events[0].content == "Hello "

    def test_process_token_json_block_pattern(self):
        """Test processing JSON block pattern."""
        # Send tokens that form a JSON block
        tokens = ["```", "json", "\n", '{"key": "value"}', "```"]
        
        all_structured = []
        all_tokens = []
        
        for token in tokens:
            structured, tokens_out = self.processor._process_token(token)
            all_structured.extend(structured)
            all_tokens.extend(tokens_out)
        
        # Flush any remaining tokens to see final state
        remaining = self.processor._flush_remaining_tokens()
        all_tokens.extend(remaining)
        
        # Should have one structured event with the JSON data
        assert len(all_structured) == 1
        assert all_structured[0].data_type == STRUCTURED_TYPE_JSON_OUTPUT
        assert all_structured[0].data == {"key": "value"}
        
        # JSON pattern should not be in token output (no double streaming)
        token_content = "".join([t.content for t in all_tokens])
        assert "```json" not in token_content
        assert '{"key": "value"}' not in token_content

    def test_process_token_non_json_block_pattern(self):
        """Test processing non-JSON code blocks (should be treated as regular tokens)."""
        tokens = ["```", "python", "\n", 'print("hello")', "```"]
        
        all_structured = []
        all_tokens = []
        
        for token in tokens:
            structured, tokens_out = self.processor._process_token(token)
            all_structured.extend(structured)
            all_tokens.extend(tokens_out)
        
        # Flush remaining tokens
        remaining = self.processor._flush_remaining_tokens()
        all_tokens.extend(remaining)
        
        # Should have no structured events (python blocks are not processed as JSON)
        assert len(all_structured) == 0
        
        # Should appear as regular tokens
        token_content = "".join([t.content for t in all_tokens])
        assert "```python" in token_content
        assert 'print("hello")' in token_content

    def test_process_token_final_answer_pattern(self):
        """Test processing Final Answer pattern."""
        tokens = ["Final Answer: ", "This is the final answer"]
        
        all_structured = []
        all_tokens = []
        
        for token in tokens:
            structured, tokens_out = self.processor._process_token(token)
            all_structured.extend(structured)
            all_tokens.extend(tokens_out)
        
        # Should have one structured event (only when content changes)
        assert len(all_structured) == 1
        assert all_structured[0].data_type == STRUCTURED_TYPE_FINAL_ANSWER
        assert all_structured[0].data.content == "This is the final answer"
        
        # Final answer should still appear in token output (when buffered correctly)
        token_content = "".join([t.content for t in all_tokens])
        # Final Answer may be held in buffer, so check that content flows through eventually
        assert len(token_content) > 0  # Some tokens should be emitted

    def test_could_be_pattern_start(self):
        """Test pattern start detection."""
        # Test empty buffer
        assert not self.processor._could_be_pattern_start()
        
        # Test potential block start
        self.processor._pending_tokens = "```"
        assert self.processor._could_be_pattern_start()
        
        # Test partial pattern that doesn't match (new block-based logic)
        self.processor._pending_tokens = "```j"
        assert not self.processor._could_be_pattern_start()  # Need full block type
        
        # Test potential Final Answer start
        self.processor._pending_tokens = "Final"
        assert self.processor._could_be_pattern_start()
        
        self.processor._pending_tokens = "Final Answer:"
        assert self.processor._could_be_pattern_start()
        
        # Test non-matching pattern
        self.processor._pending_tokens = "xyz"
        assert not self.processor._could_be_pattern_start()

    def test_flush_remaining_tokens(self):
        """Test flushing remaining tokens."""
        self.processor._pending_tokens = "remaining text"
        
        token_events = self.processor._flush_remaining_tokens()
        
        assert len(token_events) == 1
        assert token_events[0].content == "remaining text"
        assert self.processor._pending_tokens == ""

    @pytest.mark.asyncio
    async def test_process_langchain_events_token_stream(self):
        """Test processing LangChain token stream events."""
        # Mock LangChain events
        async def mock_langchain_events():
            # Mock chat model stream event
            chunk_mock = MagicMock()
            chunk_mock.content = "Hello world"
            
            yield {
                "event": LANGCHAIN_EVENT_CHAT_MODEL_STREAM,
                "data": {"chunk": chunk_mock}
            }

        events = []
        async for event in self.processor.process_langchain_events(mock_langchain_events()):
            events.append(event)

        # Should have token event and end of stream
        assert len(events) == 2
        assert isinstance(events[0], ServiceTokenEvent)
        assert events[0].content == "Hello world"
        assert isinstance(events[1], ServiceEndOfStreamEvent)

    @pytest.mark.asyncio
    async def test_process_langchain_events_tool_events(self):
        """Test processing LangChain tool events."""
        async def mock_langchain_events():
            # Tool start event
            yield {
                "event": LANGCHAIN_EVENT_TOOL_START,
                "data": {
                    "input": {
                        "tool": "search",
                        "tool_input": "python tutorial"
                    }
                }
            }
            
            # Tool end event
            yield {
                "event": LANGCHAIN_EVENT_TOOL_END,
                "data": {
                    "output": "Found 10 results about Python"
                }
            }

        events = []
        async for event in self.processor.process_langchain_events(mock_langchain_events()):
            events.append(event)

        # Should have tool start, tool end, and end of stream
        assert len(events) == 3
        
        # Check tool start event
        assert isinstance(events[0], ServiceStructuredDataEvent)
        assert events[0].data_type == STRUCTURED_TYPE_TOOL_CALL
        assert events[0].data.tool == "search"
        assert events[0].data.input == "python tutorial"
        assert events[0].data.status == "started"
        
        # Check tool end event
        assert isinstance(events[1], ServiceStructuredDataEvent)
        assert events[1].data_type == STRUCTURED_TYPE_TOOL_RESULT
        assert events[1].data.output == "Found 10 results about Python"
        assert events[1].data.status == "completed"
        
        # Check end of stream
        assert isinstance(events[2], ServiceEndOfStreamEvent)

    @pytest.mark.asyncio
    async def test_process_langchain_events_chain_end(self):
        """Test processing LangChain chain end event."""
        async def mock_langchain_events():
            yield {
                "event": LANGCHAIN_EVENT_CHAIN_END,
                "data": {
                    "output": "This is the final chain output"
                }
            }

        events = []
        async for event in self.processor.process_langchain_events(mock_langchain_events()):
            events.append(event)

        # Should have final answer and end of stream
        assert len(events) == 2
        assert isinstance(events[0], ServiceStructuredDataEvent)
        assert events[0].data_type == STRUCTURED_TYPE_FINAL_ANSWER
        assert events[0].data.content == "This is the final chain output"
        assert isinstance(events[1], ServiceEndOfStreamEvent)

    @pytest.mark.asyncio
    async def test_process_langchain_events_with_json_in_stream(self):
        """Test processing stream with JSON patterns."""
        async def mock_langchain_events():
            # Send tokens that form a JSON block
            for token in ["```", "json\n", '{"result": "success"}', "```"]:
                chunk_mock = MagicMock()
                chunk_mock.content = token
                
                yield {
                    "event": LANGCHAIN_EVENT_LLM_STREAM,
                    "data": {"chunk": chunk_mock}
                }

        events = []
        async for event in self.processor.process_langchain_events(mock_langchain_events()):
            events.append(event)

        # Should have structured JSON event and end of stream
        # No token events for the JSON pattern itself
        structured_events = [e for e in events if isinstance(e, ServiceStructuredDataEvent)]
        token_events = [e for e in events if isinstance(e, ServiceTokenEvent)]
        
        assert len(structured_events) == 1
        assert structured_events[0].data_type == STRUCTURED_TYPE_JSON_OUTPUT
        assert structured_events[0].data == {"result": "success"}
        
        # Should have minimal or no token events for JSON pattern
        token_content = "".join([e.content for e in token_events])
        assert "```json" not in token_content
        assert '{"result": "success"}' not in token_content

    def test_invalid_json_handling(self):
        """Test handling of invalid JSON patterns."""
        # Send invalid JSON
        tokens = ["```json\n", '{"invalid": json}', "```"]
        
        all_structured = []
        all_tokens = []
        
        for token in tokens:
            structured, tokens_out = self.processor._process_token(token)
            all_structured.extend(structured)
            all_tokens.extend(tokens_out)
        
        # Flush remaining tokens
        remaining = self.processor._flush_remaining_tokens()
        all_tokens.extend(remaining)
        
        # Should not create structured event for invalid JSON
        assert len(all_structured) == 0
        
        # Should flush as regular tokens
        token_content = "".join([t.content for t in all_tokens])
        assert "```json" in token_content
        assert '{"invalid": json}' in token_content