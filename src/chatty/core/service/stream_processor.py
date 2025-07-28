"""Stream processor for handling LangChain events and extracting structured patterns."""

import json
import re
from typing import Any, AsyncGenerator

from .models import (
    FIELD_CONTENT,
    FIELD_INPUT,
    FIELD_OUTPUT,
    FIELD_STATUS,
    FIELD_TOOL,
    FinalAnswerData,
    ServiceEndOfStreamEvent,
    ServiceFinalAnswerEvent,
    ServiceJsonOutputEvent,
    ServiceStreamEvent,
    ServiceStructuredDataEvent,
    ServiceTokenEvent,
    ServiceToolCallEvent,
    ServiceToolResultEvent,
    ToolCallData,
    ToolResultData,
)

# LangChain event type constants
LANGCHAIN_EVENT_CHAT_MODEL_STREAM = "on_chat_model_stream"
LANGCHAIN_EVENT_LLM_STREAM = "on_llm_stream"
LANGCHAIN_EVENT_TOOL_START = "on_tool_start"
LANGCHAIN_EVENT_TOOL_END = "on_tool_end"
LANGCHAIN_EVENT_CHAIN_START = "on_chain_start"
LANGCHAIN_EVENT_CHAIN_END = "on_chain_end"

# Pattern constants for structured output detection
BLOCK_START_PATTERN = re.compile(r"```(\w+)")  # Matches ```json, ```python, etc.
BLOCK_END_PATTERN = re.compile(r"```")
FINAL_ANSWER_PATTERN = re.compile(r"Final Answer:\s*(.*?)(?=\n\n|\Z)", re.DOTALL)

# Buffer management constants
MAX_PREFIX_BUFFER = 15  # Cache enough to match "Final Answer:" or "```json"


class StreamProcessor:
    """Processes LangChain stream events and extracts structured patterns."""

    def __init__(self):
        """Initialize the stream processor."""
        self._token_buffer = ""
        self._pending_tokens = ""
        self._last_final_answer = ""
        self._in_block = False
        self._block_type = ""
        self._block_content = ""

    def reset(self) -> None:
        """Reset internal buffers for a new stream."""
        self._token_buffer = ""
        self._pending_tokens = ""
        self._last_final_answer = ""
        self._in_block = False
        self._block_type = ""
        self._block_content = ""

    def _process_token(
        self, token: str
    ) -> tuple[list[ServiceStructuredDataEvent], list[ServiceTokenEvent]]:
        """Process a token with smart buffering to avoid double streaming.

        Args:
            token: New token to process

        Returns:
            Tuple of (structured_events, token_events) to yield
        """
        self._token_buffer += token
        self._pending_tokens += token

        structured_events = []
        token_events = []

        # If we're in a block, keep collecting until we see closing ```
        if self._in_block:
            if BLOCK_END_PATTERN.search(token):
                # Found end of block
                self._in_block = False
                
                # Try to decode JSON if it's a json block
                if self._block_type == "json":
                    try:
                        json_data = json.loads(self._block_content.strip())
                        structured_events.append(
                            ServiceJsonOutputEvent(data=json_data)
                        )
                        # Don't emit the block tokens - they're consumed by structured output
                        self._pending_tokens = ""
                        self._block_content = ""
                        self._block_type = ""
                        return structured_events, token_events
                    except json.JSONDecodeError:
                        # Invalid JSON - flush everything as regular tokens
                        pass
                
                # Not JSON or invalid JSON - flush as regular tokens
                if self._pending_tokens:
                    token_events.append(ServiceTokenEvent(content=self._pending_tokens))
                    self._pending_tokens = ""
                self._block_content = ""
                self._block_type = ""
                return structured_events, token_events
            else:
                # Still in block - collect content
                self._block_content += token
                return structured_events, token_events

        # Not in block - check for block start
        block_start_match = BLOCK_START_PATTERN.search(self._pending_tokens)
        if block_start_match:
            self._in_block = True
            self._block_type = block_start_match.group(1).lower()
            self._block_content = ""
            # Don't flush tokens yet - wait for block completion
            return structured_events, token_events

        # Check for Final Answer pattern (emit only once per unique content)
        final_answer_match = FINAL_ANSWER_PATTERN.search(self._token_buffer)
        if final_answer_match:
            current_content = final_answer_match.group(1).strip()
            # Only emit if content is different from last emission
            if current_content != self._last_final_answer:
                structured_events.append(
                    ServiceFinalAnswerEvent(
                        data=FinalAnswerData(content=current_content)
                    )
                )
                self._last_final_answer = current_content

        # Smart flushing: only flush if not potentially part of a pattern
        should_flush = (
            len(self._pending_tokens) > MAX_PREFIX_BUFFER
            or not self._could_be_pattern_start()
        )

        if should_flush and self._pending_tokens:
            token_events.append(ServiceTokenEvent(content=self._pending_tokens))
            self._pending_tokens = ""

        return structured_events, token_events

    def _could_be_pattern_start(self) -> bool:
        """Check if current pending tokens could be the start of a pattern."""
        if not self._pending_tokens:
            return False

        # Check if pending tokens could be start of any pattern
        pending_stripped = self._pending_tokens.strip()
        patterns_to_check = ["```", "Final Answer:"]

        for pattern in patterns_to_check:
            if pattern.startswith(pending_stripped):
                return True

        return False

    def _flush_remaining_tokens(self) -> list[ServiceTokenEvent]:
        """Flush any remaining pending tokens."""
        token_events = []
        if self._pending_tokens:
            token_events.append(ServiceTokenEvent(content=self._pending_tokens))
            self._pending_tokens = ""
        return token_events

    async def process_langchain_events(
        self, langchain_events: AsyncGenerator[dict[str, Any], None]
    ) -> AsyncGenerator[ServiceStreamEvent, None]:
        """Process LangChain astream_events and yield structured service events.

        Args:
            langchain_events: Async generator of LangChain events

        Yields:
            ServiceStreamEvent: Processed service-layer events
        """
        self.reset()

        async for event in langchain_events:
            event_type = event.get("event")

            # Stream tokens from LLM with smart buffering
            if event_type in (
                LANGCHAIN_EVENT_CHAT_MODEL_STREAM,
                LANGCHAIN_EVENT_LLM_STREAM,
            ):
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    token_content = chunk.content

                    # Process token with smart buffering
                    structured_events, token_events = self._process_token(token_content)

                    # Yield structured events first
                    for structured_event in structured_events:
                        yield structured_event

                    # Yield token events (only if not part of structured pattern)
                    for token_event in token_events:
                        yield token_event

            # Handle tool execution start
            elif event_type == LANGCHAIN_EVENT_TOOL_START:
                tool_data = event.get("data", {})
                input_data = tool_data.get(FIELD_INPUT, {})
                yield ServiceToolCallEvent(
                    data=ToolCallData(
                        tool=input_data.get(FIELD_TOOL, "unknown"),
                        input=input_data.get("tool_input", ""),
                        status="started",
                    )
                )

            # Handle tool execution completion
            elif event_type == LANGCHAIN_EVENT_TOOL_END:
                tool_data = event.get("data", {})
                output = tool_data.get(FIELD_OUTPUT, "")

                yield ServiceToolResultEvent(
                    data=ToolResultData(
                        output=output,
                        status="completed"
                    )
                )

            # Handle chain completion - only for final processing summary
            elif event_type == LANGCHAIN_EVENT_CHAIN_END:
                chain_data = event.get("data", {})
                output = chain_data.get(FIELD_OUTPUT)

                # Chain end represents the final agent output - emit as final answer if not already detected
                if output and isinstance(output, str) and output.strip():
                    # Only emit if we haven't already detected a final answer pattern
                    if not FINAL_ANSWER_PATTERN.search(self._token_buffer):
                        yield ServiceFinalAnswerEvent(
                            data=FinalAnswerData(content=output.strip())
                        )

        # Flush any remaining pending tokens at end of stream
        remaining_tokens = self._flush_remaining_tokens()
        for token_event in remaining_tokens:
            yield token_event

        # End of stream marker
        yield ServiceEndOfStreamEvent()