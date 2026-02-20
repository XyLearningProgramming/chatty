"""Async stream mapper: LangGraph messages -> domain StreamEvents.

Transforms the raw (message_chunk, metadata) tuples produced by
``CompiledGraph.astream(stream_mode="messages")`` into clean domain
events (ContentEvent, ThinkingEvent, ToolCallEvent).
"""

from collections.abc import AsyncGenerator

from langchain_core.messages import AIMessageChunk, ToolMessage

from .models import (
    ContentEvent,
    StreamEvent,
    ThinkingEvent,
    ToolCallEvent,
)

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


async def map_langgraph_stream(
    raw_stream: AsyncGenerator[tuple, None],
) -> AsyncGenerator[StreamEvent, None]:
    """Map LangGraph ``stream_mode="messages"`` output to domain events.

    LangGraph yields ``(message_chunk, metadata)`` tuples where:
    - ``AIMessageChunk`` with ``.content`` → text tokens
    - ``AIMessageChunk`` with ``.tool_calls`` → tool invocation start
    - ``ToolMessage`` → tool result (completed / error)

    The *metadata* dict carries ``langgraph_node`` which tells us whether
    the chunk originates from the LLM node or the ``"tools"`` node
    (tool execution).

    Inline ``<think>...</think>`` blocks (e.g. Qwen3 reasoning) are
    automatically reclassified from ``ContentEvent`` to ``ThinkingEvent``
    so downstream consumers never see raw think tags.

    Args:
        raw_stream: async iterable of (BaseMessageChunk, metadata) from
            ``graph.astream(..., stream_mode="messages")``.

    Yields:
        Domain ``StreamEvent`` instances.
    """
    in_thinking = False
    strip_leading_newlines = False
    async for chunk, metadata in raw_stream:
        event = _map_chunk(chunk, metadata)
        if event is None:
            continue

        if not isinstance(event, ContentEvent):
            yield event
            continue

        was_thinking = in_thinking
        events, in_thinking = _split_thinking(event.content, in_thinking)
        if was_thinking and not in_thinking:
            strip_leading_newlines = True

        for sub_event in events:
            if strip_leading_newlines and isinstance(sub_event, ContentEvent):
                stripped = sub_event.content.lstrip("\n")
                if not stripped:
                    continue
                sub_event = ContentEvent(content=stripped)
            strip_leading_newlines = False
            yield sub_event


def _split_thinking(
    text: str, in_thinking: bool
) -> tuple[list[ThinkingEvent | ContentEvent], bool]:
    """Split *text* at ``<think>`` / ``</think>`` boundaries.

    Returns a list of ``ThinkingEvent`` / ``ContentEvent`` fragments
    (tags stripped, empty fragments dropped) **and** the resulting
    ``in_thinking`` state.  Returning state explicitly ensures the
    caller tracks mode transitions even when a tag-only chunk produces
    no output events.
    """
    events: list[ThinkingEvent | ContentEvent] = []
    while text:
        if in_thinking:
            idx = text.find(_THINK_CLOSE)
            if idx == -1:
                events.append(ThinkingEvent(content=text))
                return events, True
            before = text[:idx]
            if before:
                events.append(ThinkingEvent(content=before))
            text = text[idx + len(_THINK_CLOSE) :].lstrip("\n")
            in_thinking = False
        else:
            idx = text.find(_THINK_OPEN)
            if idx == -1:
                events.append(ContentEvent(content=text))
                return events, False
            before = text[:idx]
            if before:
                events.append(ContentEvent(content=before))
            text = text[idx + len(_THINK_OPEN) :]
            in_thinking = True
    return events, in_thinking


def _map_chunk(chunk, metadata: dict) -> StreamEvent | None:
    """Map a single (chunk, metadata) pair to a domain event or *None*."""
    node = metadata.get("langgraph_node", "")

    # --- AIMessageChunk from the LLM ---
    if isinstance(chunk, AIMessageChunk):
        if chunk.tool_call_chunks:
            return _map_tool_call_start(chunk)

        content = chunk.content
        if not content:
            return None

        # LLM-node text is initially tagged as ContentEvent;
        # _split_thinking in the caller reclassifies think-tagged
        # fragments to ThinkingEvent.
        if node in ("agent", "model"):
            return ContentEvent(content=content)
        return ThinkingEvent(content=content)

    # --- ToolMessage from tool execution ---
    if isinstance(chunk, ToolMessage):
        return _map_tool_result(chunk)

    return None


def _map_tool_call_start(chunk: AIMessageChunk) -> ToolCallEvent | None:
    """Extract the first tool-call chunk and emit a started event."""
    tc = chunk.tool_call_chunks[0]
    name = tc.get("name") or ""
    args = tc.get("args")

    if not name:
        return None

    arguments: dict | None = None
    if isinstance(args, dict):
        arguments = args
    elif isinstance(args, str) and args.strip():
        import json

        try:
            arguments = json.loads(args)
        except (json.JSONDecodeError, ValueError):
            arguments = None

    return ToolCallEvent(name=name, status="started", arguments=arguments)


def _map_tool_result(msg: ToolMessage) -> ToolCallEvent:
    """Map a ToolMessage to a completed / error ToolCallEvent."""
    is_error = msg.status == "error" if hasattr(msg, "status") else False
    return ToolCallEvent(
        name=msg.name or "unknown",
        status="error" if is_error else "completed",
        result=str(msg.content) if msg.content else None,
    )
