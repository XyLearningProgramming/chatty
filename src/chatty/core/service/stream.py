"""Async stream mapper: LangGraph messages -> domain StreamEvents.

Transforms the raw (message_chunk, metadata) tuples produced by
``CompiledGraph.astream(stream_mode="messages")`` into clean domain
events (ContentEvent, ThinkingEvent, ToolCallEvent).
"""

from typing import AsyncGenerator

from langchain_core.messages import AIMessageChunk, ToolMessage

from .models import (
    ContentEvent,
    StreamEvent,
    ThinkingEvent,
    ToolCallEvent,
)


async def map_langgraph_stream(
    raw_stream: AsyncGenerator[tuple, None],
) -> AsyncGenerator[StreamEvent, None]:
    """Map LangGraph ``stream_mode="messages"`` output to domain events.

    LangGraph yields ``(message_chunk, metadata)`` tuples where:
    - ``AIMessageChunk`` with ``.content`` → text tokens
    - ``AIMessageChunk`` with ``.tool_calls`` → tool invocation start
    - ``ToolMessage`` → tool result (completed / error)

    The *metadata* dict carries ``langgraph_node`` which tells us whether
    the chunk originates from the ``"agent"`` node (LLM) or the
    ``"tools"`` node (tool execution).

    Args:
        raw_stream: async iterable of (BaseMessageChunk, metadata) from
            ``graph.astream(..., stream_mode="messages")``.

    Yields:
        Domain ``StreamEvent`` instances.
    """
    async for chunk, metadata in raw_stream:
        event = _map_chunk(chunk, metadata)
        if event is not None:
            yield event


def _map_chunk(chunk, metadata: dict) -> StreamEvent | None:
    """Map a single (chunk, metadata) pair to a domain event or *None*."""
    node = metadata.get("langgraph_node", "")

    # --- AIMessageChunk from the LLM ---
    if isinstance(chunk, AIMessageChunk):
        # Tool call initiation (function-calling)
        if chunk.tool_call_chunks:
            return _map_tool_call_start(chunk)

        # Text content
        content = chunk.content
        if not content:
            return None

        # Text from the "agent" node is the final user-facing answer;
        # text from any other node is intermediate thinking.
        if node == "agent":
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

    # LangGraph streams tool-call arguments incrementally; only emit on
    # the first chunk that carries the tool *name*.
    if not name:
        return None

    # ``args`` may be a string (partial JSON) or a dict; try to normalise.
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
