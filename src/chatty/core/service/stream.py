"""Thin stream mapper: LLM chunks → domain StreamEvents.

Upstream (e.g. slm-server) already emits reasoning as delta.reasoning_content
and full tool_calls in delta.tool_calls. This module only maps chunk fields
to domain events (ThinkingEvent, ContentEvent, ToolCallEvent) and
accumulates the message for the tool-call loop.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_core.messages import AIMessageChunk
from pydantic import BaseModel, ConfigDict

from .models import (
    TOOL_STATUS_STARTED,
    ContentEvent,
    StreamEvent,
    ThinkingEvent,
    ToolCallEvent,
)


def normalize_tool_call(tc: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize a tool-call dict to {name, args, id} from OpenAI/slm-server shape.

    Accepts: function.name, function.arguments (JSON string), id.
    Returns None if name is missing.
    """
    fn = tc.get("function") or {}
    name = tc.get("name") or fn.get("name") or ""
    if not name:
        return None
    raw_args = tc.get("args", fn.get("arguments"))
    if isinstance(raw_args, dict):
        args = raw_args
    elif isinstance(raw_args, str) and raw_args.strip():
        try:
            args = json.loads(raw_args)
        except (json.JSONDecodeError, ValueError):
            args = {}
    else:
        args = {}
    return {"name": name, "args": args, "id": tc.get("id") or fn.get("id")}


class StreamAccumulator(BaseModel):
    """Filled by :func:`map_llm_stream` with the accumulated message (tool_calls)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    message: AIMessageChunk | None = None


def _reasoning_from_chunk(chunk: AIMessageChunk) -> str | None:
    """Reasoning from chunk (e.g. delta.reasoning_content)."""
    content = getattr(chunk, "reasoning_content", None)
    if content is not None and content != "":
        return content
    kwargs = getattr(chunk, "additional_kwargs", None) or {}
    return kwargs.get("reasoning_content")


def chunk_to_thinking_and_content(chunk: AIMessageChunk) -> Iterator[StreamEvent]:
    """Yield ThinkingEvent and ContentEvent for a chunk (e.g. RAG; no tool calls)."""
    reasoning = _reasoning_from_chunk(chunk)
    if reasoning:
        yield ThinkingEvent(content=reasoning)
    if chunk.content:
        yield ContentEvent(content=chunk.content)


async def map_llm_stream(
    chunks: AsyncIterator[AIMessageChunk],
    accumulator: StreamAccumulator | None = None,
) -> AsyncIterator[StreamEvent]:
    """Map chunks to domain events. Upstream supplies reasoning_content and
    full tool_calls."""
    accumulated: AIMessageChunk | None = None

    async for chunk in chunks:
        accumulated = chunk if accumulated is None else accumulated + chunk

        if chunk.tool_call_chunks:
            for tc in chunk.tool_call_chunks:
                n = normalize_tool_call(tc)
                if n is None:
                    continue
                yield ToolCallEvent(
                    name=n["name"],
                    status=TOOL_STATUS_STARTED,
                    arguments=n["args"] or None,
                    message_id=n.get("id"),
                )
            continue

        reasoning = _reasoning_from_chunk(chunk)
        if reasoning:
            yield ThinkingEvent(content=reasoning)

        if chunk.content:
            yield ContentEvent(content=chunk.content)

    if accumulator is not None:
        accumulator.message = accumulated
