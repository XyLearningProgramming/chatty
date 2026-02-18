"""BaseMessage converters — every message construction/deconstruction in one place.

Covers three directions:
- DB row  → BaseMessage   (history read)
- BaseMessage → DB row    (history write)
- LangChain event → BaseMessage  (callback hooks)
"""

from typing import Any
from uuid import UUID

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import LLMResult

from chatty.infra.id_utils import generate_id

from .constants import (
    DEFAULT_TOOL_NAME,
    EXTRA_MODEL_NAME,
    EXTRA_PARENT_RUN_ID,
    EXTRA_RUN_ID,
    EXTRA_TOOL_CALL_ID,
    EXTRA_TOOL_CALLS,
    EXTRA_TOOL_NAME,
    ROLE_AI,
    ROLE_HUMAN,
    ROLE_SYSTEM,
    ROLE_TOOL,
)
from .models import ChatMessage, StoredToolCall

_VALID_ROLES = (ROLE_SYSTEM, ROLE_HUMAN, ROLE_AI, ROLE_TOOL)
_EXTRA_PASSTHROUGH_KEYS = (EXTRA_RUN_ID, EXTRA_PARENT_RUN_ID, EXTRA_MODEL_NAME)


# ------------------------------------------------------------------
# Row → BaseMessage  (used by history.aget_messages)
# ------------------------------------------------------------------


def row_to_message(row: Any) -> BaseMessage | None:
    """Convert a chat_messages row to a LangChain message.

    Expected columns: (message_id, role, content, extra).
    """
    message_id: str = row[0]
    role: str = row[1]
    content: str = row[2] or ""
    extra: dict[str, Any] | None = row[3]

    if role == ROLE_HUMAN:
        return HumanMessage(content=content, id=message_id)
    if role == ROLE_AI:
        tool_calls: list[StoredToolCall] = []
        if extra and EXTRA_TOOL_CALLS in extra:
            tool_calls = [
                StoredToolCall(
                    name=tc["name"], args=tc["args"], id=tc["id"]
                )
                for tc in extra[EXTRA_TOOL_CALLS]
            ]
        return AIMessage(
            content=content, id=message_id, tool_calls=tool_calls
        )
    if role == ROLE_TOOL:
        tool_call_id = (extra or {}).get(EXTRA_TOOL_CALL_ID, "")
        tool_name = (extra or {}).get(EXTRA_TOOL_NAME, DEFAULT_TOOL_NAME)
        return ToolMessage(
            content=content,
            id=message_id,
            tool_call_id=tool_call_id,
            name=tool_name,
        )
    return None


# ------------------------------------------------------------------
# BaseMessage → Row  (used by history.aadd_messages)
# ------------------------------------------------------------------


def message_to_extra(msg: BaseMessage) -> dict[str, Any] | None:
    """Build the ``extra`` JSONB dict from a BaseMessage."""
    extra: dict[str, Any] = {}
    if hasattr(msg, "additional_kwargs") and msg.additional_kwargs:
        for key in _EXTRA_PASSTHROUGH_KEYS:
            val = msg.additional_kwargs.get(key)
            if val:
                extra[key] = val
    if isinstance(msg, AIMessage) and msg.tool_calls:
        extra[EXTRA_TOOL_CALLS] = [
            StoredToolCall(
                name=tc["name"],
                args=tc.get("args", {}),
                id=tc.get("id"),
            )
            for tc in msg.tool_calls
        ]
    if isinstance(msg, ToolMessage):
        extra[EXTRA_TOOL_NAME] = getattr(msg, "name", DEFAULT_TOOL_NAME)
        if msg.tool_call_id:
            extra[EXTRA_TOOL_CALL_ID] = msg.tool_call_id
    return extra if extra else None


def message_to_row(
    msg: BaseMessage,
) -> tuple[str, str, str | None, dict[str, Any] | None]:
    """Convert a BaseMessage to ``(message_id, role, content, extra)``."""
    message_id = getattr(msg, "id", None) or generate_id("msg")
    role = msg.type if msg.type in _VALID_ROLES else ROLE_HUMAN
    content = msg.content if isinstance(msg.content, str) else None
    extra = message_to_extra(msg)
    return (message_id, role, content, extra)


def message_to_chat_message(
    msg: BaseMessage,
    conversation_id: str,
    trace_id: str,
) -> ChatMessage:
    """Build a ChatMessage ORM instance from a BaseMessage and scope."""
    message_id, role, content, extra = message_to_row(msg)
    return ChatMessage(
        conversation_id=conversation_id,
        trace_id=trace_id,
        message_id=message_id,
        role=role,
        content=content,
        extra=extra,
    )


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------


def run_extra(
    run_id: UUID, parent_run_id: UUID | None
) -> dict[str, str]:
    """Build ``additional_kwargs`` with run lineage for a new BaseMessage."""
    extra: dict[str, str] = {EXTRA_RUN_ID: str(run_id)}
    if parent_run_id:
        extra[EXTRA_PARENT_RUN_ID] = str(parent_run_id)
    return extra


# ------------------------------------------------------------------
# LangChain event → BaseMessage  (used by callback hooks)
# ------------------------------------------------------------------


def prompt_messages_from_event(
    messages: list[Any],
    run_id: UUID,
    parent_run_id: UUID | None,
) -> list[BaseMessage]:
    """Extract system/human messages and tag with run info."""
    extra_kw = run_extra(run_id, parent_run_id)
    out: list[BaseMessage] = []
    for msg in messages:
        if not isinstance(msg, BaseMessage):
            continue
        if msg.type == ROLE_SYSTEM:
            out.append(
                SystemMessage(
                    content=msg.content or "",
                    id=msg.id or generate_id("msg"),
                    additional_kwargs=extra_kw,
                )
            )
        elif msg.type == ROLE_HUMAN:
            out.append(
                HumanMessage(
                    content=msg.content or "",
                    id=msg.id or generate_id("msg"),
                    additional_kwargs=extra_kw,
                )
            )
    return out


def ai_message_from_result(
    result: LLMResult,
    run_id: UUID,
    parent_run_id: UUID | None,
    model_name: str | None = None,
) -> AIMessage | None:
    """Extract an AIMessage from ``on_llm_end`` LLMResult.

    Returns *None* when the result contains no usable generation.
    """
    try:
        ai_msg = result.generations[0][0].message
    except (IndexError, AttributeError):
        return None

    extra_kw = run_extra(run_id, parent_run_id)
    if model_name:
        extra_kw[EXTRA_MODEL_NAME] = model_name

    return AIMessage(
        content=ai_msg.content or "",
        id=ai_msg.id or generate_id("msg"),
        tool_calls=getattr(ai_msg, "tool_calls", None) or [],
        additional_kwargs=extra_kw,
    )


def tool_name_from_serialized(serialized: dict[str, Any]) -> str:
    """Extract tool name from ``on_tool_start`` serialized payload."""
    return serialized.get("name", DEFAULT_TOOL_NAME)


def tool_message_from_output(
    output: Any,
    run_id: UUID,
    parent_run_id: UUID | None,
    tool_name: str,
) -> ToolMessage:
    """Build a ToolMessage from ``on_tool_end`` output."""
    content = (
        str(output.content) if hasattr(output, "content") else str(output)
    )
    tool_call_id = getattr(output, "tool_call_id", None) or ""

    extra_kw = run_extra(run_id, parent_run_id)
    extra_kw[EXTRA_TOOL_NAME] = tool_name
    if tool_call_id:
        extra_kw[EXTRA_TOOL_CALL_ID] = tool_call_id

    return ToolMessage(
        content=content,
        tool_call_id=tool_call_id,
        name=tool_name,
        id=generate_id("msg"),
        additional_kwargs=extra_kw,
    )
