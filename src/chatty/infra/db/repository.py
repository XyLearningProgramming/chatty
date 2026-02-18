"""Conversation history loader — reads past messages from PostgreSQL.

Used by the API layer to reconstruct LangChain ``BaseMessage`` objects
from stored ``chat_messages`` rows so the agent can continue a
multi-turn conversation.
"""

import logging
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .models import (
    EXTRA_TOOL_CALL_ID,
    EXTRA_TOOL_CALLS,
    EXTRA_TOOL_NAME,
    ROLE_AI,
    ROLE_HUMAN,
    ROLE_SYSTEM,
    ROLE_TOOL,
    StoredToolCall,
)

logger = logging.getLogger(__name__)


async def load_conversation_history(
    session_factory: async_sessionmaker[AsyncSession],
    conversation_id: str,
    max_messages: int,
) -> list[BaseMessage]:
    """Load recent messages from a conversation, oldest-first.

    Args:
        session_factory: Async session maker injected by the caller.
        conversation_id: The prefixed conversation ID (``conv_xxx``).
        max_messages: Maximum number of messages to return.

    Returns:
        Ordered list of ``BaseMessage`` instances (oldest first).
    """
    query = text("""
        SELECT message_id, role, content, extra
        FROM chat_messages
        WHERE conversation_id = :cid
          AND role != :system_role
        ORDER BY created_at DESC
        LIMIT :lim
    """)

    try:
        async with session_factory() as session:
            result = await session.execute(
                query,
                {
                    "cid": conversation_id,
                    "system_role": ROLE_SYSTEM,
                    "lim": max_messages,
                },
            )
            rows = result.fetchall()
    except Exception:
        logger.warning(
            "Failed to load conversation history for %s",
            conversation_id,
            exc_info=True,
        )
        return []

    return [
        msg
        for row in reversed(rows)
        if (msg := _row_to_message(row)) is not None
    ]


def _row_to_message(row: Any) -> BaseMessage | None:
    """Convert a DB row to a LangChain message.

    Row columns: ``(message_id, role, content, extra)``.
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
        tool_name = (extra or {}).get(EXTRA_TOOL_NAME, "unknown")
        return ToolMessage(
            content=content,
            id=message_id,
            tool_call_id=tool_call_id,
            name=tool_name,
        )

    return None
