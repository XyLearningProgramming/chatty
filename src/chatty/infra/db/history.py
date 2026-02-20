"""LangChain BaseChatMessageHistory implementation backed by PostgreSQL.

Provides load (aget_messages) and write (aadd_messages) over the
existing chat_messages table. Used by the API for conversation
history and by the PG callback for message recording.
"""

import logging
from collections.abc import Callable

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from chatty.infra.telemetry import (
    ATTR_HISTORY_CONVERSATION_ID,
    ATTR_HISTORY_MESSAGE_COUNT,
    SPAN_HISTORY_LOAD,
    tracer,
)

from .constants import (
    DEFAULT_MAX_MESSAGES,
    PARAM_CID,
    PARAM_LIM,
    PARAM_SYSTEM_ROLE,
    ROLE_SYSTEM,
    SQL_DELETE_MESSAGES,
    SQL_SELECT_MESSAGES,
)
from .converters import message_to_chat_message, row_to_message

logger = logging.getLogger(__name__)

ChatMessageHistoryFactory = Callable[
    [str, str | None, int | None],
    BaseChatMessageHistory,
]


class PgChatMessageHistory(BaseChatMessageHistory):
    """PostgreSQL-backed chat message history (LangChain compatible).

    Scoped to a conversation_id; trace_id is required for writes
    (aadd_messages). max_messages limits the read window when set.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        conversation_id: str,
        trace_id: str | None = None,
        max_messages: int | None = None,
    ) -> None:
        super().__init__()
        self._session_factory = session_factory
        self.conversation_id = conversation_id
        self.trace_id = trace_id
        self._max_messages = max_messages

    async def aget_messages(self) -> list[BaseMessage]:
        """Load recent messages for this conversation, oldest-first."""
        with tracer.start_as_current_span(SPAN_HISTORY_LOAD) as span:
            span.set_attribute(ATTR_HISTORY_CONVERSATION_ID, self.conversation_id)
            lim = self._max_messages or DEFAULT_MAX_MESSAGES
            try:
                async with self._session_factory() as session:
                    result = await session.execute(
                        text(SQL_SELECT_MESSAGES),
                        {
                            PARAM_CID: self.conversation_id,
                            PARAM_SYSTEM_ROLE: ROLE_SYSTEM,
                            PARAM_LIM: lim,
                        },
                    )
                    rows = result.fetchall()
            except Exception:
                logger.warning(
                    "Failed to load conversation history for %s",
                    self.conversation_id,
                    exc_info=True,
                )
                return []
            messages = [
                m for row in reversed(rows) if (m := row_to_message(row)) is not None
            ]
            span.set_attribute(ATTR_HISTORY_MESSAGE_COUNT, len(messages))
            logger.debug(
                "Loaded %d messages for conversation %s",
                len(messages),
                self.conversation_id,
            )
            return messages

    async def aadd_messages(self, messages: list[BaseMessage]) -> None:
        """Append messages; requires trace_id. Fire-and-forget (log on error)."""
        if not self.trace_id:
            raise ValueError("trace_id is required for aadd_messages")
        for msg in messages:
            chat_msg = message_to_chat_message(msg, self.conversation_id, self.trace_id)
            try:
                async with self._session_factory() as session:
                    session.add(chat_msg)
                    await session.commit()
            except Exception:
                logger.warning(
                    "Failed to record %s message for trace %s",
                    chat_msg.role,
                    self.trace_id,
                    exc_info=True,
                )

    def clear(self) -> None:
        """Sync clear is not supported; use aclear() instead."""
        raise NotImplementedError("Use aclear() for async history clearing")

    async def aclear(self) -> None:
        """Remove all messages for this conversation."""
        try:
            async with self._session_factory() as session:
                await session.execute(
                    text(SQL_DELETE_MESSAGES),
                    {PARAM_CID: self.conversation_id},
                )
                await session.commit()
        except Exception:
            logger.warning(
                "Failed to clear history for %s",
                self.conversation_id,
                exc_info=True,
            )
