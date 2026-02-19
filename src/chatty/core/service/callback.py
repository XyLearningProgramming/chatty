"""PG callback factory — narrow capability for message recording."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

from fastapi import Depends

from chatty.infra.db import (ChatMessageHistoryFactory,
                             get_chat_message_history_factory)
from chatty.infra.db.callback import PGMessageCallback

PgCallbackFactory = Callable[[str, str, str | None], PGMessageCallback]


def get_pg_callback_factory(
    history_factory: Annotated[
        ChatMessageHistoryFactory,
        Depends(get_chat_message_history_factory),
    ],
) -> PgCallbackFactory:
    """Return a factory for PGMessageCallback using the chat history factory."""

    def factory(
        conversation_id: str, trace_id: str, model_name: str | None = None
    ) -> PGMessageCallback:
        history = history_factory(conversation_id, trace_id=trace_id)
        return PGMessageCallback(history=history, model_name=model_name)

    return factory
