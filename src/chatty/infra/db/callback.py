"""LangChain async callback handler that records messages to PostgreSQL.

Uses a LangChain BaseChatMessageHistory (e.g. PgChatMessageHistory) so
that load and write go through the same abstraction. Message construction
is delegated to converters; this module is pure orchestration + error
handling.

Usage::

    history = chat_message_history_factory(conversation_id, trace_id)
    callback = PGMessageCallback(history=history, model_name="gpt-3.5-turbo")
    raw_stream = graph.astream(..., config={"callbacks": [callback]})
"""

import logging
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.outputs import LLMResult

from .constants import ROLE_HUMAN, ROLE_SYSTEM
from .converters import (
    ai_message_from_result,
    prompt_messages_from_event,
    tool_message_from_output,
    tool_name_from_serialized,
)

logger = logging.getLogger(__name__)


class PGMessageCallback(AsyncCallbackHandler):
    """Records every LangChain message to PostgreSQL via BaseChatMessageHistory.

    Created per-request with a history instance (e.g. from
    get_chat_message_history_factory(conversation_id, trace_id)).
    """

    def __init__(
        self,
        history: BaseChatMessageHistory,
        model_name: str | None = None,
    ) -> None:
        super().__init__()
        self._history = history
        self.model_name = model_name
        self._initial_saved = False
        self._tool_names: dict[UUID, str] = {}

    # -- system + human (once) -------------------------------------------

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Store the system prompt and human message on first LLM call.

        Only the last system and last human message are persisted, so we do
        not re-insert messages already loaded from history (which would cause
        duplicate key errors).
        """
        if self._initial_saved:
            return

        all_prompt = prompt_messages_from_event(messages[0], run_id, parent_run_id)
        last_system = next(
            (m for m in reversed(all_prompt) if getattr(m, "type", None) == ROLE_SYSTEM),
            None,
        )
        last_human = next(
            (m for m in reversed(all_prompt) if getattr(m, "type", None) == ROLE_HUMAN),
            None,
        )
        to_add = [m for m in (last_system, last_human) if m is not None]
        if to_add:
            try:
                await self._history.aadd_messages(to_add)
            except Exception:
                logger.warning(
                    "Failed to record prompt messages for run %s",
                    run_id,
                    exc_info=True,
                )
        self._initial_saved = True

    # -- AI message ------------------------------------------------------

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Store the AI message (tool-call decision or final answer)."""
        msg = ai_message_from_result(response, run_id, parent_run_id, self.model_name)
        if msg is None:
            logger.debug("on_llm_end: could not extract AI message, skipping.")
            return
        try:
            await self._history.aadd_messages([msg])
        except Exception:
            logger.warning(
                "Failed to record AI message for run %s",
                run_id,
                exc_info=True,
            )

    # -- tool messages ---------------------------------------------------

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Track the tool name so ``on_tool_end`` can reference it."""
        self._tool_names[run_id] = tool_name_from_serialized(serialized)

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Store the tool result message."""
        name = self._tool_names.pop(run_id, tool_name_from_serialized({}))
        msg = tool_message_from_output(output, run_id, parent_run_id, name)
        try:
            await self._history.aadd_messages([msg])
        except Exception:
            logger.warning(
                "Failed to record tool message for run %s",
                run_id,
                exc_info=True,
            )
