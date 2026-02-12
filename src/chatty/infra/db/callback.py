"""LangChain async callback handler that records messages to PostgreSQL.

Each callback method performs a single fire-and-forget INSERT.  All
database errors are caught and logged so that recording never
interrupts the chat stream.

IDs use the ``{prefix}_{random}`` convention for messages we create
(``msg_xxx``), while provider-generated IDs (OpenAI ``chatcmpl-xxx``,
``call_xxx``) are used as-is.  LangChain framework IDs (``run_id``,
``parent_run_id``) are stored in ``extra`` JSONB for debugging
lineage.

Usage::

    callback = PGMessageCallback(
        conversation_id="conv_a8Kx3nQ9mP2r",
        trace_id="trace_L7wBd4Fj9Ks2",
        model_name="gpt-3.5-turbo",
    )
    raw_stream = graph.astream(
        {"messages": [("user", question)]},
        stream_mode="messages",
        config={"callbacks": [callback]},
    )
"""

import logging
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.outputs import LLMResult

from chatty.infra.id_utils import generate_id

from .engine import get_async_session_factory
from .models import (
    AIExtra,
    EXTRA_MODEL_NAME,
    EXTRA_PARENT_RUN_ID,
    EXTRA_RUN_ID,
    EXTRA_TOOL_CALL_ID,
    EXTRA_TOOL_CALLS,
    EXTRA_TOOL_NAME,
    ChatMessage,
    PromptExtra,
    Role,
    ROLE_AI,
    ROLE_HUMAN,
    ROLE_SYSTEM,
    ROLE_TOOL,
    StoredToolCall,
    ToolExtra,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fire-and-forget INSERT helper
# ---------------------------------------------------------------------------


async def _insert(
    *,
    conversation_id: str,
    trace_id: str,
    message_id: str,
    role: Role,
    content: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Insert a single ``ChatMessage`` row.

    Acquires a session from the singleton factory, commits, and
    closes.  All exceptions are caught and logged — the caller is
    never interrupted.
    """
    try:
        session_factory = get_async_session_factory()
        async with session_factory() as session:
            msg = ChatMessage(
                conversation_id=conversation_id,
                trace_id=trace_id,
                message_id=message_id,
                role=role,
                content=content,
                extra=extra,
            )
            session.add(msg)
            await session.commit()
    except Exception:
        logger.warning(
            "Failed to record %s message for trace %s",
            role,
            trace_id,
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prompt_extra(
    run_id: UUID, parent_run_id: UUID | None
) -> PromptExtra:
    """Build the ``extra`` dict for system / human messages."""
    extra = PromptExtra(**{EXTRA_RUN_ID: str(run_id)})
    if parent_run_id:
        extra[EXTRA_PARENT_RUN_ID] = str(parent_run_id)
    return extra


# ---------------------------------------------------------------------------
# Callback handler
# ---------------------------------------------------------------------------


class PGMessageCallback(AsyncCallbackHandler):
    """Records every LangChain message to PostgreSQL.

    Created per-request and passed to
    ``graph.astream(config={"callbacks": [cb]})``.

    Hooks used:

    - ``on_chat_model_start`` — stores the system prompt and human
      message (once, on the first LLM call in a ReAct loop).
    - ``on_llm_end`` — stores the AI message (tool-call or final
      answer).
    - ``on_tool_start`` / ``on_tool_end`` — stores tool result
      messages.
    """

    def __init__(
        self,
        conversation_id: str,
        trace_id: str,
        model_name: str | None = None,
    ) -> None:
        super().__init__()
        self.conversation_id = conversation_id
        self.trace_id = trace_id
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
        """Store the system prompt and human message on the first LLM call."""
        if self._initial_saved:
            return

        for msg in messages[0]:
            if not isinstance(msg, BaseMessage):
                continue

            role: Role | None = None
            if msg.type == ROLE_SYSTEM:
                role = ROLE_SYSTEM
            elif msg.type == ROLE_HUMAN:
                role = ROLE_HUMAN

            if role is None:
                continue

            msg_id = msg.id or generate_id("msg")
            extra = _make_prompt_extra(run_id, parent_run_id)

            await _insert(
                conversation_id=self.conversation_id,
                trace_id=self.trace_id,
                message_id=msg_id,
                role=role,
                content=msg.content if isinstance(msg.content, str) else None,
                extra=extra,
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
        """Store the AI message (tool-call decision or final answer).

        Uses ``ai_msg.id`` from the provider (OpenAI ``chatcmpl-xxx``)
        as the ``message_id``.
        """
        try:
            ai_msg = response.generations[0][0].message
        except (IndexError, AttributeError):
            logger.debug("on_llm_end: could not extract AI message, skipping.")
            return

        msg_id = ai_msg.id or generate_id("msg")

        extra = AIExtra(**{EXTRA_RUN_ID: str(run_id)})
        if parent_run_id:
            extra[EXTRA_PARENT_RUN_ID] = str(parent_run_id)
        if self.model_name:
            extra[EXTRA_MODEL_NAME] = self.model_name
        if ai_msg.tool_calls:
            extra[EXTRA_TOOL_CALLS] = [
                StoredToolCall(name=tc["name"], args=tc["args"], id=tc["id"])
                for tc in ai_msg.tool_calls
            ]

        content = ai_msg.content
        await _insert(
            conversation_id=self.conversation_id,
            trace_id=self.trace_id,
            message_id=msg_id,
            role=ROLE_AI,
            content=content if isinstance(content, str) and content else None,
            extra=extra,
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
        self._tool_names[run_id] = serialized.get("name", "unknown")

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Store the tool result message."""
        name = self._tool_names.pop(run_id, "unknown")
        content = str(output.content) if hasattr(output, "content") else str(output)

        msg_id = generate_id("msg")

        extra = ToolExtra(
            **{EXTRA_RUN_ID: str(run_id)},
            **{EXTRA_TOOL_NAME: name},
        )
        if parent_run_id:
            extra[EXTRA_PARENT_RUN_ID] = str(parent_run_id)
        if isinstance(output, ToolMessage):
            extra[EXTRA_TOOL_CALL_ID] = output.tool_call_id

        await _insert(
            conversation_id=self.conversation_id,
            trace_id=self.trace_id,
            message_id=msg_id,
            role=ROLE_TOOL,
            content=content,
            extra=extra,
        )
