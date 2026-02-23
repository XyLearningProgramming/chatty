"""Chat service with native tool calling via OpenAI-compatible API.

Sends ``tools`` and ``tool_choice="auto"`` directly to the downstream
model server.  When the model emits tool calls, executes them via the
``ToolRegistry`` and loops until the model produces a final text answer.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from chatty.configs.config import AppConfig

from .callback import PgCallbackFactory
from .models import (
    TOOL_STATUS_COMPLETED,
    TOOL_STATUS_ERROR,
    ChatContext,
    ChatService,
    StreamEvent,
    ToolCallEvent,
)
from .stream import StreamAccumulator, map_llm_stream, normalize_tool_call
from .tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_MAX_TOOL_ROUNDS = 3
_TOOL_CHOICE_AUTO = "auto"


class OneStepChatService(ChatService):
    """Chat service that passes tools natively to the LLM.

    Each request builds a fresh tools list from the registry so that
    hot-reloaded tool definitions from the ConfigMap are picked up
    immediately.
    """

    chat_service_name = "one_step"

    def __init__(
        self,
        llm: BaseLanguageModel,
        tools_registry: ToolRegistry,
        config: AppConfig,
        pg_callback_factory: PgCallbackFactory,
    ):
        self._llm = llm
        self._tools_registry = tools_registry
        self._config = config
        self._pg_callback_factory = pg_callback_factory
        self._system_prompt = config.prompt.render_system_prompt(config.persona)

    async def stream_response(
        self, ctx: ChatContext
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream a response, handling tool calls in a loop."""
        pg_callback = self._pg_callback_factory(
            ctx.conversation_id,
            ctx.trace_id,
            self._config.llm.model_name,
        )

        tool_defs = self._tools_registry.get_tools()
        llm = self._llm
        if tool_defs:
            tool_dicts = [t.model_dump(exclude_none=True) for t in tool_defs]
            llm = llm.bind_tools(
                tool_dicts,
                tool_choice=_TOOL_CHOICE_AUTO,
            )

        messages: list = [
            SystemMessage(content=self._system_prompt),
            *ctx.history,
            HumanMessage(content=ctx.query),
        ]

        for _round in range(_MAX_TOOL_ROUNDS):
            acc = StreamAccumulator()
            async for event in map_llm_stream(
                llm.astream(messages, config={"callbacks": [pg_callback]}), acc
            ):
                yield event

            if not acc.message or not acc.message.tool_calls:
                break

            messages.append(
                AIMessage(
                    content=acc.message.content or "",
                    tool_calls=acc.message.tool_calls,
                )
            )

            for raw_tc in acc.message.tool_calls:
                tc = normalize_tool_call(raw_tc)
                if tc is None:
                    continue
                name, args, tc_id = tc["name"], tc["args"], tc.get("id") or ""
                try:
                    result = await self._tools_registry.execute(name, args)
                    yield ToolCallEvent(
                        name=name,
                        status=TOOL_STATUS_COMPLETED,
                        result=result,
                        message_id=tc_id or None,
                    )
                except Exception as exc:
                    logger.exception("Tool %s failed", name)
                    result = f"Error: {exc}"
                    yield ToolCallEvent(
                        name=name,
                        status=TOOL_STATUS_ERROR,
                        result=result,
                        message_id=tc_id or None,
                    )
                messages.append(
                    ToolMessage(content=result, tool_call_id=tc_id, name=name)
                )
