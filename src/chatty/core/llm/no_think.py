"""NoThinkChatModel — appends a no-think directive to the last user message.

Wraps any ``BaseChatModel`` so that every call automatically appends a
model-specific suffix (default: Qwen3 ``/no_think``) to the last
``HumanMessage``.  The RAG graph signals *whether* to suppress
thinking; this wrapper owns the *how*.
"""

from __future__ import annotations

import copy
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable

_DEFAULT_SUFFIX = " /no_think"


class QwenNoThinkChatModel(BaseChatModel):
    """Chat model wrapper that injects a no-think directive.

    Appends ``suffix`` to the content of the last ``HumanMessage``
    before delegating to *inner*.  Everything else (trimming,
    concurrency gating, tool binding) is handled by the inner model
    chain — this wrapper is intentionally minimal.
    """

    inner: BaseChatModel
    suffix: str = _DEFAULT_SUFFIX

    @property
    def _llm_type(self) -> str:  # type: ignore[override]
        return f"no-think-{self.inner._llm_type}"

    # ------------------------------------------------------------------
    # Message injection
    # ------------------------------------------------------------------

    def _inject(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """Return a shallow copy with the suffix appended to the last HumanMessage."""
        if not messages:
            return messages

        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):
                patched = copy.copy(messages[i])
                patched.content = (patched.content or "") + self.suffix
                return messages[:i] + [patched] + messages[i + 1 :]

        return messages

    # ------------------------------------------------------------------
    # Required abstract implementations
    # ------------------------------------------------------------------

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self.inner._generate(
            self._inject(messages), stop, run_manager, **kwargs
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return await self.inner._agenerate(
            self._inject(messages), stop, run_manager, **kwargs
        )

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        async for chunk in self.inner._astream(
            self._inject(messages), stop, run_manager, **kwargs
        ):
            yield chunk

    # ------------------------------------------------------------------
    # Delegation
    # ------------------------------------------------------------------

    def bind_tools(
        self,
        tools: Any,
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]:
        """Format tools via the inner model, then bind to *self*.

        Same pattern as ``GatedChatModel.bind_tools``: the inner model
        formats the tool definitions, and re-binding those kwargs onto
        *self* ensures subsequent calls still route through the
        no-think injection.
        """
        inner_bound = self.inner.bind_tools(tools, **kwargs)
        return self.bind(**inner_bound.kwargs)
