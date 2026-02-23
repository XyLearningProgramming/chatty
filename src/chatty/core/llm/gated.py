"""GatedChatModel — per-invocation concurrency gating around a chat model.

Wraps any ``BaseChatModel`` (e.g. ``ChatOpenAI``) so that every
``_agenerate`` / ``_astream`` call acquires a semaphore slot first.
Between tool-call rounds the slot is released, letting other requests
use the model — AI-gateway-style concurrency control.

Messages are trimmed to fit ``context_window - max_tokens`` before
every call.  If even the system prompt + user query exceed the budget,
a ``PromptBudgetExceeded`` exception is raised (caught by the API
layer as HTTP 400).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from pydantic import PrivateAttr

from chatty.core.service.metrics import (
    LLM_CALLS_IN_FLIGHT,
    LLM_INPUT_TOKENS,
    LLM_PROMPT_TRIMMED_TOTAL,
)
from chatty.infra.concurrency.semaphore import ModelSemaphore
from chatty.infra.tokens import estimate_tokens

logger = logging.getLogger(__name__)


class PromptBudgetExceeded(Exception):
    """System prompt + query alone exceed the context budget."""


class GatedChatModel(BaseChatModel):
    """Chat model wrapper that gates every invocation behind a semaphore.

    The wrapper transparently delegates to an *inner* chat model,
    adding ``ModelSemaphore.slot()`` around each ``_agenerate`` /
    ``_astream`` call.  ``bind_tools`` is forwarded so that OpenAI
    tool-formatting logic (on the inner model) is preserved, but the
    resulting ``RunnableBinding`` routes back through the gate.

    Messages are trimmed to fit the token budget before every call.
    """

    inner: BaseChatModel
    model_name: str
    max_tokens: int
    context_window: int
    _semaphore: ModelSemaphore = PrivateAttr()

    def __init__(self, *, semaphore: ModelSemaphore, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._semaphore = semaphore

    # ------------------------------------------------------------------
    # Token-budget message trimming
    # ------------------------------------------------------------------

    def _trim_messages(
        self, messages: list[BaseMessage]
    ) -> list[BaseMessage]:
        """Trim *messages* to fit ``context_window - max_tokens``.

        Strategy:
        1. System message(s) and the last user message are non-negotiable.
        2. If they alone exceed the budget → raise ``PromptBudgetExceeded``.
        3. Otherwise, keep as many middle (history) messages as fit,
           dropping the oldest first.
        """
        input_budget = self.context_window - self.max_tokens

        system_msgs: list[BaseMessage] = []
        history_msgs: list[BaseMessage] = []
        last_msg: BaseMessage | None = None

        for i, msg in enumerate(messages):
            if isinstance(msg, SystemMessage):
                system_msgs.append(msg)
            elif i == len(messages) - 1:
                last_msg = msg
            else:
                history_msgs.append(msg)

        fixed_cost = sum(
            estimate_tokens(m.content or "") for m in system_msgs
        )
        if last_msg is not None:
            fixed_cost += estimate_tokens(last_msg.content or "")

        if fixed_cost > input_budget:
            raise PromptBudgetExceeded(
                f"System prompt + query require ~{fixed_cost} tokens "
                f"but input budget is {input_budget} "
                f"(context_window={self.context_window}, "
                f"max_tokens={self.max_tokens}). "
                f"Shorten the query or increase the context window."
            )

        remaining = input_budget - fixed_cost
        kept: list[BaseMessage] = []
        original_count = len(history_msgs)

        for msg in reversed(history_msgs):
            cost = estimate_tokens(msg.content or "")
            if remaining >= cost:
                kept.append(msg)
                remaining -= cost
            else:
                break
        kept.reverse()

        trimmed_count = original_count - len(kept)
        if trimmed_count > 0:
            logger.warning(
                "Trimmed %d history message(s) to fit context budget "
                "(budget=%d, context_window=%d, max_tokens=%d)",
                trimmed_count,
                input_budget,
                self.context_window,
                self.max_tokens,
            )
            LLM_PROMPT_TRIMMED_TOTAL.labels(model_name=self.model_name).inc()

        result = system_msgs + kept + ([last_msg] if last_msg else [])

        total_tokens = sum(
            estimate_tokens(m.content or "") for m in result
        )
        LLM_INPUT_TOKENS.labels(model_name=self.model_name).observe(total_tokens)

        return result

    # ------------------------------------------------------------------
    # Required abstract implementations
    # ------------------------------------------------------------------

    @property
    def _llm_type(self) -> str:  # type: ignore[override]
        return f"gated-{self.inner._llm_type}"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Sync path — delegate without async gating.

        The production code-path is fully async (``_agenerate`` /
        ``_astream``), so the sync path simply forwards to the inner
        model without acquiring the semaphore.
        """
        return self.inner._generate(messages, stop, run_manager, **kwargs)

    # ------------------------------------------------------------------
    # Async gated paths
    # ------------------------------------------------------------------

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        messages = self._trim_messages(messages)
        async with self._semaphore.slot():
            LLM_CALLS_IN_FLIGHT.labels(model_name=self.model_name).inc()
            try:
                return await self.inner._agenerate(
                    messages, stop, run_manager, **kwargs
                )
            finally:
                LLM_CALLS_IN_FLIGHT.labels(model_name=self.model_name).dec()

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        messages = self._trim_messages(messages)
        async with self._semaphore.slot():
            LLM_CALLS_IN_FLIGHT.labels(model_name=self.model_name).inc()
            try:
                async for chunk in self.inner._astream(
                    messages, stop, run_manager, **kwargs
                ):
                    yield chunk
            finally:
                LLM_CALLS_IN_FLIGHT.labels(model_name=self.model_name).dec()

    # ------------------------------------------------------------------
    # Tool binding — delegate formatting to inner, re-bind to self
    # ------------------------------------------------------------------

    def bind_tools(
        self,
        tools: Any,
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]:
        """Format tools via the inner model, then bind to *self*.

        ``inner.bind_tools(tools)`` produces a ``RunnableBinding`` whose
        ``.kwargs`` contain the provider-formatted tool definitions.
        Re-binding those kwargs onto *self* ensures that subsequent
        ``ainvoke`` / ``astream`` calls still route through the gated
        ``_agenerate`` / ``_astream``.
        """
        inner_bound = self.inner.bind_tools(tools, **kwargs)
        return self.bind(**inner_bound.kwargs)
