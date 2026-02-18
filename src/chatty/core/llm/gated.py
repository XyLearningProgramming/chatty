"""GatedChatModel — per-invocation concurrency gating around a chat model.

Wraps any ``BaseChatModel`` (e.g. ``ChatOpenAI``) so that every
``_agenerate`` / ``_astream`` call acquires a semaphore slot first.
Between tool-call rounds the slot is released, letting other requests
use the model — AI-gateway-style concurrency control.
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
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from pydantic import ConfigDict, Field

from chatty.infra.concurrency.semaphore import ModelSemaphore

logger = logging.getLogger(__name__)


class GatedChatModel(BaseChatModel):
    """Chat model wrapper that gates every invocation behind a semaphore.

    The wrapper transparently delegates to an *inner* chat model,
    adding ``ModelSemaphore.slot()`` around each ``_agenerate`` /
    ``_astream`` call.  ``bind_tools`` is forwarded so that OpenAI
    tool-formatting logic (on the inner model) is preserved, but the
    resulting ``RunnableBinding`` routes back through the gate.
    """

    inner: BaseChatModel
    semaphore: ModelSemaphore = Field(exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

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
        async with self.semaphore.slot():
            logger.debug("LLM generate: acquired semaphore slot")
            return await self.inner._agenerate(
                messages, stop, run_manager, **kwargs
            )

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        async with self.semaphore.slot():
            logger.debug("LLM stream: acquired semaphore slot")
            async for chunk in self.inner._astream(
                messages, stop, run_manager, **kwargs
            ):
                yield chunk

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
