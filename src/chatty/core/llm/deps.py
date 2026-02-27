"""LLM generation utilities and factory functions."""

import logging
from typing import Annotated

from fastapi import Depends
from langchain_core.language_models import BaseChatModel
from chatty.configs.config import get_llm_config
from chatty.configs.system import LLMConfig
from chatty.infra.concurrency.semaphore import ModelSemaphore, get_model_semaphore

from .gated import GatedChatModel
from .no_think import QwenNoThinkChatModel
from .reasoning import ReasoningChatOpenAI

logger = logging.getLogger(__name__)


def get_llm(
    config: Annotated[LLMConfig, Depends(get_llm_config)],
) -> ReasoningChatOpenAI:
    """Create and return a ChatOpenAI instance that preserves reasoning_content."""
    return ReasoningChatOpenAI(
        base_url=config.endpoint,
        api_key=config.api_key,
        model=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.model_timeout.total_seconds(),
        top_p=config.top_p,
        max_retries=config.max_retries,
        streaming=True,
        verbose=True,
    )


def get_gated_llm(
    config: Annotated[LLMConfig, Depends(get_llm_config)],
    llm: Annotated[BaseChatModel, Depends(get_llm)],
    semaphore: Annotated[ModelSemaphore, Depends(get_model_semaphore)],
) -> GatedChatModel:
    """Wrap the base LLM with per-invocation concurrency gating.

    The returned ``GatedChatModel`` acquires a ``ModelSemaphore`` slot
    around every ``_agenerate`` / ``_astream`` call, giving
    AI-gateway-style concurrency control on the model itself.
    """
    return GatedChatModel(
        inner=llm,
        semaphore=semaphore,
        model_name=config.model_name or "unknown",
        max_tokens=config.max_tokens,
        context_window=config.context_window,
    )


def get_no_think_llm(
    config: Annotated[LLMConfig, Depends(get_llm_config)],
    gated_llm: Annotated[GatedChatModel, Depends(get_gated_llm)],
) -> BaseChatModel:
    """Wrap the gated LLM with a no-think directive for supported models.

    Only Qwen models support the ``/no_think`` suffix.  For other models
    the gated LLM is returned as-is so the rest of the pipeline still
    works without model-specific assumptions.
    """
    model_name = (config.model_name or "").lower()
    if "qwen" not in model_name:
        logger.warning(
            "NoThinkChatModel only supports Qwen models; "
            "model_name=%r does not contain 'qwen'. "
            "Falling back to the gated model without /no_think injection.",
            config.model_name,
        )
        return gated_llm
    return QwenNoThinkChatModel(inner=gated_llm)
