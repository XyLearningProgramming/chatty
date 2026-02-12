"""LLM generation utilities and factory functions."""

from typing import Annotated

from fastapi import Depends
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from chatty.configs.config import get_app_config
from chatty.configs.system import LLMConfig
from chatty.infra.concurrency.semaphore import get_model_semaphore

from .gated import GatedChatModel


def get_llm(
    config: Annotated[LLMConfig, Depends(lambda: get_app_config().llm)],
) -> ChatOpenAI:
    """Create and return a raw ChatOpenAI instance.

    Args:
        config: LLMConfig instance containing LLM configuration parameters

    Returns:
        ChatOpenAI instance configured with the provided settings
    """
    return ChatOpenAI(
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
    llm: Annotated[BaseChatModel, Depends(get_llm)],
) -> GatedChatModel:
    """Wrap the base LLM with per-invocation concurrency gating.

    The returned ``GatedChatModel`` acquires a ``ModelSemaphore`` slot
    around every ``_agenerate`` / ``_astream`` call, giving
    AI-gateway-style concurrency control on the model itself.
    """
    return GatedChatModel(inner=llm, semaphore=get_model_semaphore())
