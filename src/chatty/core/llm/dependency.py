"""LLM generation utilities and factory functions."""

from typing import Annotated

from fastapi import Depends
from langchain_openai import ChatOpenAI

from chatty.configs.config import get_app_config
from chatty.configs.system import LLMConfig


def get_llm(
    config: Annotated[LLMConfig, Depends(lambda: get_app_config().llm)],
) -> ChatOpenAI:
    """Create and return a ChatOpenAI instance with the provided configuration.

    Args:
        config: ChatConfig instance containing LLM configuration parameters

    Returns:
        ChatOpenAI instance configured with the provided settings
    """
    return ChatOpenAI(
        base_url=config.endpoint,
        api_key=config.api_key,
        model=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
        top_p=config.top_p,
        max_retries=config.max_retries,
        streaming=True,
        verbose=True,
    )
