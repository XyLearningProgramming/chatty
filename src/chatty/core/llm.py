"""LLM generation utilities and factory functions."""

from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI

from chatty.configs.system import ChatConfig


def get_llm(config: ChatConfig) -> BaseLanguageModel:
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
        streaming=True,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
        top_p=config.top_p,
        max_retries=config.max_retries,
    )
