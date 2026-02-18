"""LLM client object as BaseLanguageModel in langchain."""

from .deps import get_gated_llm, get_llm  # noqa: F401
from .gated import GatedChatModel  # noqa: F401
