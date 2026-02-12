"""Abstract chat service base class."""

from abc import ABC, abstractmethod
from typing import AsyncGenerator

from langchain_core.language_models import BaseLanguageModel

from chatty.configs.config import AppConfig
from chatty.core.service.tools.registry import ToolRegistry

from .context import ChatContext
from .events import StreamEvent


class ChatService(ABC):
    """Abstract base class for chat services."""

    @abstractmethod
    def __init__(
        self,
        llm: BaseLanguageModel,
        tools_registry: ToolRegistry,
        config: AppConfig,
    ) -> None:
        """Initialize the chat service with required dependencies."""
        super().__init__()

    @abstractmethod
    async def stream_response(
        self, ctx: ChatContext
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream response as domain events.

        Args:
            ctx: Per-request chat context with query, IDs, and history.

        Yields:
            StreamEvent instances (thinking, content, tool_call, error)
        """
        pass
