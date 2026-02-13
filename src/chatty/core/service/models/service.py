"""Abstract chat service base class."""

from abc import ABC, abstractmethod
from typing import AsyncGenerator

from .context import ChatContext
from .events import StreamEvent


class ChatService(ABC):
    """Abstract base class for chat services.

    Defines the public streaming interface.  Each concrete service
    declares its own ``__init__`` signature (OneStep needs a tool
    registry, RAG needs an embedding client, etc.).
    """

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
