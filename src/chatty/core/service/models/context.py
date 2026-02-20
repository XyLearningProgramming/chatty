"""Chat context — per-request data passed to the chat service."""

from dataclasses import dataclass, field

from langchain_core.messages import BaseMessage


@dataclass
class ChatContext:
    """Per-request context passed to the chat service.

    Carries all IDs and the loaded conversation history so the service
    doesn't need to know about the database.
    """

    query: str
    conversation_id: str
    trace_id: str
    history: list[BaseMessage] = field(default_factory=list)
