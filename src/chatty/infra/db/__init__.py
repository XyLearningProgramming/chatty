"""Async PostgreSQL infrastructure (engine builder, ORM models)."""

from .cache import CacheRepository
from .embedding import EmbeddingRepository
from .deps import (get_cache_repository, get_chat_message_history_factory,
                   get_embedding_repository)
from .history import ChatMessageHistoryFactory, PgChatMessageHistory
from .models import (ROLE_AI, ROLE_HUMAN, ROLE_SYSTEM, ROLE_TOOL, AIExtra,
                     Base, ChatMessage, MessageExtra, PromptExtra,
                     Role, SourceEmbedding, StoredToolCall, ToolExtra)

from chatty.infra.db_engine import (build_db, get_async_session,
                                    get_session_factory)

__all__ = [
    "build_db",
    "CacheRepository",
    "ChatMessageHistoryFactory",
    "EmbeddingRepository",
    "get_async_session",
    "get_cache_repository",
    "get_chat_message_history_factory",
    "get_embedding_repository",
    "get_session_factory",
    "PgChatMessageHistory",
    "AIExtra",
    "Base",
    "ChatMessage",
    "MessageExtra",
    "PromptExtra",
    "Role",
    "ROLE_AI",
    "ROLE_HUMAN",
    "ROLE_SYSTEM",
    "ROLE_TOOL",
    "SourceEmbedding",
    "StoredToolCall",
    "ToolExtra",
]
