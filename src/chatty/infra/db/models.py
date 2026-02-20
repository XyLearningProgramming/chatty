"""SQLAlchemy ORM models for the Chatty application.

All tables are managed by Alembic migrations.  The ``Base.metadata``
naming convention ensures deterministic constraint names for
auto-generated migrations.
"""

from typing import Any, Literal

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, DateTime, Index, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from typing_extensions import NotRequired, TypedDict

# ---------------------------------------------------------------------------
# Declarative base with naming convention
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """Shared declarative base with explicit naming convention.

    The naming convention ensures that auto-generated constraint names
    are deterministic across environments, which is required for
    reliable Alembic ``--autogenerate`` diffs.
    """

    metadata_naming_convention = {
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    }

    # Apply the naming convention to the metadata
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)


# Apply naming convention to Base metadata
Base.metadata.naming_convention = Base.metadata_naming_convention


# ---------------------------------------------------------------------------
# Role constants & type
# ---------------------------------------------------------------------------

ROLE_SYSTEM: Literal["system"] = "system"
ROLE_HUMAN: Literal["human"] = "human"
ROLE_AI: Literal["ai"] = "ai"
ROLE_TOOL: Literal["tool"] = "tool"

Role = Literal["system", "human", "ai", "tool"]


# ---------------------------------------------------------------------------
# Extra JSONB key constants
# ---------------------------------------------------------------------------

EXTRA_RUN_ID = "run_id"
EXTRA_PARENT_RUN_ID = "parent_run_id"
EXTRA_MODEL_NAME = "model_name"
EXTRA_TOOL_CALLS = "tool_calls"
EXTRA_TOOL_NAME = "tool_name"
EXTRA_TOOL_CALL_ID = "tool_call_id"


# ---------------------------------------------------------------------------
# Typed shapes for the ``extra`` JSONB column
# ---------------------------------------------------------------------------


class StoredToolCall(TypedDict):
    """A single tool call as persisted in the ``extra`` JSONB column."""

    name: str
    args: dict[str, Any]
    id: str | None


class _ExtraBase(TypedDict):
    """Fields common to every ``extra`` JSONB value."""

    run_id: str
    parent_run_id: NotRequired[str]


class PromptExtra(_ExtraBase):
    """``extra`` shape for system / human messages."""


class AIExtra(_ExtraBase):
    """``extra`` shape for AI messages."""

    model_name: NotRequired[str]
    tool_calls: NotRequired[list[StoredToolCall]]


class ToolExtra(_ExtraBase):
    """``extra`` shape for tool-result messages."""

    tool_name: str
    tool_call_id: NotRequired[str]


MessageExtra = PromptExtra | AIExtra | ToolExtra


EMBEDDING_DIMENSIONS = 1024
"""Must match ``EmbeddingConfig.dimensions`` in ``configs/system.py``.
Changing this value requires an alembic migration to ALTER the
``Vector()`` columns in the database."""


# ---------------------------------------------------------------------------
# Chat messages table
# ---------------------------------------------------------------------------


class ChatMessage(Base):
    """A single message in a chat conversation.

    Every LangChain message (system, human, AI, tool) is stored as an
    equal row.  Messages from the same agent invocation share a
    ``trace_id`` and are ordered by ``created_at``.  Messages from the
    same multi-turn conversation share a ``conversation_id``.

    The ``query_embedding`` column is populated only on first-turn human
    messages to enable semantic response caching.  A partial HNSW index
    covers ``role = 'human' AND query_embedding IS NOT NULL``.
    """

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )
    conversation_id: Mapped[str] = mapped_column(
        String,
        nullable=False,
        index=False,
    )
    trace_id: Mapped[str] = mapped_column(
        String,
        nullable=False,
        index=False,
    )
    message_id: Mapped[str] = mapped_column(
        String,
        unique=True,
        nullable=False,
    )
    role: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )
    content: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
    )
    query_embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS),
        nullable=True,
    )
    extra: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )
    created_at: Mapped[str] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[str] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        Index(
            "ix_chat_messages_conversation_id_created_at",
            "conversation_id",
            "created_at",
        ),
        Index(
            "ix_chat_messages_trace_id_created_at",
            "trace_id",
            "created_at",
        ),
        Index("ix_chat_messages_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<ChatMessage(id={self.id}, conversation_id={self.conversation_id!r}, "
            f"trace_id={self.trace_id!r}, role={self.role!r})>"
        )


# ---------------------------------------------------------------------------
# Text embeddings table (RAG match hints)
# ---------------------------------------------------------------------------


class SourceEmbedding(Base):
    """Embedded match-hint vector for a persona knowledge source.

    Each row stores **one** hint phrase and its embedding vector.
    Multiple rows can share the same ``source_id`` (one per hint).
    Uniqueness is enforced on ``(source_id, text, model_name)``.
    """

    __tablename__ = "source_embeddings"

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )
    source_id: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )
    text: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )
    embedding: Mapped[list[float]] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS),
        nullable=False,
    )
    model_name: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )
    created_at: Mapped[str] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    __table_args__ = (
        Index(
            "uq_source_embeddings_source_text_model",
            "source_id",
            "text",
            "model_name",
            unique=True,
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<SourceEmbedding(id={self.id}, "
            f"source_id={self.source_id!r}, "
            f"text={self.text!r}, "
            f"model_name={self.model_name!r})>"
        )
