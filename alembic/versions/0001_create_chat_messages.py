"""create chat_messages table

Revision ID: 0001
Revises:
Create Date: 2026-02-12

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "chat_messages",
        sa.Column(
            "id", sa.BigInteger(), autoincrement=True, nullable=False
        ),
        sa.Column(
            "conversation_id",
            sa.String(),
            nullable=False,
        ),
        sa.Column(
            "trace_id",
            sa.String(),
            nullable=False,
        ),
        sa.Column(
            "message_id",
            sa.String(),
            nullable=False,
        ),
        sa.Column("role", sa.String(length=20), nullable=False),
        sa.Column("content", sa.String(), nullable=True),
        sa.Column("extra", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_chat_messages"),
        sa.UniqueConstraint("message_id", name="uq_chat_messages_message_id"),
    )
    op.create_index(
        "ix_chat_messages_conversation_id_created_at",
        "chat_messages",
        ["conversation_id", "created_at"],
    )
    op.create_index(
        "ix_chat_messages_trace_id_created_at",
        "chat_messages",
        ["trace_id", "created_at"],
    )
    op.create_index(
        "ix_chat_messages_created_at",
        "chat_messages",
        ["created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_chat_messages_created_at", table_name="chat_messages"
    )
    op.drop_index(
        "ix_chat_messages_trace_id_created_at",
        table_name="chat_messages",
    )
    op.drop_index(
        "ix_chat_messages_conversation_id_created_at",
        table_name="chat_messages",
    )
    op.drop_table("chat_messages")
