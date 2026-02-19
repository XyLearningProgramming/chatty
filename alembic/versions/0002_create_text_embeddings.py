"""create source_embeddings table

Revision ID: 0002
Revises: 0001
Create Date: 2026-02-18

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import text

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

EMBEDDING_DIMENSIONS = 1024


def upgrade() -> None:
    op.create_table(
        "source_embeddings",
        sa.Column(
            "id", sa.BigInteger(), autoincrement=True, nullable=False
        ),
        sa.Column("source_id", sa.String(), nullable=False),
        sa.Column("text", sa.String(), nullable=False),
        sa.Column("embedding", sa.Text(), nullable=False),
        sa.Column("model_name", sa.String(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_source_embeddings"),
    )

    op.execute(
        text(
            f"ALTER TABLE source_embeddings "
            f"ALTER COLUMN embedding TYPE vector({EMBEDDING_DIMENSIONS}) "
            f"USING embedding::text::float[]::vector"
        )
    )

    op.create_index(
        "uq_source_embeddings_source_text_model",
        "source_embeddings",
        ["source_id", "text", "model_name"],
        unique=True,
    )

    op.execute(
        text(
            "CREATE INDEX ix_source_embeddings_embedding_hnsw "
            "ON source_embeddings "
            "USING hnsw (embedding vector_cosine_ops)"
        )
    )


def downgrade() -> None:
    op.execute(
        text(
            "DROP INDEX IF EXISTS ix_source_embeddings_embedding_hnsw"
        )
    )
    op.drop_index(
        "uq_source_embeddings_source_text_model",
        table_name="source_embeddings",
    )
    op.drop_table("source_embeddings")
