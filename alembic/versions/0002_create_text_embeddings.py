"""create text_embeddings table

Revision ID: 0002
Revises: 0001
Create Date: 2026-02-13

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Default embedding dimensions (text-embedding-ada-002)
EMBEDDING_DIMENSIONS = 1536


def upgrade() -> None:
    # Enable pgvector extension
    op.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    # Create table (embedding as text initially, will be altered)
    op.create_table(
        "text_embeddings",
        sa.Column(
            "id", sa.BigInteger(), autoincrement=True, nullable=False
        ),
        sa.Column("text_hash", sa.String(), nullable=False),
        sa.Column("text_content", sa.String(), nullable=False),
        sa.Column("embedding", sa.Text(), nullable=False),
        sa.Column("model_name", sa.String(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_text_embeddings"),
    )

    # Convert embedding column to vector type
    # Cast from text (array format {0.1,0.2,...}) to float array, then to vector
    op.execute(
        text(
            f"ALTER TABLE text_embeddings "
            f"ALTER COLUMN embedding TYPE vector({EMBEDDING_DIMENSIONS}) "
            f"USING embedding::text::float[]::vector"
        )
    )

    # Create unique index on (text_hash, model_name)
    op.create_index(
        "uq_text_embeddings_text_hash_model_name",
        "text_embeddings",
        ["text_hash", "model_name"],
        unique=True,
    )

    # Create HNSW index for fast similarity search
    op.execute(
        text(
            "CREATE INDEX ix_text_embeddings_embedding_hnsw "
            "ON text_embeddings "
            "USING hnsw (embedding vector_cosine_ops)"
        )
    )


def downgrade() -> None:
    op.drop_index(
        "uq_text_embeddings_text_hash_model_name",
        table_name="text_embeddings",
    )
    op.execute(text("DROP INDEX IF EXISTS ix_text_embeddings_embedding_hnsw"))
    op.drop_table("text_embeddings")
    # Note: We don't drop the vector extension as it might be used elsewhere
