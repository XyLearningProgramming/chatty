"""resize embedding vectors from 1024 to 384

Revision ID: 0003
Revises: 0002
Create Date: 2026-02-23

"""

from typing import Sequence, Union

from sqlalchemy import text

from alembic import op

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

OLD_DIM = 1024
NEW_DIM = 384


def upgrade() -> None:
    op.execute(text("DELETE FROM source_embeddings"))
    op.execute(text("DELETE FROM chat_messages WHERE query_embedding IS NOT NULL"))

    op.execute(
        text(
            f"ALTER TABLE source_embeddings "
            f"ALTER COLUMN embedding TYPE vector({NEW_DIM})"
        )
    )
    op.execute(
        text(
            f"ALTER TABLE chat_messages "
            f"ALTER COLUMN query_embedding TYPE vector({NEW_DIM})"
        )
    )


def downgrade() -> None:
    op.execute(text("DELETE FROM source_embeddings"))
    op.execute(text("DELETE FROM chat_messages WHERE query_embedding IS NOT NULL"))

    op.execute(
        text(
            f"ALTER TABLE source_embeddings "
            f"ALTER COLUMN embedding TYPE vector({OLD_DIM})"
        )
    )
    op.execute(
        text(
            f"ALTER TABLE chat_messages "
            f"ALTER COLUMN query_embedding TYPE vector({OLD_DIM})"
        )
    )
