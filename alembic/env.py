"""Alembic environment configuration — async-aware.

Reads the database URL from the ``CHATTY_THIRD_PARTY__POSTGRES_URI``
environment variable (or falls back to ``ThirdPartyConfig`` default)
so that the same URI is used locally, in CI, and in Kubernetes.

Uses ``asyncpg`` as the async driver — no separate sync driver needed.
"""

import asyncio
import os

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

# Import Base.metadata so --autogenerate can detect model changes.
from chatty.infra.db.models import Base  # noqa: F401

target_metadata = Base.metadata


def _get_database_url() -> str:
    """Resolve the database URL from env or application defaults."""
    url = os.environ.get("CHATTY_THIRD_PARTY__POSTGRES_URI")
    if url:
        return url
    # Fall back to the pydantic-settings default (ThirdPartyConfig).
    from chatty.configs.system import ThirdPartyConfig

    return ThirdPartyConfig().postgres_uri


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL without a live DB)."""
    url = _get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    """Run migrations against a live connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations in 'online' mode with an async engine."""
    engine = create_async_engine(_get_database_url())
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
