"""Async SQLAlchemy engine and session factory — **leaf module**.

``build_db`` is a lifespan dependency: it creates the engine +
session factory, attaches them to ``app.state``, and disposes the
engine on shutdown.  Per-request dependencies read from ``app.state``.

This module lives *outside* the ``db`` package on purpose so that
``telemetry`` can ``Depends(build_db)`` without triggering
``db/__init__`` (which re-exports repositories that import
``telemetry`` — a circular dependency).

The higher-level factories (``get_chat_message_history_factory``,
``get_embedding_repository``, ``get_cache_repository``) stay in
``db.engine`` because they need intra-package imports.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from chatty.configs.config import AppConfig, get_app_config
from chatty.infra.lifespan import get_app

# ---------------------------------------------------------------------------
# Lifespan dependency
# ---------------------------------------------------------------------------


async def build_db(
    app: Annotated[FastAPI, Depends(get_app)],
    config: Annotated[AppConfig, Depends(get_app_config)],
) -> AsyncGenerator[None, None]:
    """Create engine + session factory, attach to ``app.state``."""
    tp = config.third_party
    engine = create_async_engine(
        tp.postgres_uri,
        pool_pre_ping=True,
        pool_size=tp.postgres_pool_size,
        max_overflow=tp.postgres_max_overflow,
    )
    factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
        engine, expire_on_commit=False
    )
    app.state.engine = engine
    app.state.session_factory = factory
    yield
    await engine.dispose()


# ---------------------------------------------------------------------------
# Per-request dependencies — read from app.state
# ---------------------------------------------------------------------------


def get_session_factory(
    request: Request,
) -> async_sessionmaker[AsyncSession]:
    """Return the ``async_sessionmaker`` from ``app.state``."""
    return request.app.state.session_factory


async def get_async_session(
    request: Request,
) -> AsyncGenerator[AsyncSession, None]:
    """Yield one ``AsyncSession`` per request, auto-closed on exit."""
    factory: async_sessionmaker[AsyncSession] = request.app.state.session_factory
    async with factory() as session:
        yield session
