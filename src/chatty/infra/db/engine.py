"""Async SQLAlchemy engine and session factory singletons.

Follows the same singleton pattern as ``chatty.infra.redis``.  The
engine is created lazily on first call and cached for the process
lifetime.  Callers are responsible for calling ``engine.dispose()``
on shutdown (wired through the FastAPI lifespan in ``app.py``).
"""

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)

from chatty.configs.config import get_app_config
from chatty.infra.singleton import singleton


@singleton
def get_async_engine() -> AsyncEngine:
    """Create and cache an ``AsyncEngine`` from application config.

    Uses the ``postgresql+asyncpg://`` URI from
    ``ThirdPartyConfig.postgres_uri``.  The engine is **not** connected
    until the first query; call ``await engine.connect()`` or run a
    statement to verify reachability.
    """
    third_party_config = get_app_config().third_party
    return create_async_engine(
        third_party_config.postgres_uri,
        pool_pre_ping=True,
        pool_size=third_party_config.postgres_pool_size,
        max_overflow=third_party_config.postgres_max_overflow,
    )


@singleton
def get_async_session_factory() -> async_sessionmaker:
    """Create and cache an ``async_sessionmaker`` bound to the engine."""
    engine = get_async_engine()
    return async_sessionmaker(engine, expire_on_commit=False)
