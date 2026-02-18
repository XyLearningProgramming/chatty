"""Async Redis client lifespan dependency.

``build_redis`` creates a Redis client, verifies the connection, and
falls back to ``None`` when Redis is unreachable.  Downstream deps
(inbox, semaphore) declare ``Depends(build_redis)`` to receive the
shared client.  Cleanup runs automatically via ``yield``.
"""

import logging
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends
from redis.asyncio import Redis

from chatty.configs.config import AppConfig, get_app_config

logger = logging.getLogger(__name__)


async def build_redis(
    config: Annotated[AppConfig, Depends(get_app_config)],
) -> AsyncGenerator[Redis | None, None]:
    """Create a Redis client; yield ``None`` if unreachable."""
    client = Redis.from_url(
        config.third_party.redis_uri, decode_responses=True
    )
    verified: Redis | None = None
    try:
        await client.ping()
        verified = client
    except Exception:
        logger.warning(
            "Redis unavailable -- falling back to local "
            "concurrency."
        )

    yield verified

    if verified is not None:
        await client.aclose()
