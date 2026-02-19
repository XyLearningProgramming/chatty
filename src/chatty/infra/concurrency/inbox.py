"""Inbox — bounded admission control for incoming requests."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from redis.asyncio import Redis

from chatty.configs.config import AppConfig, get_app_config
from chatty.core.service.metrics import INBOX_OCCUPANCY
from chatty.infra.lifespan import get_app
from chatty.infra.redis import build_redis
from chatty.infra.telemetry import (
    ATTR_INBOX_POSITION,
    ATTR_INBOX_REJECTED,
    SPAN_INBOX_ENTER,
    tracer,
)

from .base import InboxBackend, InboxFull
from .local_backend import LocalInboxBackend
from .redis_backend import RedisInboxBackend

logger = logging.getLogger(__name__)

_KEY_PREFIX = "chatty:gate"


class Inbox:
    """Bounded admission control.

    Tracks how many requests are currently in-flight. Rejects with
    ``InboxFull`` when ``inbox_max_size`` is reached.

    Usage::

        inbox = get_inbox(request)

        position = await inbox.enter()   # InboxFull → 429
        try:
            ...  # process request
        finally:
            await inbox.leave()
    """

    def __init__(self, backend: InboxBackend) -> None:
        self._backend = backend

    async def enter(self) -> int:
        """Admit into the inbox.  Returns position.  Raises ``InboxFull``."""
        with tracer.start_as_current_span(SPAN_INBOX_ENTER) as span:
            try:
                position = await self._backend.enter()
            except InboxFull:
                span.set_attribute(ATTR_INBOX_REJECTED, True)
                logger.info("Inbox full — rejecting request")
                raise
            INBOX_OCCUPANCY.inc()
            span.set_attribute(ATTR_INBOX_POSITION, position)
            logger.debug("Inbox enter: position=%d", position)
            return position

    async def leave(self) -> None:
        """Leave the inbox (always call, even on error)."""
        INBOX_OCCUPANCY.dec()
        await self._backend.leave()

    async def aclose(self) -> None:
        """Shut down the underlying backend."""
        await self._backend.aclose()


# ---------------------------------------------------------------------------
# Lifespan dependency
# ---------------------------------------------------------------------------


async def build_inbox(
    app: Annotated[FastAPI, Depends(get_app)],
    redis_client: Annotated[Redis | None, Depends(build_redis)],
    config: Annotated[AppConfig, Depends(get_app_config)],
) -> AsyncGenerator[None, None]:
    """Create an ``Inbox``, attach to ``app.state``; close on shutdown."""
    cc = config.concurrency
    agent_name = config.chat.agent_name

    if redis_client is not None:
        inbox_key = f"{_KEY_PREFIX}:{agent_name}:inbox"
        backend: InboxBackend = RedisInboxBackend(
            redis=redis_client,
            inbox_key=inbox_key,
            inbox_max_size=cc.inbox_max_size,
            ttl=cc.slot_timeout,
        )
        logger.info(
            "Inbox: Redis backend (inbox=%d, key=%s)",
            cc.inbox_max_size,
            inbox_key,
        )
    else:
        backend = LocalInboxBackend(inbox_max_size=cc.inbox_max_size)
        logger.info(
            "Inbox: local backend (inbox=%d)",
            cc.inbox_max_size,
        )

    inbox = Inbox(backend)
    app.state.inbox = inbox
    yield
    await inbox.aclose()


# ---------------------------------------------------------------------------
# Per-request dependency — reads from app.state
# ---------------------------------------------------------------------------


def get_inbox(request: Request) -> Inbox:
    """Return the ``Inbox`` stored on ``app.state`` by the lifespan."""
    return request.app.state.inbox
