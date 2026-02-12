"""Inbox — bounded admission control for incoming requests."""

from __future__ import annotations

import logging

from redis.asyncio import Redis

from chatty.configs.config import get_app_config
from chatty.infra.singleton import singleton

from .base import InboxBackend
from .local_backend import LocalInboxBackend
from .redis_backend import RedisInboxBackend

logger = logging.getLogger(__name__)

_KEY_PREFIX = "chatty:gate"


class Inbox:
    """Bounded admission control.

    Tracks how many requests are currently in-flight. Rejects with
    ``InboxFull`` when ``inbox_max_size`` is reached.

    Usage::

        inbox = get_inbox()

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
        return await self._backend.enter()

    async def leave(self) -> None:
        """Leave the inbox (always call, even on error)."""
        await self._backend.leave()

    async def aclose(self) -> None:
        """Shut down the underlying backend."""
        await self._backend.aclose()


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------


def _build_inbox_backend(redis_client: Redis | None) -> InboxBackend:
    """Choose and construct the right inbox backend."""
    config = get_app_config()
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

    return backend


@singleton
def get_inbox(redis_client: Redis | None = None) -> Inbox:
    """Return (or create) the singleton ``Inbox``.

    On the **first** call, supply *redis_client* (or ``None`` for the
    local fallback).  Subsequent calls ignore arguments and return the
    cached instance.
    """
    return Inbox(_build_inbox_backend(redis_client))
