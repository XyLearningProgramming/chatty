"""ConcurrencyGate — public API and singleton factory."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import AsyncGenerator

from redis.asyncio import Redis

from chatty.configs.config import get_app_config
from chatty.infra.singleton import singleton

from .base import ClientDisconnected, ConcurrencyBackend, GateFull
from .local_backend import LocalConcurrencyBackend
from .redis_backend import RedisConcurrencyBackend

logger = logging.getLogger(__name__)

# Type alias for the disconnect-check callback accepted by ``slot``.
# The callable should return *True* when the client is gone.
DisconnectCheck = Callable[[], Awaitable[bool]]

_KEY_PREFIX = "chatty:gate"


class ConcurrencyGate:
    """Bounded gate with inbox admission + concurrency semaphore.

    Usage::

        gate = get_concurrency_gate()

        # In the request handler:
        position = await gate.enter()      # GateFull → 429
        try:
            async with gate.slot():        # waits for semaphore
                async for ev in service.stream_response(q):
                    yield ev
        finally:
            await gate.leave()
    """

    def __init__(
        self, backend: ConcurrencyBackend, acquire_timeout: timedelta
    ) -> None:
        self._backend = backend
        self._acquire_timeout = acquire_timeout.total_seconds()

    async def enter(self) -> int:
        """Admit into the inbox. Returns position. Raises ``GateFull``."""
        return await self._backend.enter()

    async def acquire(
        self, disconnected: DisconnectCheck | None = None
    ) -> None:
        """Wait for a concurrency slot (with timeout), then claim it.

        Args:
            disconnected: Optional async callable checked *once* right after
                the slot is acquired.  If the client has already left, the
                slot is released immediately and ``ClientDisconnected`` is
                raised — avoiding expensive LLM work for a gone client.

        Raises:
            GateFull: if the slot cannot be acquired within
                ``acquire_timeout``.
        """
        try:
            async with asyncio.timeout(self._acquire_timeout):
                await self._backend.acquire()
        except TimeoutError:
            raise GateFull(
                "Timed out waiting for a concurrency slot. Try again later."
            ) from None

        # Single proactive check right after dequeue.
        if disconnected is not None and await disconnected():
            await self._backend.release()
            raise ClientDisconnected(
                "Client disconnected before slot was used."
            )

    async def release(self) -> None:
        """Free the concurrency slot."""
        await self._backend.release()

    @asynccontextmanager
    async def slot(
        self, disconnected: DisconnectCheck | None = None
    ) -> AsyncGenerator[None, None]:
        """Convenience context-manager: acquire a slot, yield, release."""
        await self.acquire(disconnected=disconnected)
        try:
            yield
        finally:
            await self.release()

    async def leave(self) -> None:
        """Leave the inbox (always call, even on error)."""
        await self._backend.leave()

    async def aclose(self) -> None:
        """Shut down the underlying backend."""
        await self._backend.aclose()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


def _build_backend(redis_client: Redis | None) -> ConcurrencyBackend:
    """Choose and construct the right backend."""
    config = get_app_config()
    cc = config.concurrency
    agent_name = config.chat.agent_name

    if redis_client is not None:
        inbox_key = f"{_KEY_PREFIX}:{agent_name}:inbox"
        slots_key = f"{_KEY_PREFIX}:{agent_name}:slots"
        notify_channel = f"{_KEY_PREFIX}:{agent_name}:notify"
        backend: ConcurrencyBackend = RedisConcurrencyBackend(
            redis=redis_client,
            inbox_key=inbox_key,
            slots_key=slots_key,
            notify_channel=notify_channel,
            max_concurrency=cc.max_concurrency,
            inbox_max_size=cc.inbox_max_size,
            ttl=cc.slot_timeout,
            acquire_timeout=cc.acquire_timeout,
        )
        logger.info(
            "Concurrency gate: Redis backend "
            "(max_concurrency=%d, inbox=%d, keys=%s/%s)",
            cc.max_concurrency,
            cc.inbox_max_size,
            inbox_key,
            slots_key,
        )
    else:
        backend = LocalConcurrencyBackend(
            max_concurrency=cc.max_concurrency,
            inbox_max_size=cc.inbox_max_size,
        )
        logger.info(
            "Concurrency gate: local backend "
            "(max_concurrency=%d, inbox=%d)",
            cc.max_concurrency,
            cc.inbox_max_size,
        )

    return backend


@singleton
def get_concurrency_gate(redis_client: Redis | None = None) -> ConcurrencyGate:
    """Return (or create) the singleton ``ConcurrencyGate``.

    On the **first** call, supply *redis_client* (or ``None`` for the
    local fallback).  Subsequent calls ignore arguments and return the
    cached instance — just like ``get_redis_client()``.
    """
    cc = get_app_config().concurrency
    return ConcurrencyGate(_build_backend(redis_client), acquire_timeout=cc.acquire_timeout)
