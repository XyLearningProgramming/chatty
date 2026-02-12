"""ModelSemaphore — concurrency limiter for LLM invocations."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import AsyncGenerator

from redis.asyncio import Redis

from chatty.configs.config import get_app_config
from chatty.infra.singleton import singleton

from .base import AcquireTimeout, SemaphoreBackend
from .local_backend import LocalSemaphoreBackend
from .redis_backend import RedisSemaphoreBackend

logger = logging.getLogger(__name__)

_KEY_PREFIX = "chatty:gate"


class ModelSemaphore:
    """Async semaphore with timeout for LLM concurrency.

    Wraps a ``SemaphoreBackend`` and adds a wall-clock timeout on
    ``acquire``.  Intended to be attached to a ``GatedChatModel`` so
    that every model invocation is individually gated.

    Usage::

        async with semaphore.slot():
            result = await inner_model._agenerate(...)
    """

    def __init__(
        self, backend: SemaphoreBackend, acquire_timeout: timedelta
    ) -> None:
        self._backend = backend
        self._acquire_timeout = acquire_timeout.total_seconds()

    async def acquire(self) -> None:
        """Wait for a concurrency slot (with timeout), then claim it.

        Raises:
            AcquireTimeout: if the slot cannot be acquired within the
                configured timeout.
        """
        try:
            async with asyncio.timeout(self._acquire_timeout):
                await self._backend.acquire()
        except TimeoutError:
            raise AcquireTimeout(
                "Timed out waiting for a model concurrency slot. "
                "Try again later."
            ) from None

    async def release(self) -> None:
        """Free the concurrency slot."""
        await self._backend.release()

    @asynccontextmanager
    async def slot(self) -> AsyncGenerator[None, None]:
        """Convenience context-manager: acquire a slot, yield, release."""
        await self.acquire()
        try:
            yield
        finally:
            await self.release()

    async def aclose(self) -> None:
        """Shut down the underlying backend."""
        await self._backend.aclose()


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------


def _build_semaphore_backend(redis_client: Redis | None) -> SemaphoreBackend:
    """Choose and construct the right semaphore backend."""
    config = get_app_config()
    cc = config.concurrency
    agent_name = config.chat.agent_name

    if redis_client is not None:
        slots_key = f"{_KEY_PREFIX}:{agent_name}:slots"
        notify_channel = f"{_KEY_PREFIX}:{agent_name}:notify"
        backend: SemaphoreBackend = RedisSemaphoreBackend(
            redis=redis_client,
            slots_key=slots_key,
            notify_channel=notify_channel,
            max_concurrency=cc.max_concurrency,
            ttl=cc.slot_timeout,
            acquire_timeout=cc.acquire_timeout,
        )
        logger.info(
            "ModelSemaphore: Redis backend "
            "(max_concurrency=%d, keys=%s)",
            cc.max_concurrency,
            slots_key,
        )
    else:
        backend = LocalSemaphoreBackend(max_concurrency=cc.max_concurrency)
        logger.info(
            "ModelSemaphore: local backend (max_concurrency=%d)",
            cc.max_concurrency,
        )

    return backend


@singleton
def get_model_semaphore(redis_client: Redis | None = None) -> ModelSemaphore:
    """Return (or create) the singleton ``ModelSemaphore``.

    On the **first** call, supply *redis_client* (or ``None`` for the
    local fallback).  Subsequent calls ignore arguments and return the
    cached instance.
    """
    cc = get_app_config().concurrency
    return ModelSemaphore(
        _build_semaphore_backend(redis_client),
        acquire_timeout=cc.acquire_timeout,
    )
