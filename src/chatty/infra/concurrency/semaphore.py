"""ModelSemaphore — concurrency limiter for LLM invocations."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from redis.asyncio import Redis

from chatty.configs.config import AppConfig, get_app_config
from chatty.core.service.metrics import SEMAPHORE_ACQUIRES_TOTAL, SEMAPHORE_WAIT_SECONDS
from chatty.infra.lifespan import get_app
from chatty.infra.redis import build_redis
from chatty.infra.telemetry import ATTR_SEMAPHORE_TIMEOUT, SPAN_SEMAPHORE_SLOT, tracer

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
        start = time.monotonic()
        try:
            async with asyncio.timeout(self._acquire_timeout):
                await self._backend.acquire()
        except TimeoutError:
            elapsed = time.monotonic() - start
            SEMAPHORE_WAIT_SECONDS.observe(elapsed)
            SEMAPHORE_ACQUIRES_TOTAL.labels(result="timeout").inc()
            logger.debug("Semaphore acquire timed out after %.3fs", elapsed)
            raise AcquireTimeout(
                "Timed out waiting for a model concurrency slot. "
                "Try again later."
            ) from None
        elapsed = time.monotonic() - start
        SEMAPHORE_WAIT_SECONDS.observe(elapsed)
        SEMAPHORE_ACQUIRES_TOTAL.labels(result="ok").inc()
        logger.debug("Semaphore acquired in %.3fs", elapsed)

    async def release(self) -> None:
        """Free the concurrency slot."""
        await self._backend.release()

    @asynccontextmanager
    async def slot(self) -> AsyncGenerator[None, None]:
        """Convenience context-manager: acquire a slot, yield, release."""
        with tracer.start_as_current_span(SPAN_SEMAPHORE_SLOT) as span:
            span.set_attribute(ATTR_SEMAPHORE_TIMEOUT, self._acquire_timeout)
            await self.acquire()
            try:
                yield
            finally:
                await self.release()

    async def aclose(self) -> None:
        """Shut down the underlying backend."""
        await self._backend.aclose()


# ---------------------------------------------------------------------------
# Lifespan dependency
# ---------------------------------------------------------------------------


async def build_semaphore(
    app: Annotated[FastAPI, Depends(get_app)],
    redis_client: Annotated[Redis | None, Depends(build_redis)],
    config: Annotated[AppConfig, Depends(get_app_config)],
) -> AsyncGenerator[None, None]:
    """Create a ``ModelSemaphore``, attach to ``app.state``; close on shutdown."""
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
        backend = LocalSemaphoreBackend(
            max_concurrency=cc.max_concurrency
        )
        logger.info(
            "ModelSemaphore: local backend (max_concurrency=%d)",
            cc.max_concurrency,
        )

    semaphore = ModelSemaphore(
        backend, acquire_timeout=cc.acquire_timeout
    )
    app.state.semaphore = semaphore
    yield
    await semaphore.aclose()


# ---------------------------------------------------------------------------
# Per-request dependency — reads from app.state
# ---------------------------------------------------------------------------


def get_model_semaphore(request: Request) -> ModelSemaphore:
    """Return the ``ModelSemaphore`` from ``app.state``."""
    return request.app.state.semaphore
