"""Single-process concurrency backends using ``asyncio`` primitives."""

from __future__ import annotations

import asyncio

from .base import InboxBackend, InboxFull, SemaphoreBackend


class LocalInboxBackend(InboxBackend):
    """In-process inbox counter backed by an ``asyncio.Lock``."""

    def __init__(self, inbox_max_size: int) -> None:
        self._inbox_max_size = inbox_max_size
        self._inbox_count = 0
        self._inbox_lock = asyncio.Lock()

    async def enter(self) -> int:
        async with self._inbox_lock:
            if self._inbox_count >= self._inbox_max_size:
                raise InboxFull(
                    f"Inbox full ({self._inbox_max_size}): too many requests in flight."
                )
            self._inbox_count += 1
            return self._inbox_count

    async def leave(self) -> None:
        async with self._inbox_lock:
            self._inbox_count = max(0, self._inbox_count - 1)

    async def aclose(self) -> None:
        pass


class LocalSemaphoreBackend(SemaphoreBackend):
    """In-process semaphore backed by ``asyncio.Semaphore``."""

    def __init__(self, max_concurrency: int) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def acquire(self) -> None:
        await self._semaphore.acquire()

    async def release(self) -> None:
        self._semaphore.release()

    async def aclose(self) -> None:
        pass
