"""Distributed concurrency backends backed by Redis Lua scripts + Pub/Sub."""

from __future__ import annotations

import logging
import time
from datetime import timedelta

from redis.asyncio import Redis

from .base import AcquireTimeout, InboxBackend, InboxFull, SemaphoreBackend

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lua scripts — inbox
# ---------------------------------------------------------------------------

# Atomically increment inbox counter if below max.
# KEYS[1] = inbox key, ARGV[1] = max size, ARGV[2] = TTL seconds.
# Returns new count on success, -1 when full.
_LUA_ENTER = """
local key = KEYS[1]
local max = tonumber(ARGV[1])
local ttl = tonumber(ARGV[2])
local cur = tonumber(redis.call('GET', key) or '0')
if cur < max then
    local n = redis.call('INCR', key)
    redis.call('EXPIRE', key, ttl)
    return n
end
return -1
"""

# Decrement inbox counter (floor at 0).
_LUA_LEAVE = """
local key = KEYS[1]
local ttl = tonumber(ARGV[1])
local cur = tonumber(redis.call('GET', key) or '0')
if cur > 0 then
    redis.call('DECR', key)
    redis.call('EXPIRE', key, ttl)
end
return 0
"""

# ---------------------------------------------------------------------------
# Lua scripts — semaphore
# ---------------------------------------------------------------------------

# Atomically try to acquire a semaphore slot.
# KEYS[1] = slots key, ARGV[1] = max concurrency, ARGV[2] = TTL.
# Returns 1 on success, 0 when all slots are taken.
_LUA_ACQUIRE = """
local key = KEYS[1]
local max = tonumber(ARGV[1])
local ttl = tonumber(ARGV[2])
local cur = tonumber(redis.call('GET', key) or '0')
if cur < max then
    redis.call('INCR', key)
    redis.call('EXPIRE', key, ttl)
    return 1
end
return 0
"""

# Release a semaphore slot and publish a notification so waiters wake up.
# KEYS[1] = slots key, KEYS[2] = notify channel.
# ARGV[1] = TTL seconds.
_LUA_RELEASE = """
local key = KEYS[1]
local channel = KEYS[2]
local ttl = tonumber(ARGV[1])
local cur = tonumber(redis.call('GET', key) or '0')
if cur > 0 then
    redis.call('DECR', key)
    redis.call('EXPIRE', key, ttl)
end
redis.call('PUBLISH', channel, '1')
return 0
"""


# ---------------------------------------------------------------------------
# Inbox backend
# ---------------------------------------------------------------------------


class RedisInboxBackend(InboxBackend):
    """Distributed inbox counter backed by a Redis Lua script."""

    def __init__(
        self,
        redis: Redis,
        inbox_key: str,
        inbox_max_size: int,
        ttl: timedelta,
    ) -> None:
        self._redis = redis
        self._inbox_key = inbox_key
        self._inbox_max_size = inbox_max_size
        self._ttl_seconds = int(ttl.total_seconds())

        self._enter_sha: str | None = None
        self._leave_sha: str | None = None

    async def _ensure_scripts(self) -> None:
        if self._enter_sha is None:
            self._enter_sha = await self._redis.script_load(_LUA_ENTER)
            self._leave_sha = await self._redis.script_load(_LUA_LEAVE)

    async def enter(self) -> int:
        await self._ensure_scripts()
        result = await self._redis.evalsha(
            self._enter_sha,  # type: ignore[arg-type]
            1,
            self._inbox_key,
            str(self._inbox_max_size),
            str(self._ttl_seconds),
        )
        count = int(result)
        if count == -1:
            raise InboxFull(
                f"Inbox full ({self._inbox_max_size}): "
                "too many requests in flight."
            )
        return count

    async def leave(self) -> None:
        await self._ensure_scripts()
        await self._redis.evalsha(
            self._leave_sha,  # type: ignore[arg-type]
            1,
            self._inbox_key,
            str(self._ttl_seconds),
        )

    async def aclose(self) -> None:
        # Redis client lifecycle is managed externally (infra/redis.py).
        pass


# ---------------------------------------------------------------------------
# Semaphore backend
# ---------------------------------------------------------------------------


class RedisSemaphoreBackend(SemaphoreBackend):
    """Distributed semaphore backed by Redis Lua scripts + Pub/Sub.

    The ``acquire`` method waits for a Pub/Sub notification from
    ``release()`` instead of polling, giving instant wake-up with zero
    busy-looping.
    """

    def __init__(
        self,
        redis: Redis,
        slots_key: str,
        notify_channel: str,
        max_concurrency: int,
        ttl: timedelta,
        acquire_timeout: timedelta,
    ) -> None:
        self._redis = redis
        self._slots_key = slots_key
        self._notify_channel = notify_channel
        self._max_concurrency = max_concurrency
        self._ttl_seconds = int(ttl.total_seconds())
        self._acquire_timeout = acquire_timeout.total_seconds()

        self._acquire_sha: str | None = None
        self._release_sha: str | None = None

    async def _ensure_scripts(self) -> None:
        if self._acquire_sha is None:
            self._acquire_sha = await self._redis.script_load(_LUA_ACQUIRE)
            self._release_sha = await self._redis.script_load(_LUA_RELEASE)

    async def _try_acquire(self) -> bool:
        """Attempt to claim a semaphore slot (non-blocking)."""
        result = await self._redis.evalsha(
            self._acquire_sha,  # type: ignore[arg-type]
            1,
            self._slots_key,
            str(self._max_concurrency),
            str(self._ttl_seconds),
        )
        return int(result) == 1

    async def acquire(self) -> None:
        await self._ensure_scripts()
        deadline = time.monotonic() + self._acquire_timeout

        # Fast path — try immediately before subscribing.
        if await self._try_acquire():
            return

        # Subscribe and wait for release notifications.
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(self._notify_channel)
        try:
            while True:
                # A release may have happened between our last attempt and
                # the subscribe, so try once before blocking.
                if await self._try_acquire():
                    return

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AcquireTimeout(
                        "Timed out waiting for a concurrency slot. "
                        "Try again later."
                    )

                # Block on the socket for up to ``remaining`` seconds.
                # Positive timeout is required — redis-py treats 0 / None
                # as non-blocking and would spin the CPU.
                await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=remaining
                )
        finally:
            await pubsub.unsubscribe(self._notify_channel)
            await pubsub.aclose()

    async def release(self) -> None:
        await self._ensure_scripts()
        await self._redis.evalsha(
            self._release_sha,  # type: ignore[arg-type]
            2,
            self._slots_key,
            self._notify_channel,
            str(self._ttl_seconds),
        )

    async def aclose(self) -> None:
        # Redis client lifecycle is managed externally (infra/redis.py).
        pass
