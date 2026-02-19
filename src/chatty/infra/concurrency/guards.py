"""Unified anti-flood gate: rate limiting + request deduplication.

Combines per-IP rate limiting, global QPS cap, nonce dedup, and
server-side fingerprint dedup into a single ``RequestGuard`` class.
When Redis is available all checks run in **one pipeline call**
(~10 ops, 1 round-trip).  Falls back to in-process data structures
when Redis is unavailable.

The ``enforce_request_guards`` async-generator dependency performs the
check as a side effect — the endpoint declares it via
``Annotated[None, Depends(enforce_request_guards)]`` and never
touches anti-flood logic directly.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import AsyncGenerator
from datetime import timedelta
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from redis.asyncio import Redis

from chatty.configs.config import AppConfig, get_app_config
from chatty.infra.lifespan import get_app
from chatty.infra.redis import build_redis

from .base import DuplicateRequest, RateLimited
from .inbox import Inbox, get_inbox
from .real_ip import get_real_ip

logger = logging.getLogger(__name__)

_RATELIMIT_KEY = "chatty:ratelimit:{scope}"
_DEDUP_FP_KEY = "chatty:dedup:fp:{digest}"
_DEDUP_NONCE_KEY = "chatty:dedup:nonce:{nonce}"
_NONCE_TTL = 60


class RequestGuard:
    """Unified anti-flood gate.  Runs all checks in a single Redis pipeline."""

    def __init__(
        self,
        *,
        redis: Redis | None,
        per_ip_limit: int,
        global_limit: int,
        dedup_window: timedelta,
        rate_window_seconds: float = 1.0,
    ) -> None:
        self._redis = redis
        self._per_ip_limit = per_ip_limit
        self._global_limit = global_limit
        self._fp_ttl = int(dedup_window.total_seconds())
        self._rate_window = rate_window_seconds

        # Local fallback state
        self._local_buckets: dict[str, list[float]] = {}
        self._local_nonces: dict[str, float] = {}
        self._local_fingerprints: dict[str, float] = {}

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    async def check(self, ip: str, query: str, nonce: str | None) -> None:
        """Run all anti-flood checks.  Raises on rejection."""
        if self._redis is not None:
            await self._check_redis(ip, query, nonce)
        else:
            self._check_local(ip, query, nonce)

    # -----------------------------------------------------------------
    # Redis path — single pipeline
    # -----------------------------------------------------------------

    async def _check_redis(
        self, ip: str, query: str, nonce: str | None
    ) -> None:
        now = time.time()
        window_start = now - self._rate_window
        member = str(now)

        ip_key = _RATELIMIT_KEY.format(scope=f"ip:{ip}")
        global_key = _RATELIMIT_KEY.format(scope="global")
        fp_digest = hashlib.sha256(f"{ip}:{query}".encode()).hexdigest()[:16]
        fp_key = _DEDUP_FP_KEY.format(digest=fp_digest)

        pipe = self._redis.pipeline()

        # Per-IP sliding window (indices 0-3)
        pipe.zremrangebyscore(ip_key, 0, window_start)
        pipe.zadd(ip_key, {member: now})
        pipe.zcard(ip_key)
        pipe.expire(ip_key, int(self._rate_window) + 1)

        # Global sliding window (indices 4-7)
        pipe.zremrangebyscore(global_key, 0, window_start)
        pipe.zadd(global_key, {member: now})
        pipe.zcard(global_key)
        pipe.expire(global_key, int(self._rate_window) + 1)

        # Fingerprint dedup (index 8)
        if self._fp_ttl > 0:
            pipe.set(fp_key, "1", nx=True, ex=self._fp_ttl)
        else:
            pipe.echo("skip")  # placeholder to keep indices stable

        # Nonce dedup (index 9, optional)
        has_nonce = bool(nonce)
        if has_nonce:
            nonce_key = _DEDUP_NONCE_KEY.format(nonce=nonce)
            pipe.set(nonce_key, "1", nx=True, ex=_NONCE_TTL)

        results = await pipe.execute()

        ip_count = results[2]
        global_count = results[6]

        if self._per_ip_limit > 0 and ip_count > self._per_ip_limit:
            raise RateLimited(
                f"Per-IP rate limit exceeded ({self._per_ip_limit}/s)",
                scope="ip",
            )
        if self._global_limit > 0 and global_count > self._global_limit:
            raise RateLimited(
                f"Global rate limit exceeded ({self._global_limit}/s)",
                scope="global",
            )
        if self._fp_ttl > 0 and results[8] is None:
            raise DuplicateRequest("Duplicate request detected")
        if has_nonce and results[9] is None:
            raise DuplicateRequest(f"Duplicate nonce: {nonce}")

    # -----------------------------------------------------------------
    # Local fallback path
    # -----------------------------------------------------------------

    def _check_local(
        self, ip: str, query: str, nonce: str | None
    ) -> None:
        now = time.time()

        # Per-IP rate limit
        if self._per_ip_limit > 0:
            ip_key = _RATELIMIT_KEY.format(scope=f"ip:{ip}")
            if not self._check_local_bucket(ip_key, self._per_ip_limit, now):
                raise RateLimited(
                    f"Per-IP rate limit exceeded ({self._per_ip_limit}/s)",
                    scope="ip",
                )

        # Global rate limit
        if self._global_limit > 0:
            global_key = _RATELIMIT_KEY.format(scope="global")
            if not self._check_local_bucket(global_key, self._global_limit, now):
                raise RateLimited(
                    f"Global rate limit exceeded ({self._global_limit}/s)",
                    scope="global",
                )

        # Fingerprint dedup
        if self._fp_ttl > 0:
            fp_digest = hashlib.sha256(
                f"{ip}:{query}".encode()
            ).hexdigest()[:16]
            fp_key = _DEDUP_FP_KEY.format(digest=fp_digest)
            if self._check_local_seen(self._local_fingerprints, fp_key, self._fp_ttl, now):
                raise DuplicateRequest("Duplicate request detected")

        # Nonce dedup
        if nonce:
            nonce_key = _DEDUP_NONCE_KEY.format(nonce=nonce)
            if self._check_local_seen(self._local_nonces, nonce_key, _NONCE_TTL, now):
                raise DuplicateRequest(f"Duplicate nonce: {nonce}")

    def _check_local_bucket(self, key: str, limit: int, now: float) -> bool:
        """Return ``True`` if within limit."""
        window_start = now - self._rate_window
        timestamps = self._local_buckets.get(key, [])
        timestamps = [t for t in timestamps if t > window_start]
        timestamps.append(now)
        self._local_buckets[key] = timestamps
        return len(timestamps) <= limit

    def _check_local_seen(
        self,
        store: dict[str, float],
        key: str,
        ttl: int,
        now: float,
    ) -> bool:
        """Return ``True`` if the key was already seen (= duplicate)."""
        expired = [k for k, exp in store.items() if exp <= now]
        for k in expired:
            del store[k]
        if key in store:
            return True
        store[key] = now + ttl
        return False

    async def aclose(self) -> None:
        """Clean up resources."""
        self._local_buckets.clear()
        self._local_nonces.clear()
        self._local_fingerprints.clear()


# ---------------------------------------------------------------------------
# Lifespan dependency
# ---------------------------------------------------------------------------


async def build_request_guard(
    app: Annotated[FastAPI, Depends(get_app)],
    redis_client: Annotated[Redis | None, Depends(build_redis)],
    config: Annotated[AppConfig, Depends(get_app_config)],
) -> AsyncGenerator[None, None]:
    """Create a ``RequestGuard``, attach to ``app.state``; close on shutdown."""
    api = config.api
    backend = "Redis" if redis_client is not None else "local"

    guard = RequestGuard(
        redis=redis_client,
        per_ip_limit=api.chat_rate_limit_per_second,
        global_limit=api.chat_global_rate_limit,
        dedup_window=api.dedup_window,
    )
    app.state.request_guard = guard
    logger.info(
        "RequestGuard: %s backend (per_ip=%d/s, global=%d/s, dedup=%s)",
        backend,
        api.chat_rate_limit_per_second,
        api.chat_global_rate_limit,
        api.dedup_window,
    )
    yield
    await guard.aclose()


# ---------------------------------------------------------------------------
# Per-request dependencies
# ---------------------------------------------------------------------------


def get_request_guard(request: Request) -> RequestGuard:
    """Return the ``RequestGuard`` stored on ``app.state`` by the lifespan."""
    return request.app.state.request_guard


async def enforce_request_guards(
    request: Request,
    real_ip: Annotated[str, Depends(get_real_ip)],
    guard: Annotated[RequestGuard, Depends(get_request_guard)],
) -> AsyncGenerator[None, None]:
    """Side-effect dependency: rate-limit + dedup.

    Extracts the query and nonce from the already-parsed request body.
    Raises ``RateLimited`` or ``DuplicateRequest`` on rejection — the
    exception handlers in ``exceptions.py`` convert these to HTTP
    responses.
    """
    body = await request.json()
    query = body.get("query", "")
    nonce = body.get("nonce")
    await guard.check(real_ip, query, nonce)
    yield


async def enforce_inbox(
    inbox: Annotated[Inbox, Depends(get_inbox)],
) -> AsyncGenerator[int, None]:
    """Side-effect dependency: inbox admission control.

    Calls ``inbox.enter()`` before the endpoint runs and guarantees
    ``inbox.leave()`` when the response finishes (including streaming).
    Raises ``InboxFull`` on rejection — the exception handler in
    ``exceptions.py`` converts it to a 429 response.

    Yields the inbox position so the endpoint can pass it to
    ``_chat_events``.
    """
    position = await inbox.enter()
    try:
        yield position
    finally:
        await inbox.leave()
