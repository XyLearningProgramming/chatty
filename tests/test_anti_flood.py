"""Tests for anti-flood defenses: real-IP extraction and RequestGuard."""

from __future__ import annotations

import asyncio
import time
from datetime import timedelta
from unittest.mock import MagicMock

import pytest

from chatty.infra.concurrency.base import DuplicateRequest, RateLimited
from chatty.infra.concurrency.guards import RequestGuard
from chatty.infra.concurrency.real_ip import get_real_ip

# =========================================================================
# Real IP extraction
# =========================================================================


class _FakeRequest:
    """Minimal stand-in for ``fastapi.Request``."""

    def __init__(
        self, headers: dict[str, str] | None = None, client_host: str | None = None
    ) -> None:
        self.headers = headers or {}
        self.client = MagicMock(host=client_host) if client_host else None


class TestGetRealIP:
    def test_cf_connecting_ip_preferred(self):
        req = _FakeRequest(
            headers={
                "cf-connecting-ip": "1.2.3.4",
                "x-real-ip": "5.6.7.8",
                "x-forwarded-for": "9.10.11.12, 1.1.1.1",
            },
            client_host="10.0.0.1",
        )
        assert get_real_ip(req) == "1.2.3.4"

    def test_x_real_ip_fallback(self):
        req = _FakeRequest(
            headers={"x-real-ip": "5.6.7.8", "x-forwarded-for": "9.10.11.12"},
            client_host="10.0.0.1",
        )
        assert get_real_ip(req) == "5.6.7.8"

    def test_x_forwarded_for_leftmost(self):
        req = _FakeRequest(
            headers={"x-forwarded-for": "9.10.11.12, 1.1.1.1"},
            client_host="10.0.0.1",
        )
        assert get_real_ip(req) == "9.10.11.12"

    def test_client_host_last_resort(self):
        req = _FakeRequest(client_host="10.0.0.1")
        assert get_real_ip(req) == "10.0.0.1"

    def test_no_client_returns_unknown(self):
        req = _FakeRequest()
        assert get_real_ip(req) == "unknown"

    def test_strips_whitespace(self):
        req = _FakeRequest(headers={"cf-connecting-ip": "  1.2.3.4  "})
        assert get_real_ip(req) == "1.2.3.4"


# =========================================================================
# Helpers
# =========================================================================


def _make_guard(
    *,
    per_ip: int = 100,
    global_limit: int = 100,
    dedup_window: int = 5,
    rate_window: float = 1.0,
) -> RequestGuard:
    return RequestGuard(
        redis=None,
        per_ip_limit=per_ip,
        global_limit=global_limit,
        dedup_window=timedelta(seconds=dedup_window),
        rate_window_seconds=rate_window,
    )


# =========================================================================
# RequestGuard — per-IP rate limiting (local backend)
# =========================================================================


class TestGuardPerIPRateLimit:
    @pytest.mark.asyncio
    async def test_allows_within_limit(self):
        guard = _make_guard(per_ip=3, dedup_window=0)
        for _ in range(3):
            await guard.check("1.2.3.4", f"query-{_}", None)

    @pytest.mark.asyncio
    async def test_rejects_over_limit(self):
        guard = _make_guard(per_ip=2, dedup_window=0)
        await guard.check("1.2.3.4", "q1", None)
        await guard.check("1.2.3.4", "q2", None)
        with pytest.raises(RateLimited, match="Per-IP") as exc_info:
            await guard.check("1.2.3.4", "q3", None)
        assert exc_info.value.scope == "ip"

    @pytest.mark.asyncio
    async def test_different_ips_independent(self):
        guard = _make_guard(per_ip=1, dedup_window=0)
        await guard.check("1.1.1.1", "q", None)
        await guard.check("2.2.2.2", "q", None)

    @pytest.mark.asyncio
    async def test_disabled_when_zero(self):
        guard = _make_guard(per_ip=0, global_limit=0, dedup_window=0)
        for i in range(50):
            await guard.check("1.2.3.4", f"q{i}", None)

    @pytest.mark.asyncio
    async def test_window_expiry(self):
        guard = _make_guard(per_ip=1, dedup_window=0, rate_window=0.1)
        await guard.check("1.2.3.4", "q1", None)
        with pytest.raises(RateLimited):
            await guard.check("1.2.3.4", "q2", None)
        await asyncio.sleep(0.15)
        await guard.check("1.2.3.4", "q3", None)


# =========================================================================
# RequestGuard — global rate limiting (local backend)
# =========================================================================


class TestGuardGlobalRateLimit:
    @pytest.mark.asyncio
    async def test_global_limit(self):
        guard = _make_guard(per_ip=100, global_limit=2, dedup_window=0)
        await guard.check("1.1.1.1", "q1", None)
        await guard.check("2.2.2.2", "q2", None)
        with pytest.raises(RateLimited, match="Global") as exc_info:
            await guard.check("3.3.3.3", "q3", None)
        assert exc_info.value.scope == "global"


# =========================================================================
# RequestGuard — fingerprint dedup (local backend)
# =========================================================================


class TestGuardFingerprintDedup:
    @pytest.mark.asyncio
    async def test_allows_first(self):
        guard = _make_guard()
        await guard.check("1.2.3.4", "hello world", None)

    @pytest.mark.asyncio
    async def test_rejects_duplicate(self):
        guard = _make_guard()
        await guard.check("1.2.3.4", "hello world", None)
        with pytest.raises(DuplicateRequest, match="Duplicate request"):
            await guard.check("1.2.3.4", "hello world", None)

    @pytest.mark.asyncio
    async def test_different_ip_ok(self):
        guard = _make_guard()
        await guard.check("1.2.3.4", "hello world", None)
        await guard.check("5.6.7.8", "hello world", None)

    @pytest.mark.asyncio
    async def test_different_query_ok(self):
        guard = _make_guard()
        await guard.check("1.2.3.4", "hello world", None)
        await guard.check("1.2.3.4", "goodbye world", None)

    @pytest.mark.asyncio
    async def test_disabled_when_zero(self):
        guard = _make_guard(dedup_window=0)
        await guard.check("1.2.3.4", "hello", None)
        await guard.check("1.2.3.4", "hello", None)


# =========================================================================
# RequestGuard — nonce dedup (local backend)
# =========================================================================


class TestGuardNonceDedup:
    @pytest.mark.asyncio
    async def test_allows_first(self):
        guard = _make_guard(dedup_window=0)
        await guard.check("1.2.3.4", "q", "abc-123")

    @pytest.mark.asyncio
    async def test_rejects_duplicate_nonce(self):
        guard = _make_guard(dedup_window=0)
        await guard.check("1.2.3.4", "q1", "abc-123")
        with pytest.raises(DuplicateRequest, match="nonce"):
            await guard.check("1.2.3.4", "q2", "abc-123")

    @pytest.mark.asyncio
    async def test_none_nonce_is_noop(self):
        guard = _make_guard(dedup_window=0)
        await guard.check("1.2.3.4", "q", None)
        await guard.check("1.2.3.4", "q", None)

    @pytest.mark.asyncio
    async def test_different_nonces_ok(self):
        guard = _make_guard(dedup_window=0)
        await guard.check("1.2.3.4", "q", "aaa")
        await guard.check("1.2.3.4", "q", "bbb")

    @pytest.mark.asyncio
    async def test_local_nonce_expiry(self):
        guard = _make_guard(dedup_window=0)
        nonce_key = "chatty:dedup:nonce:expired"
        guard._local_nonces[nonce_key] = time.time() - 10
        await guard.check("1.2.3.4", "q", "expired")


# =========================================================================
# RequestGuard — combined checks (local backend)
# =========================================================================


class TestGuardCombined:
    @pytest.mark.asyncio
    async def test_rate_limit_checked_before_dedup(self):
        """Rate limit fires even if fingerprint would also reject."""
        guard = _make_guard(per_ip=1, dedup_window=5)
        await guard.check("1.2.3.4", "hello", None)
        with pytest.raises(RateLimited):
            await guard.check("1.2.3.4", "hello", None)

    @pytest.mark.asyncio
    async def test_all_layers_pass(self):
        guard = _make_guard(per_ip=10, global_limit=10, dedup_window=5)
        await guard.check("1.2.3.4", "hello", "nonce-1")
        await guard.check("1.2.3.4", "goodbye", "nonce-2")
        await guard.check("5.6.7.8", "hello", "nonce-3")


# =========================================================================
# RateLimited exception — scope attribute
# =========================================================================


class TestRateLimitedScope:
    def test_default_scope_is_ip(self):
        exc = RateLimited("test")
        assert exc.scope == "ip"

    def test_custom_scope(self):
        exc = RateLimited("test", scope="global")
        assert exc.scope == "global"

    def test_str(self):
        exc = RateLimited("Per-IP rate limit exceeded")
        assert str(exc) == "Per-IP rate limit exceeded"
