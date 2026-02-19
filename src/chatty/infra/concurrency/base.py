"""Concurrency primitives: abstract backends and exceptions."""

from __future__ import annotations

from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class InboxFull(Exception):
    """Raised when the inbox is at capacity and cannot admit more requests."""


class AcquireTimeout(Exception):
    """Raised when a concurrency slot cannot be acquired within the timeout."""


class ClientDisconnected(Exception):
    """Raised when the client disconnects while waiting for a slot."""


class RateLimited(Exception):
    """Raised when a request exceeds the rate limit (per-IP or global)."""

    def __init__(self, message: str, *, scope: str = "ip") -> None:
        super().__init__(message)
        self.scope = scope


class DuplicateRequest(Exception):
    """Raised when a duplicate request is detected (nonce or fingerprint)."""

# ---------------------------------------------------------------------------
# Abstract backends
# ---------------------------------------------------------------------------


class InboxBackend(ABC):
    """Interface for inbox admission-control backends."""

    @abstractmethod
    async def enter(self) -> int:
        """Admit a request into the inbox.

        Returns:
            The current inbox occupancy *after* entering.

        Raises:
            InboxFull: when the inbox is already at ``inbox_max_size``.
        """

    @abstractmethod
    async def leave(self) -> None:
        """Decrement the inbox counter (request finished or errored)."""

    @abstractmethod
    async def aclose(self) -> None:
        """Release any resources held by the backend."""


class SemaphoreBackend(ABC):
    """Interface for concurrency-semaphore backends."""

    @abstractmethod
    async def acquire(self) -> None:
        """Wait until a concurrency slot is available, then claim it."""

    @abstractmethod
    async def release(self) -> None:
        """Free a concurrency slot so the next waiter can proceed."""

    @abstractmethod
    async def aclose(self) -> None:
        """Release any resources held by the backend."""
