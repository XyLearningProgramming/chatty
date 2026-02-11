"""Concurrency gate primitives: abstract backend, exceptions."""

from __future__ import annotations

from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class GateFull(Exception):
    """Raised when the inbox is at capacity and cannot admit more requests."""


class ClientDisconnected(Exception):
    """Raised when the client disconnects while waiting for a slot."""


# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------


class ConcurrencyBackend(ABC):
    """Interface that every concurrency backend must implement."""

    @abstractmethod
    async def enter(self) -> int:
        """Admit a request into the inbox.

        Returns:
            The current inbox occupancy *after* entering.

        Raises:
            GateFull: when the inbox is already at ``inbox_max_size``.
        """

    @abstractmethod
    async def acquire(self) -> None:
        """Wait until a concurrency slot is available, then claim it."""

    @abstractmethod
    async def release(self) -> None:
        """Free a concurrency slot so the next waiter can proceed."""

    @abstractmethod
    async def leave(self) -> None:
        """Decrement the inbox counter (request finished or errored)."""

    @abstractmethod
    async def aclose(self) -> None:
        """Release any resources held by the backend."""
