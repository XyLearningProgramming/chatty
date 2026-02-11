"""Concurrency gate for LLM agent runs.

One gate, two layers inside:

1. **Inbox counter** (bounded by ``inbox_max_size``): admission control.
   Atomic increment on enter, decrement on leave.  If already at capacity
   the request is rejected immediately (``GateFull``).

2. **Semaphore** (``max_concurrency`` slots): controls how many admitted
   requests actively run the LLM agent.  Requests wait on the semaphore
   after being admitted.

Two concrete backends are provided:

* ``RedisConcurrencyBackend`` – distributed, uses Lua scripts for
  atomicity and Pub/Sub for event-driven slot notification.
  Keys carry a TTL for crash-safety.
* ``LocalConcurrencyBackend`` – in-process, backed by ``asyncio``
  primitives.  Used automatically when Redis is unavailable.
"""

from .base import ClientDisconnected, GateFull
from .gate import ConcurrencyGate, DisconnectCheck, get_concurrency_gate

__all__ = [
    "ClientDisconnected",
    "ConcurrencyGate",
    "DisconnectCheck",
    "GateFull",
    "get_concurrency_gate",
]
