"""Concurrency primitives for LLM agent runs.

Three independent layers:

1. **Inbox** (bounded by ``inbox_max_size``): admission control.
   Atomic increment on enter, decrement on leave.  If already at capacity
   the request is rejected immediately (``InboxFull``).

2. **ModelSemaphore** (``max_concurrency`` slots): controls how many LLM
   invocations run concurrently.  Wrapped around the chat model via
   ``GatedChatModel`` so every ``_agenerate`` / ``_astream`` call is
   individually gated.

3. **RequestGuard** — unified anti-flood gate combining per-IP rate
   limiting, global QPS cap, nonce dedup, and fingerprint dedup in a
   single Redis pipeline call.

Two concrete backends are provided for each layer:

* Redis — distributed, uses Lua scripts for atomicity and Pub/Sub for
  event-driven slot notification.  Keys carry a TTL for crash-safety.
* Local — in-process, backed by ``asyncio`` primitives.  Used
  automatically when Redis is unavailable.
"""

from .base import (
    AcquireTimeout,
    ClientDisconnected,
    DuplicateRequest,
    InboxFull,
    RateLimited,
)
from .guards import (
    RequestGuard,
    build_request_guard,
    enforce_inbox,
    enforce_request_guards,
    get_request_guard,
)
from .inbox import Inbox, build_inbox, get_inbox
from .real_ip import get_real_ip
from .semaphore import ModelSemaphore, build_semaphore, get_model_semaphore

__all__ = [
    "AcquireTimeout",
    "ClientDisconnected",
    "DuplicateRequest",
    "Inbox",
    "InboxFull",
    "ModelSemaphore",
    "RateLimited",
    "RequestGuard",
    "build_inbox",
    "build_request_guard",
    "build_semaphore",
    "enforce_inbox",
    "enforce_request_guards",
    "get_inbox",
    "get_model_semaphore",
    "get_real_ip",
    "get_request_guard",
]
