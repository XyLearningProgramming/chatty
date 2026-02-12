"""Concurrency primitives for LLM agent runs.

Two independent layers:

1. **Inbox** (bounded by ``inbox_max_size``): admission control.
   Atomic increment on enter, decrement on leave.  If already at capacity
   the request is rejected immediately (``InboxFull``).

2. **ModelSemaphore** (``max_concurrency`` slots): controls how many LLM
   invocations run concurrently.  Wrapped around the chat model via
   ``GatedChatModel`` so every ``_agenerate`` / ``_astream`` call is
   individually gated.

Two concrete backends are provided for each layer:

* Redis — distributed, uses Lua scripts for atomicity and Pub/Sub for
  event-driven slot notification.  Keys carry a TTL for crash-safety.
* Local — in-process, backed by ``asyncio`` primitives.  Used
  automatically when Redis is unavailable.
"""

from .base import AcquireTimeout, ClientDisconnected, InboxFull
from .inbox import Inbox, get_inbox
from .semaphore import ModelSemaphore, get_model_semaphore

__all__ = [
    "AcquireTimeout",
    "ClientDisconnected",
    "Inbox",
    "InboxFull",
    "ModelSemaphore",
    "get_inbox",
    "get_model_semaphore",
]
