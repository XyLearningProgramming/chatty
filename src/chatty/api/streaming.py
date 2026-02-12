"""Reusable SSE streaming infrastructure.

Wraps an async generator of domain ``StreamEvent`` objects into a
formatted SSE byte-stream with timeout enforcement, error boundaries,
and cleanup.  Business-logic generators stay free of SSE formatting
and exception handling — this module handles all of that.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from datetime import timedelta

from chatty.core.service.models import ErrorEvent, StreamEvent
from chatty.infra.concurrency import AcquireTimeout, ClientDisconnected

from .models import format_error_sse, format_sse

logger = logging.getLogger(__name__)


async def sse_stream(
    events: AsyncGenerator[StreamEvent, None],
    *,
    request_timeout: timedelta,
    on_finish: Callable[[], Awaitable[None]] | None = None,
) -> AsyncGenerator[str, None]:
    """Format domain events as SSE with timeout and error handling.

    Parameters
    ----------
    events:
        Async generator of ``StreamEvent`` instances (business logic).
    request_timeout:
        Wall-clock timeout for the entire streaming lifecycle.
    on_finish:
        Optional async callback invoked in the ``finally`` block
        (e.g. ``inbox.leave``).

    Yields
    ------
    SSE-formatted strings (``data: {...}\\n\\n``).
    """
    try:
        async with asyncio.timeout(request_timeout.total_seconds()):
            async for event in events:
                yield format_sse(event)

    except AcquireTimeout:
        logger.warning("Model semaphore acquire timeout — no slot available.")
        yield format_sse(
            ErrorEvent(
                message="Model is busy. Try again later.",
                code="MODEL_BUSY",
            )
        )
    except TimeoutError:
        logger.warning("Request timed out after %s.", request_timeout)
        yield format_sse(
            ErrorEvent(message="Request timed out.", code="REQUEST_TIMEOUT")
        )
    except ClientDisconnected:
        logger.debug("Client disconnected while waiting for slot.")
    except asyncio.CancelledError:
        raise
    except Exception as e:
        yield format_error_sse(e)
    finally:
        if on_finish:
            await on_finish()
