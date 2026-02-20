"""Reusable SSE streaming infrastructure.

Wraps an async generator of domain ``StreamEvent`` objects into a
formatted SSE byte-stream with timeout enforcement, error boundaries,
cleanup, and unified metrics/tracing.  Business-logic generators stay
free of SSE formatting and exception handling — this module handles
all of that.
"""

import asyncio
import json
import logging
import time
from collections import Counter as EventCounter
from collections.abc import AsyncGenerator, Awaitable, Callable
from datetime import timedelta

from openai import APIConnectionError

from chatty.core.service.metrics import (
    CHAT_SESSION_DURATION_SECONDS,
    CHAT_SESSIONS_ACTIVE,
    CHAT_SESSIONS_TOTAL,
    SSE_STREAM_OUTCOMES_TOTAL,
    STREAM_EVENTS_TOTAL,
    TOOL_CALLS_TOTAL,
)
from chatty.core.service.models import ErrorEvent, StreamEvent, ToolCallEvent
from chatty.infra.concurrency import AcquireTimeout, ClientDisconnected
from chatty.infra.telemetry import (
    ATTR_SSE_ERROR_CODE,
    ATTR_SSE_EVENT_COUNTS,
    ATTR_SSE_SERVICE,
    SPAN_SSE_STREAM,
    tracer,
)

from .models import format_error_sse, format_sse

logger = logging.getLogger(__name__)


async def sse_stream(
    events: AsyncGenerator[StreamEvent, None],
    *,
    request_timeout: timedelta,
    service_name: str = "",
    send_traceback: bool = False,
    on_finish: Callable[[], Awaitable[None]] | None = None,
) -> AsyncGenerator[str, None]:
    """Format domain events as SSE with timeout, error handling, and metrics.

    Parameters
    ----------
    events:
        Async generator of ``StreamEvent`` instances (business logic).
    request_timeout:
        Wall-clock timeout for the entire streaming lifecycle.
    service_name:
        Logical service label for Prometheus metrics (e.g. ``"rag"``).
    on_finish:
        Optional async callback invoked in the ``finally`` block
        (e.g. ``inbox.leave``).

    Yields
    ------
    SSE-formatted strings (``data: {...}\\n\\n``).
    """
    with tracer.start_as_current_span(SPAN_SSE_STREAM) as span:
        span.set_attribute(ATTR_SSE_SERVICE, service_name)
        code = "ok"
        event_counts: EventCounter[str] = EventCounter()
        CHAT_SESSIONS_ACTIVE.labels(service=service_name).inc()
        start = time.monotonic()
        try:
            async with asyncio.timeout(request_timeout.total_seconds()):
                async for event in events:
                    event_counts[event.type] += 1
                    STREAM_EVENTS_TOTAL.labels(
                        service=service_name, event_type=event.type
                    ).inc()
                    if isinstance(event, ToolCallEvent):
                        TOOL_CALLS_TOTAL.labels(
                            service=service_name,
                            tool_name=event.name,
                            status=event.status,
                        ).inc()
                    yield format_sse(event)

        except AcquireTimeout:
            code = "MODEL_BUSY"
            logger.warning("Model semaphore acquire timeout — no slot available.")
            yield format_sse(
                ErrorEvent(
                    message="Model is busy. Try again later.",
                    code=code,
                )
            )
        except APIConnectionError:
            code = "MODEL_UNREACHABLE"
            logger.warning("LLM/embedding model unreachable.")
            yield format_sse(
                ErrorEvent(
                    message="Model is temporarily unavailable. Please try again later.",
                    code=code,
                )
            )
        except TimeoutError:
            code = "REQUEST_TIMEOUT"
            logger.warning("Request timed out after %s.", request_timeout)
            yield format_sse(ErrorEvent(message="Request timed out.", code=code))
        except ClientDisconnected:
            code = "CLIENT_DISCONNECTED"
            logger.debug("Client disconnected while waiting for slot.")
        except asyncio.CancelledError:
            code = "CANCELLED"
            raise
        except Exception as e:
            code = "PROCESSING_ERROR"
            span.record_exception(e)
            logger.warning("Unexpected error in SSE stream", exc_info=True)
            yield format_error_sse(e, send_traceback=send_traceback)
        finally:
            span.set_attribute(ATTR_SSE_ERROR_CODE, code)
            span.set_attribute(ATTR_SSE_EVENT_COUNTS, json.dumps(event_counts))
            SSE_STREAM_OUTCOMES_TOTAL.labels(code=code).inc()
            CHAT_SESSIONS_ACTIVE.labels(service=service_name).dec()
            CHAT_SESSIONS_TOTAL.labels(service=service_name, status=code).inc()
            CHAT_SESSION_DURATION_SECONDS.labels(service=service_name).observe(
                time.monotonic() - start
            )
            if on_finish:
                await on_finish()
