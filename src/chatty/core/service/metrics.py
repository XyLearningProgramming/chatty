"""Prometheus metrics for the Chatty application.

Custom business metrics that complement the auto-instrumented HTTP
metrics provided by ``prometheus-fastapi-instrumentator``.

All metrics use the ``chatty_`` prefix.
"""

import asyncio
import functools
import time
from collections.abc import AsyncGenerator, Callable
from typing import Any, overload

from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------------------
# Chat session metrics
# ---------------------------------------------------------------------------

CHAT_SESSIONS_ACTIVE = Gauge(
    "chatty_chat_sessions_active",
    "Number of streaming chat sessions currently in progress",
    ["service"],
)

CHAT_SESSIONS_TOTAL = Counter(
    "chatty_chat_sessions_total",
    "Total number of chat sessions started",
    ["service", "status"],  # "ok" | "error" | "cancelled"
)

CHAT_SESSION_DURATION_SECONDS = Histogram(
    "chatty_chat_session_duration_seconds",
    "End-to-end duration of a chat streaming session",
    ["service"],
    buckets=(0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600, 900, 1800),
)

# ---------------------------------------------------------------------------
# Stream event metrics
# ---------------------------------------------------------------------------

STREAM_EVENTS_TOTAL = Counter(
    "chatty_stream_events_total",
    "Total stream events emitted, by event type",
    ["service", "event_type"],  # thinking | content | tool_call | error
)

# ---------------------------------------------------------------------------
# Tool call metrics
# ---------------------------------------------------------------------------

TOOL_CALLS_TOTAL = Counter(
    "chatty_tool_calls_total",
    "Total tool invocations, by tool name and outcome",
    ["service", "tool_name", "status"],  # status: started | completed | error
)


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def _make_wrapper(
    fn: Callable[..., AsyncGenerator], service: str
) -> Callable[..., AsyncGenerator]:
    """Build the observing async-generator wrapper for *fn*."""

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> AsyncGenerator:
        CHAT_SESSIONS_ACTIVE.labels(service=service).inc()
        start = time.monotonic()
        status = "ok"
        try:
            async for event in fn(*args, **kwargs):
                STREAM_EVENTS_TOTAL.labels(
                    service=service, event_type=event.type
                ).inc()
                if event.type == "tool_call":
                    TOOL_CALLS_TOTAL.labels(
                        service=service,
                        tool_name=event.name,
                        status=event.status,
                    ).inc()
                yield event
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        except Exception:
            status = "error"
            STREAM_EVENTS_TOTAL.labels(
                service=service, event_type="error"
            ).inc()
            raise
        finally:
            CHAT_SESSIONS_ACTIVE.labels(service=service).dec()
            CHAT_SESSIONS_TOTAL.labels(service=service, status=status).inc()
            CHAT_SESSION_DURATION_SECONDS.labels(service=service).observe(
                time.monotonic() - start
            )

    return wrapper


@overload
def observe_stream_response(
    fn: Callable[..., AsyncGenerator],
) -> Callable[..., AsyncGenerator]: ...


@overload
def observe_stream_response(
    service_name: str = "",
) -> Callable[[Callable[..., AsyncGenerator]], Callable[..., AsyncGenerator]]: ...


def observe_stream_response(
    fn_or_name: Callable[..., AsyncGenerator] | str = "",
) -> (
    Callable[..., AsyncGenerator]
    | Callable[[Callable[..., AsyncGenerator]], Callable[..., AsyncGenerator]]
):
    """Decorator for ``stream_response`` that records all chat metrics.

    Can be used bare or with a *service_name* argument::

        @observe_stream_response          # service=""
        @observe_stream_response("one_step")

    Wraps an async-generator method to track:
    - active session gauge (inc on entry, dec on exit)
    - session total counter by outcome (ok / error / cancelled)
    - session duration histogram
    - per-event counters (event type, tool name + status)

    All counters carry a ``service`` label set to *service_name*.
    """
    # Called as @observe_stream_response (no parens) — fn_or_name is the fn.
    if callable(fn_or_name):
        return _make_wrapper(fn_or_name, service="")

    # Called as @observe_stream_response("one_step") — return a decorator.
    service = fn_or_name

    def decorator(fn: Callable[..., AsyncGenerator]) -> Callable[..., AsyncGenerator]:
        return _make_wrapper(fn, service=service)

    return decorator
