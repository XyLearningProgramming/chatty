"""Prometheus metrics for the Chatty application.

Custom business metrics that complement the auto-instrumented HTTP
metrics provided by ``prometheus-fastapi-instrumentator``.

All metrics use the ``chatty_`` prefix.
"""

import asyncio
import functools
import logging
import time
from collections.abc import AsyncGenerator, Callable
from typing import Annotated, Any, overload

from fastapi import Depends, FastAPI
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from chatty.configs.config import AppConfig, get_app_config
from chatty.infra.lifespan import get_app

logger = logging.getLogger(__name__)

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
# Embedding metrics
# ---------------------------------------------------------------------------

EMBEDDING_LATENCY_SECONDS = Histogram(
    "chatty_embedding_latency_seconds",
    "Latency of embedding API calls",
    ["operation"],  # "embed" | "search"
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)

EMBEDDING_CRON_RUNS_TOTAL = Counter(
    "chatty_embedding_cron_runs_total",
    "Total embedding cron tick outcomes",
    ["status"],  # "ok" | "error" | "skipped"
)

EMBEDDING_CRON_HINTS_TOTAL = Counter(
    "chatty_embedding_cron_hints_total",
    "Total hints embedded by the cron",
)

# ---------------------------------------------------------------------------
# RAG retrieval metrics
# ---------------------------------------------------------------------------

RAG_RETRIEVAL_LATENCY_SECONDS = Histogram(
    "chatty_rag_retrieval_latency_seconds",
    "End-to-end RAG retrieval latency (embed + search + resolve)",
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10),
)

RAG_SOURCES_RETURNED = Histogram(
    "chatty_rag_sources_returned",
    "Number of sources returned per RAG retrieval",
    buckets=(0, 1, 2, 3, 5, 10),
)

RAG_CACHE_LOOKUPS_TOTAL = Counter(
    "chatty_rag_cache_lookups_total",
    "Total RAG cache lookups by outcome",
    ["result"],  # "hit" | "miss" | "skip"
)

# ---------------------------------------------------------------------------
# Concurrency metrics
# ---------------------------------------------------------------------------

INBOX_OCCUPANCY = Gauge(
    "chatty_inbox_occupancy",
    "Current number of requests admitted into the inbox",
)

INBOX_REJECTIONS_TOTAL = Counter(
    "chatty_inbox_rejections_total",
    "Total inbox rejections (429 responses)",
)

RATE_LIMIT_REJECTIONS_TOTAL = Counter(
    "chatty_rate_limit_rejections_total",
    "Total rate-limit rejections (429 responses)",
    ["scope"],  # "ip" | "global"
)

DEDUP_REJECTIONS_TOTAL = Counter(
    "chatty_dedup_rejections_total",
    "Total duplicate-request rejections (409 responses)",
)

SEMAPHORE_ACQUIRES_TOTAL = Counter(
    "chatty_semaphore_acquires_total",
    "Total semaphore acquire attempts",
    ["result"],  # "ok" | "timeout"
)

SEMAPHORE_WAIT_SECONDS = Histogram(
    "chatty_semaphore_wait_seconds",
    "Time spent waiting for a semaphore slot",
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30),
)

# ---------------------------------------------------------------------------
# Model concurrency gauges
# ---------------------------------------------------------------------------

LLM_CALLS_IN_FLIGHT = Gauge(
    "chatty_llm_calls_in_flight",
    "Number of LLM calls currently in-flight",
    ["model_name"],
)

EMBEDDING_CALLS_IN_FLIGHT = Gauge(
    "chatty_embedding_calls_in_flight",
    "Number of embedding calls currently in-flight",
    ["model_name"],
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


# ---------------------------------------------------------------------------
# Lifespan dependency
# ---------------------------------------------------------------------------


async def build_metrics(
    app: Annotated[FastAPI, Depends(get_app)],
    config: Annotated[AppConfig, Depends(get_app_config)],
) -> AsyncGenerator[None, None]:
    """Set up Prometheus HTTP instrumentation.

    Attaches ``prometheus-fastapi-instrumentator`` middleware and the
    ``/metrics`` endpoint to the FastAPI *app*.
    """
    Instrumentator(
        should_instrument_requests_inprogress=True,
        excluded_handlers=config.tracing.excluded_urls,
    ).instrument(app).expose(app, endpoint="/metrics")

    logger.info("Prometheus metrics initialised")
    yield
