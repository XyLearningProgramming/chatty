"""Prometheus metrics for the Chatty application.

Custom business metrics that complement the auto-instrumented HTTP
metrics provided by ``prometheus-fastapi-instrumentator``.

All metrics use the ``chatty_`` prefix.
"""

import logging

from fastapi import FastAPI
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from chatty.configs.config import AppConfig

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

SSE_STREAM_OUTCOMES_TOTAL = Counter(
    "chatty_sse_stream_outcomes_total",
    "Final outcome of each SSE stream",
    ["code"],
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
# HTTP instrumentation (must run before app starts)
# ---------------------------------------------------------------------------


def build_metrics(app: FastAPI, config: AppConfig) -> None:
    """Attach Prometheus HTTP middleware and ``/metrics`` endpoint.

    Called at app-construction time (inside ``get_app``) because
    Starlette forbids adding middleware after the ASGI app has started.
    """
    Instrumentator(
        should_instrument_requests_inprogress=True,
        excluded_handlers=config.tracing.excluded_urls,
    ).instrument(app).expose(app, endpoint="/metrics")

    logger.info("Prometheus metrics initialised")
