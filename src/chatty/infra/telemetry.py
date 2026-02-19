"""OpenTelemetry bootstrap — tracing initialisation and helpers.

Configures a ``TracerProvider`` with an OTLP HTTP exporter when tracing is
enabled via ``TracingConfig``.  When disabled the module is a graceful no-op
(local dev without a collector).

Auto-instrumentations wired here:

- **FastAPI** (inbound HTTP spans)
- **httpx** (outbound HTTP spans — covers ``langchain-openai`` LLM calls)
- **SQLAlchemy** (DB spans)

``build_telemetry`` is a lifespan dependency (same pattern as ``build_db``,
``build_inbox``, etc.).  It explicitly ``Depends(build_db)`` (imported
from the leaf module ``chatty.infra.db_engine``) so the engine is
guaranteed to exist for SQLAlchemy instrumentation.

Usage::

    # In request handlers:
    from chatty.infra.telemetry import get_current_trace_id, tracer
    trace_id = get_current_trace_id()  # 32-char hex or None

    with tracer.start_as_current_span(SPAN_RAG_PIPELINE) as span:
        ...
"""

from __future__ import annotations

import base64
import logging
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends, FastAPI
from opentelemetry import trace
from opentelemetry.trace import format_trace_id

from chatty.configs.config import AppConfig, get_app_config
from chatty.configs.system import TracingConfig
from chatty.infra.db_engine import build_db
from chatty.infra.lifespan import get_app

logger = logging.getLogger(__name__)

_otel_enabled = False

tracer = trace.get_tracer("chatty")

# ---------------------------------------------------------------------------
# Span names — single source of truth for all custom spans
# ---------------------------------------------------------------------------

SPAN_RAG_PIPELINE = "rag.pipeline"
SPAN_RAG_RETRIEVE = "rag.retrieve"
SPAN_RAG_CACHE_CHECK = "rag.cache_check"
SPAN_EMBEDDING_EMBED = "embedding.embed"
SPAN_EMBEDDING_SEARCH = "embedding.search"
SPAN_EMBEDDING_CRON_TICK = "embedding.cron_tick"
SPAN_TOOL_URL_DISPATCHER = "tool.url_dispatcher"
SPAN_SEMAPHORE_SLOT = "semaphore.slot"
SPAN_INBOX_ENTER = "inbox.enter"
SPAN_HISTORY_LOAD = "history.load"
SPAN_SSE_STREAM = "sse.stream"

# ---------------------------------------------------------------------------
# Span attribute keys
# ---------------------------------------------------------------------------

ATTR_RAG_QUERY_LEN = "rag.query_len"
ATTR_RAG_THRESHOLD = "rag.threshold"
ATTR_RAG_TOP_K = "rag.top_k"
ATTR_RAG_RESULT_COUNT = "rag.result_count"
ATTR_RAG_CACHE_HIT = "rag.cache_hit"

ATTR_EMBEDDING_MODEL = "embedding.model"
ATTR_EMBEDDING_TEXT_LEN = "embedding.text_len"
ATTR_EMBEDDING_TOP_K = "embedding.top_k"
ATTR_EMBEDDING_THRESHOLD = "embedding.threshold"
ATTR_EMBEDDING_RESULT_COUNT = "embedding.result_count"

ATTR_CRON_EMBEDDED = "cron.embedded"
ATTR_CRON_TOTAL_PENDING = "cron.total_pending"

ATTR_TOOL_SOURCE = "tool.source"
ATTR_TOOL_ERROR = "tool.error"

ATTR_SEMAPHORE_TIMEOUT = "semaphore.timeout"

ATTR_INBOX_POSITION = "inbox.position"
ATTR_INBOX_REJECTED = "inbox.rejected"

ATTR_HISTORY_CONVERSATION_ID = "history.conversation_id"
ATTR_HISTORY_MESSAGE_COUNT = "history.message_count"

ATTR_SSE_ERROR_CODE = "sse.error_code"
ATTR_SSE_EVENT_COUNTS = "sse.event_counts"
ATTR_SSE_SERVICE = "sse.service"


def init_telemetry(
    app: object | None = None,
    settings: TracingConfig | None = None,
) -> None:
    """Initialise the OTEL ``TracerProvider`` and auto-instrumentations.

    Parameters
    ----------
    app:
        The FastAPI application instance.  Passed to the FastAPI
        instrumentor so it can attach ASGI middleware.
    settings:
        Tracing configuration.  When ``None`` or ``enabled`` is
        ``False``, this function is a no-op.
    """
    global _otel_enabled  # noqa: PLW0603

    if settings is None or not settings.enabled:
        logger.info("OpenTelemetry tracing disabled.")
        return

    if not settings.endpoint or not settings.username or not settings.password:
        logger.warning(
            "Tracing enabled but endpoint/credentials not configured — "
            "skipping OpenTelemetry setup."
        )
        return

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

    resource = Resource.create({"service.name": settings.service_name})

    sampler = ParentBased(root=TraceIdRatioBased(settings.sample_rate))
    provider = TracerProvider(resource=resource, sampler=sampler)

    credentials = f"{settings.username}:{settings.password}"
    encoded = base64.b64encode(credentials.encode()).decode()
    headers = {"Authorization": f"Basic {encoded}"}

    exporter = OTLPSpanExporter(
        endpoint=settings.endpoint,
        headers=headers,
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    # --- Auto-instrumentations ---

    if app is not None:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        excluded = ",".join(settings.excluded_urls) if settings.excluded_urls else ""
        FastAPIInstrumentor.instrument_app(app, excluded_urls=excluded)

    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

    HTTPXClientInstrumentor().instrument()

    _otel_enabled = True
    logger.info(
        "OpenTelemetry tracing initialised (service=%s).", settings.service_name
    )


def instrument_sqlalchemy(engine: object) -> None:
    """Instrument a SQLAlchemy engine for DB span tracing.

    Call this *after* the engine has been created (inside the lifespan).
    No-op when OTEL is not enabled.
    """
    if not _otel_enabled:
        return

    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

    sync_engine = getattr(engine, "sync_engine", engine)
    SQLAlchemyInstrumentor().instrument(engine=sync_engine)
    logger.info("SQLAlchemy engine instrumented for OTEL tracing.")


def get_current_trace_id() -> str | None:
    """Return the active OTEL trace ID as a 32-char hex string, or ``None``.

    Returns ``None`` when:
    - OTEL is not enabled, or
    - there is no active span, or
    - the span carries the invalid (all-zero) trace ID.
    """
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if ctx is None or not ctx.is_valid:
        return None
    return format_trace_id(ctx.trace_id)


# ---------------------------------------------------------------------------
# Lifespan dependency
# ---------------------------------------------------------------------------


async def build_telemetry(
    app: Annotated[FastAPI, Depends(get_app)],
    config: Annotated[AppConfig, Depends(get_app_config)],
    _db: Annotated[None, Depends(build_db)],
) -> AsyncGenerator[None, None]:
    """Initialise OTEL tracing + SQLAlchemy instrumentation.

    ``build_db`` is imported from the leaf module
    ``chatty.infra.db_engine`` (not the barrel ``db.__init__``)
    so there is no circular import.
    """
    init_telemetry(app, config.tracing)
    instrument_sqlalchemy(app.state.engine)
    yield
