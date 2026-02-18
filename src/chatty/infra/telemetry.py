"""OpenTelemetry bootstrap — tracing initialisation and helpers.

Configures a ``TracerProvider`` with an OTLP HTTP exporter when tracing is
enabled via ``TracingConfig``.  When disabled the module is a graceful no-op
(local dev without a collector).

Auto-instrumentations wired here:

- **FastAPI** (inbound HTTP spans)
- **httpx** (outbound HTTP spans — covers ``langchain-openai`` LLM calls)
- **SQLAlchemy** (DB spans)

Usage::

    # In app.py — before lifespan:
    from chatty.infra.telemetry import init_telemetry
    init_telemetry(app, config.tracing)

    # In app.py — inside lifespan, after engine is created:
    from chatty.infra.telemetry import instrument_sqlalchemy
    instrument_sqlalchemy(engine)

    # In request handlers:
    from chatty.infra.telemetry import get_current_trace_id
    trace_id = get_current_trace_id()  # 32-char hex or None
"""

from __future__ import annotations

import base64
import logging

from opentelemetry import trace
from opentelemetry.trace import format_trace_id

from chatty.configs.system import TracingConfig

logger = logging.getLogger(__name__)

_otel_enabled = False


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
