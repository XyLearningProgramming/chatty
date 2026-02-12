"""OpenTelemetry bootstrap — tracing initialisation and helpers.

Configures a ``TracerProvider`` with an OTLP exporter when the standard
``OTEL_EXPORTER_OTLP_ENDPOINT`` environment variable is set.  When the
variable is absent the module is a graceful no-op (local dev without a
collector).

The OTEL SDK reads its configuration from well-known env vars
automatically:

- ``OTEL_EXPORTER_OTLP_ENDPOINT``
- ``OTEL_EXPORTER_OTLP_HEADERS``
- ``OTEL_SERVICE_NAME``

Auto-instrumentations wired here:

- **FastAPI** (inbound HTTP spans)
- **httpx** (outbound HTTP spans — covers ``langchain-openai`` LLM calls)
- **SQLAlchemy** (DB spans)

Usage::

    # In app.py — before lifespan:
    from chatty.infra.telemetry import init_telemetry
    init_telemetry(app)

    # In app.py — inside lifespan, after engine is created:
    from chatty.infra.telemetry import instrument_sqlalchemy
    instrument_sqlalchemy(engine)

    # In request handlers:
    from chatty.infra.telemetry import get_current_trace_id
    trace_id = get_current_trace_id()  # 32-char hex or None
"""

import logging
import os

from opentelemetry import trace
from opentelemetry.trace import format_trace_id

logger = logging.getLogger(__name__)

# Sentinel: is the OTEL SDK actually exporting?
_otel_enabled = False


def _is_otel_configured() -> bool:
    """Return ``True`` when the OTLP endpoint env var is set."""
    return bool(os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"))


def init_telemetry(app: object | None = None) -> None:
    """Initialise the OTEL ``TracerProvider`` and auto-instrumentations.

    Parameters
    ----------
    app:
        The FastAPI application instance.  Passed to the FastAPI
        instrumentor so it can attach ASGI middleware.

    This function is idempotent — safe to call more than once.
    """
    global _otel_enabled  # noqa: PLW0603

    if not _is_otel_configured():
        logger.info(
            "OTEL_EXPORTER_OTLP_ENDPOINT not set — "
            "OpenTelemetry tracing disabled."
        )
        return

    # Import heavy SDK pieces only when actually needed.
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    service_name = os.environ.get("OTEL_SERVICE_NAME", "chatty")
    resource = Resource.create({"service.name": service_name})

    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)

    # --- Auto-instrumentations ---

    # FastAPI (inbound)
    if app is not None:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)

    # httpx (outbound — covers langchain-openai LLM calls)
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

    HTTPXClientInstrumentor().instrument()

    _otel_enabled = True
    logger.info("OpenTelemetry tracing initialised (service=%s).", service_name)


def instrument_sqlalchemy(engine: object) -> None:
    """Instrument a SQLAlchemy engine for DB span tracing.

    Call this *after* the engine has been created (inside the lifespan).
    No-op when OTEL is not enabled.
    """
    if not _otel_enabled:
        return

    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

    # The async engine exposes its sync counterpart via .sync_engine.
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
