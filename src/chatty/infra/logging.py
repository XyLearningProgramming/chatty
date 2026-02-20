"""Structured logging bootstrap.

Configures the root logger so every ``logging.getLogger(__name__)`` call
across the app (and uvicorn) emits either:

* **JSON lines** (``json=True``, default) — machine-parseable by Loki /
  Promtail with ``| json`` in LogQL queries.
* **Human-readable** (``json=False``) — coloured, timestamp-prefixed
  lines for local development.

When OpenTelemetry tracing is active the current ``trace_id`` and
``span_id`` are injected into every log record so Grafana can jump
from a log line straight to the matching trace.
"""

from __future__ import annotations

import logging
import sys

from opentelemetry import trace

from chatty.configs.system import LoggingConfig


class _TraceContextFilter(logging.Filter):
    """Injects OTEL trace/span IDs into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx and ctx.is_valid:
            record.trace_id = format(ctx.trace_id, "032x")  # type: ignore[attr-defined]
            record.span_id = format(ctx.span_id, "016x")  # type: ignore[attr-defined]
        else:
            record.trace_id = ""  # type: ignore[attr-defined]
            record.span_id = ""  # type: ignore[attr-defined]
        return True


_DEV_FORMAT = "%(levelprefix)s %(asctime)s %(name)s  %(message)s"
_DEV_DATEFMT = "%H:%M:%S"


def setup_logging(config: LoggingConfig | None = None) -> None:
    """Configure the root logger (call once at startup, before lifespan)."""
    if config is None:
        config = LoggingConfig()

    level = config.level.upper()
    root = logging.getLogger()
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_TraceContextFilter())

    if config.json_output:
        from pythonjsonlogger.json import JsonFormatter

        formatter = JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s "
            "%(trace_id)s %(span_id)s",
            rename_fields={
                "asctime": "timestamp",
                "levelname": "level",
                "name": "logger",
            },
            defaults={"trace_id": "", "span_id": ""},
        )
    else:
        from uvicorn.logging import DefaultFormatter

        formatter = DefaultFormatter(
            fmt=_DEV_FORMAT,
            datefmt=_DEV_DATEFMT,
            use_colors=True,
        )

    handler.setFormatter(formatter)

    root.handlers = [handler]

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvi = logging.getLogger(name)
        uvi.handlers = [handler]
        uvi.propagate = False

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("opentelemetry").setLevel(logging.WARNING)
