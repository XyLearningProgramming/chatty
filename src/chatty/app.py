"""FastAPI application entry point.

Infrastructure is set up by lifespan dependencies (``build_*``).
Each dep declares what it needs via ``Depends()``; ordering is
resolved automatically by ``inject``.
"""

import logging
from typing import Annotated

from fastapi import Depends, FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.middleware.cors import CORSMiddleware

from chatty.api.chat import router as chat_router
from chatty.api.exceptions import build_exception_handlers
from chatty.api.health import router as health_router
from chatty.configs.config import get_app_config
from chatty.core.embedding.cron import build_cron
from chatty.infra.concurrency.inbox import build_inbox
from chatty.infra.concurrency.semaphore import build_semaphore
from chatty.infra.db.engine import build_db
from chatty.infra.lifespan import inject
from chatty.infra.telemetry import build_telemetry

logger = logging.getLogger(__name__)


@inject
async def lifespan(
    app: FastAPI,
    _db: Annotated[None, Depends(build_db)],
    _telemetry: Annotated[None, Depends(build_telemetry)],
    _inbox: Annotated[None, Depends(build_inbox)],
    _semaphore: Annotated[None, Depends(build_semaphore)],
    _cron: Annotated[None, Depends(build_cron)],
    _exc: Annotated[None, Depends(build_exception_handlers)],
):
    """Application lifespan — deps injected & cleaned up automatically."""
    logger.info("Starting Chatty application...")
    yield
    logger.info("Shutting down Chatty application...")


def _build_api_prefix(route_prefix: str) -> str:
    base = "/api/v1"
    if route_prefix:
        return f"{base}/{route_prefix.strip('/')}"
    return base


def get_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    config = get_app_config()

    app = FastAPI(
        title="Chatty",
        description=(
            "A persona-driven chatbot with multi-agent pipeline"
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    api_prefix = _build_api_prefix(config.api.route_prefix)
    app.include_router(chat_router, prefix=api_prefix)
    app.include_router(health_router)

    Instrumentator().instrument(app).expose(
        app, endpoint="/metrics"
    )

    return app


app = get_app()
