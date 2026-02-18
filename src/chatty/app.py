"""FastAPI application entry point."""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from chatty.api.chat import router as chat_router
from chatty.configs.config import get_app_config
from chatty.core.embedding.cron import embedding_cron_loop
from chatty.core.service.dependency import get_embedding_client
from chatty.infra.concurrency.inbox import get_inbox
from chatty.infra.concurrency.semaphore import get_model_semaphore
from chatty.infra.redis import get_redis_client
from chatty.infra.telemetry import init_telemetry

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup/shutdown events."""
    # Startup logic
    logger.info("Starting Chatty application...")

    # Redis -- tolerant of failure (None -> local fallback).
    redis_client = None
    try:
        redis_client = get_redis_client()
        await redis_client.ping()
    except Exception:
        logger.warning("Redis unavailable -- falling back to local concurrency backend.")

    # Concurrency singletons -- first call initialises each.
    get_inbox(redis_client)
    get_model_semaphore(redis_client)

    # Embedding cron -- pre-embeds persona sections during idle time.
    cron_task: asyncio.Task | None = None
    config = get_app_config()
    if config.chat.agent_name == "rag":
        embedding_client = get_embedding_client()
        cron_task = asyncio.create_task(
            embedding_cron_loop(embedding_client),
            name="embedding-cron",
        )
        logger.info("Embedding cron started (interval=%ds)", config.rag.cron_interval)

    yield

    # Shutdown logic
    logger.info("Shutting down Chatty application...")

    if cron_task is not None:
        cron_task.cancel()
        try:
            await cron_task
        except asyncio.CancelledError:
            pass
        logger.info("Embedding cron stopped.")

    await get_inbox().aclose()
    await get_model_semaphore().aclose()
    if redis_client is not None:
        await redis_client.aclose()


def _build_api_prefix(route_prefix: str) -> str:
    """Build the API prefix from the configurable route prefix.

    Result is ``/api/v1/<route_prefix>`` when *route_prefix* is non-empty,
    otherwise just ``/api/v1``.
    """
    base = "/api/v1"
    if route_prefix:
        return f"{base}/{route_prefix.strip('/')}"
    return base


def get_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    config = get_app_config()

    app = FastAPI(
        title="Chatty",
        description="A persona-driven chatbot with multi-agent pipeline",
        version="0.1.0",
        lifespan=lifespan,
    )

    # TODO: Customize CORS middleware
    #
    # app.add_middleware(
    #     CORSMiddleware,
    #     allow_origins=["*"],  # Configure appropriately for production
    #     allow_credentials=True,
    #     allow_methods=["*"],
    #     allow_headers=["*"],
    # )

    # Include routers with configurable prefix
    api_prefix = _build_api_prefix(config.api.route_prefix)
    app.include_router(chat_router, prefix=api_prefix)

    app.get("/health")(lambda: "ok")

    # OpenTelemetry — instruments FastAPI + httpx; no-op if tracing is disabled.
    init_telemetry(app, config.tracing)

    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

    return app


app = get_app()
