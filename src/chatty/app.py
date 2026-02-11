"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from chatty.api.chat import router as chat_router
from chatty.configs.config import get_app_config

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup/shutdown events."""
    # Startup logic
    logger.info("Starting Chatty application...")

    # TODO: Initialize database connections
    # TODO: Initialize Redis connections
    # TODO: Initialize model server connections

    yield

    # Shutdown logic
    logger.info("Shutting down Chatty application...")


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

    return app


app = get_app()
