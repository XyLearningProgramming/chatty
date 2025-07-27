"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from chatty.api.chat import router as chat_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup/shutdown events."""
    # Startup logic
    print("Starting Chatty application...")

    # TODO: Initialize vector database connections
    # TODO: Initialize Redis connections
    # TODO: Initialize model server connections
    # TODO: Load golden dataset for memory

    yield

    # Shutdown logic
    print("Shutting down Chatty application...")
    # TODO: Cleanup connections


def get_app() -> FastAPI:
    """Create and configure the FastAPI application."""
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

    # Include routers
    app.include_router(chat_router)

    return app


app = get_app()
