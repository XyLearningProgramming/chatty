"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.chat import router as chat_router
from .config import AppConfig


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


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Chatty",
        description="A persona-driven chatbot with multi-agent pipeline",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(chat_router)

    return app


app = create_app()


def main() -> None:
    """Run the application with uvicorn."""
    config = AppConfig()
    uvicorn.run(
        "chatty.app:app",
        host=config.host,
        port=config.port,
        reload=True,  # Set to False in production
    )


if __name__ == "__main__":
    main()
