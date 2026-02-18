"""Global exception handlers — lifespan dependency."""

from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from chatty.infra.concurrency.base import AcquireTimeout
from chatty.infra.lifespan import get_app


async def build_exception_handlers(
    app: Annotated[FastAPI, Depends(get_app)],
) -> AsyncGenerator[None, None]:
    """Register custom exception handlers on ``app``."""

    @app.exception_handler(AcquireTimeout)
    async def handle_acquire_timeout(
        request: Request, exc: AcquireTimeout
    ) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"detail": str(exc)},
        )

    yield
