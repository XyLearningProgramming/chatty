"""Global exception handlers — lifespan dependency."""

from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from chatty.core.service.metrics import (
    DEDUP_REJECTIONS_TOTAL,
    INBOX_REJECTIONS_TOTAL,
    RATE_LIMIT_REJECTIONS_TOTAL,
)
from chatty.infra.concurrency.base import (
    AcquireTimeout,
    DuplicateRequest,
    InboxFull,
    RateLimited,
)
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
            content={"detail": str(exc), "code": "ACQUIRE_TIMEOUT"},
        )

    @app.exception_handler(RateLimited)
    async def handle_rate_limited(request: Request, exc: RateLimited) -> JSONResponse:
        RATE_LIMIT_REJECTIONS_TOTAL.labels(scope=exc.scope).inc()
        return JSONResponse(
            status_code=429,
            content={"detail": str(exc), "code": "RATE_LIMITED"},
            headers={"Retry-After": "1"},
        )

    @app.exception_handler(DuplicateRequest)
    async def handle_duplicate_request(
        request: Request, exc: DuplicateRequest
    ) -> JSONResponse:
        DEDUP_REJECTIONS_TOTAL.inc()
        return JSONResponse(
            status_code=409,
            content={"detail": str(exc), "code": "DUPLICATE_REQUEST"},
        )

    @app.exception_handler(InboxFull)
    async def handle_inbox_full(request: Request, exc: InboxFull) -> JSONResponse:
        INBOX_REJECTIONS_TOTAL.inc()
        return JSONResponse(
            status_code=429,
            content={"detail": str(exc), "code": "INBOX_FULL"},
            headers={"Retry-After": "5"},
        )

    yield
