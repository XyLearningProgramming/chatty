"""Chat API endpoint implementation."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Annotated, AsyncGenerator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from chatty.configs.config import get_app_config
from chatty.configs.system import APIConfig
from chatty.core.service import (
    ChatService,
    get_chat_service,
)
from chatty.core.service.models import DequeuedEvent, ErrorEvent, QueuedEvent
from chatty.infra.concurrency import (
    ClientDisconnected,
    ConcurrencyGate,
    GateFull,
    get_concurrency_gate,
)

from .models import (
    ChatRequest,
    format_error_sse,
    format_sse,
)

logger = logging.getLogger(__name__)

STREAMING_RESPONSE_MEDIA_TYPE = "text/plain"
STREAMING_RESPONSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Access-Control-Allow-Headers": "Cache-Control",
}

router = APIRouter(tags=["chat"])


def _get_api_config() -> APIConfig:
    return get_app_config().api


async def _stream_chat_response(
    chat_request: ChatRequest,
    service: ChatService,
    gate: ConcurrencyGate,
    position: int,
    is_disconnected: Callable[[], Awaitable[bool]],
    request_timeout: timedelta,
) -> AsyncGenerator[str, None]:
    """Stream agent events as SSE, gated by the concurrency semaphore.

    The caller has already admitted the request into the inbox (position
    is known).  This generator:

    1. Yields ``QueuedEvent`` to notify the client.
    2. Acquires a concurrency slot via ``gate.slot()``.  If the
       configured ``acquire_timeout`` is exceeded, yields an error
       event with type ``TOO_MANY_REQUESTS`` and closes the stream.
    3. Yields ``DequeuedEvent`` once the slot is acquired.
    4. Delegates to ``service.stream_response()``, checking for
       client disconnect between each streamed event.
    5. Always releases the slot (if held) and leaves the inbox.

    The entire flow is wrapped in ``asyncio.timeout()`` so requests
    that exceed ``request_timeout`` are cancelled with an error event.
    """
    try:
        async with asyncio.timeout(request_timeout.total_seconds()):
            # 1. Notify client it is queued.
            yield format_sse(QueuedEvent(position=position))

            # 2-3. Acquire a concurrency slot (disconnect-aware, with timeout).
            async with gate.slot(disconnected=is_disconnected):
                yield format_sse(DequeuedEvent())

                # 4. Stream the actual agent response.
                async for event in service.stream_response(
                    chat_request.query
                ):
                    if await is_disconnected():
                        logger.debug("Client disconnected during streaming.")
                        return
                    yield format_sse(event)

    except GateFull:
        logger.warning("Acquire timeout — no concurrency slot available.")
        yield format_sse(
            ErrorEvent(
                message="Too many requests in flight. Try again later.",
                code="TOO_MANY_REQUESTS",
            )
        )
    except TimeoutError:
        logger.warning("Request timed out after %s.", request_timeout)
        yield format_sse(
            ErrorEvent(message="Request timed out.", code="REQUEST_TIMEOUT")
        )
    except ClientDisconnected:
        logger.debug("Client disconnected while waiting for slot.")
    except asyncio.CancelledError:
        raise
    except Exception as e:
        yield format_error_sse(e)
    finally:
        # 5. Always leave the inbox.
        await gate.leave()


@router.post("/chat")
async def chat(
    request: Request,
    chat_request: ChatRequest,
    chat_service: Annotated[ChatService, Depends(get_chat_service)],
    api_config: Annotated[APIConfig, Depends(_get_api_config)],
) -> StreamingResponse | JSONResponse:
    """Process a chat request and return a streaming response.

    The response is a stream of Server-Sent Events, where each event
    is a JSON object with a ``type`` discriminator:

    - **queued**: request admitted into the concurrency gate
    - **dequeued**: concurrency slot acquired, agent starting
    - **thinking**: agent internal reasoning
    - **content**: user-facing streamed text tokens
    - **tool_call**: tool invocation lifecycle (started / completed / error)
    - **error**: stream-level error (including ``TOO_MANY_REQUESTS``,
      ``REQUEST_TIMEOUT``)

    Returns **429** when the concurrency gate inbox is full.
    """
    gate = get_concurrency_gate()

    # Admission control — reject early with a proper HTTP status code.
    try:
        position = await gate.enter()
    except GateFull:
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests in flight. Try again later."},
        )

    return StreamingResponse(
        _stream_chat_response(
            chat_request,
            chat_service,
            gate,
            position,
            is_disconnected=request.is_disconnected,
            request_timeout=api_config.request_timeout,
        ),
        media_type=STREAMING_RESPONSE_MEDIA_TYPE,
        headers=STREAMING_RESPONSE_HEADERS,
    )
