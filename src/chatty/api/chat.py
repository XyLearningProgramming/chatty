"""Chat API endpoint implementation."""

import asyncio
from typing import Annotated, AsyncGenerator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from chatty.core.service import (
    ChatService,
    get_chat_service,
)

from .models import (
    ChatRequest,
    format_error_sse,
    format_sse,
)

STREAMING_RESPONSE_MEDIA_TYPE = "text/plain"
STREAMING_RESPONSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Access-Control-Allow-Headers": "Cache-Control",
}

router = APIRouter(prefix="/api/v1", tags=["chat"])


async def stream_chat_response(
    request: ChatRequest, service: ChatService
) -> AsyncGenerator[str, None]:
    """Iterate domain events from the service and serialize to SSE."""
    try:
        async for event in service.stream_response(request.query):
            yield format_sse(event)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        yield format_error_sse(e)


@router.post("/chat")
async def chat(
    chat_request: ChatRequest,
    chat_service: Annotated[ChatService, Depends(get_chat_service)],
) -> StreamingResponse:
    """Process a chat request and return a streaming response.

    The response is a stream of Server-Sent Events, where each event
    is a JSON object with a ``type`` discriminator:

    - **thinking**: agent internal reasoning
    - **content**: user-facing streamed text tokens
    - **tool_call**: tool invocation lifecycle (started / completed / error)
    - **error**: stream-level error

    Stream ends when the connection is closed.
    """
    return StreamingResponse(
        stream_chat_response(chat_request, chat_service),
        media_type=STREAMING_RESPONSE_MEDIA_TYPE,
        headers=STREAMING_RESPONSE_HEADERS,
    )
