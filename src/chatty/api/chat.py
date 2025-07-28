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
    convert_service_event_to_api_event,
    convert_service_exc_to_api_error_event,
)

STREAMING_RESPONSE_MEDIA_TYPE = "text/plain"
STREAMING_RESPONSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Access-Control-Allow-Headers": "Cache-Control",
}

router = APIRouter(prefix="/api/v1", tags=["chat"])


async def stream_chat_response(request, service) -> AsyncGenerator[str, None]:
    try:
        async for service_event in service.stream_response(request.query):
            yield convert_service_event_to_api_event(service_event)
    except asyncio.CancelledError:
        # The `stream_response` generator itself will see this Cancellation
        # from yield at its next await, so it can clean up there.
        raise  # Raise again without wrapping into an error event.
    except Exception as e:
        yield convert_service_exc_to_api_error_event(e)


@router.post("/chat")
async def chat(
    chat_request: ChatRequest,
    chat_service: Annotated[ChatService, Depends(get_chat_service)],
) -> StreamingResponse:
    """
    Process a chat request and return a streaming response.

    The response is a stream of Server-Sent Events, where each event
    is a JSON object representing different types of data:
    - token: Streaming text tokens
    - structured_data: JSON objects for frontend rendering
    - end_of_stream: Marks the end of the response
    - error: Error information

    Handles client disconnection by cancelling ongoing LLM processing.
    """
    return StreamingResponse(
        stream_chat_response(chat_request, chat_service),
        media_type=STREAMING_RESPONSE_MEDIA_TYPE,
        headers=STREAMING_RESPONSE_HEADERS,
    )
