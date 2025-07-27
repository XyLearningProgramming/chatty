"""Chat API endpoint implementation."""

from typing import Annotated, AsyncGenerator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from chatty.configs import AppConfig, get_app_config

from .models import ChatRequest, EndOfStreamEvent, ErrorEvent, TokenEvent

router = APIRouter(prefix="/api/v1", tags=["chat"])


async def stream_chat_response(
    request: ChatRequest,
    app_config: AppConfig,
) -> AsyncGenerator[str, None]:
    """Stream chat response as Server-Sent Events."""
    try:
        # TODO: Implement the actual chat pipeline here
        # For now, return a simple echo response

        # Simulate token streaming
        response_text = f"You asked: {request.query}"
        for token in response_text.split():
            event = TokenEvent(content=token + " ")
            yield f"data: {event.model_dump_json()}\n\n"

        # End of stream marker
        end_event = EndOfStreamEvent()
        yield f"data: {end_event.model_dump_json()}\n\n"

    except Exception as e:
        # Send error event
        error_event = ErrorEvent(
            message=f"An error occurred during processing: {str(e)}",
            code="PROCESSING_ERROR",
        )
        yield f"data: {error_event.model_dump_json()}\n\n"


@router.post("/chat")
async def chat(
    request: ChatRequest,
    app_config: Annotated[AppConfig, Depends(get_app_config)],
) -> StreamingResponse:
    """
    Process a chat request and return a streaming response.

    The response is a stream of Server-Sent Events, where each event
    is a JSON object representing different types of data:
    - token: Streaming text tokens
    - structured_data: JSON objects for frontend rendering
    - end_of_stream: Marks the end of the response
    - error: Error information
    """
    return StreamingResponse(
        stream_chat_response(request, app_config),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Headers": "Cache-Control",
        },
    )
