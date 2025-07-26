"""Chat API endpoint implementation."""

import json
from typing import Annotated, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from ..config import AppConfig, AuthorConfig, load_author_config
from .models import ChatRequest, StreamEvent, TokenEvent, EndOfStreamEvent, ErrorEvent

router = APIRouter(prefix="/api/v1", tags=["chat"])


async def get_app_config() -> AppConfig:
    """Dependency to provide application configuration."""
    return AppConfig()


async def get_author_config(
    app_config: Annotated[AppConfig, Depends(get_app_config)],
) -> AuthorConfig:
    """Dependency to provide author configuration."""
    try:
        return load_author_config(app_config.author_config_path)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500, detail=f"Author configuration not found: {e}"
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to load author configuration: {e}"
        ) from e


async def stream_chat_response(
    request: ChatRequest,
    app_config: AppConfig,
    author_config: AuthorConfig,
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
    author_config: Annotated[AuthorConfig, Depends(get_author_config)],
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
        stream_chat_response(request, app_config, author_config),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control",
        },
    )
