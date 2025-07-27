"""Chat API endpoint implementation."""

from typing import Annotated, AsyncGenerator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from chatty.configs import AppConfig, get_app_config
from chatty.core.chat_service import ChatService
from chatty.core.tools import ToolsRegistry

from .models import ChatRequest, EndOfStreamEvent, ErrorEvent, TokenEvent

router = APIRouter(prefix="/api/v1", tags=["chat"])


def get_chat_service(app_config: Annotated[AppConfig, Depends(get_app_config)]) -> ChatService:
    """Get configured chat service with tools registry."""
    tools_registry = ToolsRegistry(app_config.persona.tools)
    return ChatService(app_config.chat, tools_registry)


async def stream_chat_response(
    request: ChatRequest,
    chat_service: ChatService,
) -> AsyncGenerator[str, None]:
    """Stream chat response as Server-Sent Events."""
    try:
        # Stream response from the chat service
        async for event in chat_service.stream_response(request.query):
            if event["type"] == "token":
                token_event = TokenEvent(content=event["data"]["token"])
                yield f"data: {token_event.model_dump_json()}\n\n"
            elif event["type"] == "end_of_stream":
                end_event = EndOfStreamEvent()
                yield f"data: {end_event.model_dump_json()}\n\n"
            elif event["type"] == "error":
                error_event = ErrorEvent(
                    message=event["data"]["error"],
                    code="PROCESSING_ERROR",
                )
                yield f"data: {error_event.model_dump_json()}\n\n"

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
    """
    return StreamingResponse(
        stream_chat_response(request, chat_service),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Headers": "Cache-Control",
        },
    )
