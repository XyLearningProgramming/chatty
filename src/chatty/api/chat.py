"""Chat API endpoint implementation."""

import logging
from collections.abc import AsyncGenerator, Awaitable, Callable

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from chatty.core.service.models import (
    ChatContext,
    DequeuedEvent,
    QueuedEvent,
    StreamEvent,
)
from chatty.infra.concurrency import InboxFull
from chatty.infra.id_utils import generate_id
from chatty.infra.telemetry import get_current_trace_id

from .deps import (
    APIConfigDep,
    ChatConfigDep,
    ChatMessageHistoryFactoryDep,
    ChatServiceDep,
    InboxDep,
)
from .models import ChatRequest
from .streaming import sse_stream

logger = logging.getLogger(__name__)

STREAMING_RESPONSE_MEDIA_TYPE = "text/plain"
STREAMING_RESPONSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Access-Control-Allow-Headers": (
        "Cache-Control, X-Chatty-Trace, X-Chatty-Conversation"
    ),
    "Access-Control-Expose-Headers": ("X-Chatty-Trace, X-Chatty-Conversation"),
}

router = APIRouter(tags=["chat"])


# ---------------------------------------------------------------------------
# Business-logic generator — pure domain events, no SSE formatting
# ---------------------------------------------------------------------------


async def _chat_events(
    ctx: ChatContext,
    service: ChatServiceDep,
    position: int,
    is_disconnected: Callable[[], Awaitable[bool]],
) -> AsyncGenerator[StreamEvent, None]:
    """Yield domain events for a single chat request.

    1. ``QueuedEvent`` — client knows it is waiting.
    2. ``DequeuedEvent`` — request admitted, agent starting.
    3. Delegate to ``service.stream_response()``, checking for
       client disconnect between each event.

    Concurrency gating on the LLM is handled transparently by
    ``GatedChatModel`` — there is no semaphore logic here.
    """
    yield QueuedEvent(position=position)
    yield DequeuedEvent()

    async for event in service.stream_response(ctx):
        if await is_disconnected():
            logger.debug("Client disconnected during streaming.")
            return
        yield event


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/chat")
async def chat(
    request: Request,
    chat_request: ChatRequest,
    chat_service: ChatServiceDep,
    api_config: APIConfigDep,
    chat_config: ChatConfigDep,
    inbox: InboxDep,
    chat_message_history_factory: ChatMessageHistoryFactoryDep,
) -> StreamingResponse | JSONResponse:
    """Process a chat request and return a streaming response.

    Supports two modes (ChatGPT-style):

    - **New conversation**: omit ``conversation_id`` — a fresh ID is
      generated and returned in the ``X-Chatty-Conversation`` header.
    - **Continue conversation**: pass an existing ``conversation_id`` —
      the server loads recent history from the DB (up to
      ``max_conversation_length``) before invoking the agent.

    The response is a stream of Server-Sent Events, where each event
    is a JSON object with a ``type`` discriminator:

    - **queued** / **dequeued** — concurrency lifecycle
    - **thinking** / **content** / **tool_call** — agent output
    - **error** — stream-level error

    Infrastructure IDs are returned via response headers:

    - ``X-Chatty-Trace`` — trace ID for debugging
    - ``X-Chatty-Conversation`` — conversation ID for follow-up turns

    Returns **429** when the inbox is full.
    """
    # Admission control — reject early with a proper HTTP status code.
    try:
        position = await inbox.enter()
    except InboxFull:
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests in flight. Try again later."},
        )

    # --- Resolve conversation ID ---
    conversation_id = chat_request.conversation_id or generate_id("conv")
    trace_id = get_current_trace_id() or generate_id("trace")

    # --- Load history for continuing conversations ---
    history: list = []
    if chat_request.conversation_id:
        history_obj = chat_message_history_factory(
            conversation_id,
            trace_id=None,
            max_messages=chat_config.max_conversation_length,
        )
        history = await history_obj.aget_messages()

    ctx = ChatContext(
        query=chat_request.query,
        conversation_id=conversation_id,
        trace_id=trace_id,
        history=history,
    )

    logger.info(f"Started chat streaming with context: {ctx}")

    return StreamingResponse(
        sse_stream(
            _chat_events(ctx, chat_service, position, request.is_disconnected),
            request_timeout=api_config.request_timeout,
            on_finish=inbox.leave,
        ),
        media_type=STREAMING_RESPONSE_MEDIA_TYPE,
        headers={
            **STREAMING_RESPONSE_HEADERS,
            "X-Chatty-Trace": trace_id,
            "X-Chatty-Conversation": conversation_id,
        },
    )
