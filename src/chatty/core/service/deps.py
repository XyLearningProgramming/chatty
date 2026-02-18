"""FastAPI dependency factories for chat services.

``get_embedding_client`` reads from ``app.state`` (created in lifespan).
``get_chat_service`` is a per-request ``Depends`` factory with an
explicit parameter chain — no hidden calls.
"""

from typing import Annotated

from fastapi import Depends, Request
from langchain_core.language_models import BaseLanguageModel
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from chatty.configs.config import AppConfig, get_app_config
from chatty.core.embedding.client import EmbeddingClient
from chatty.core.llm import get_gated_llm
from chatty.core.service.one_step import OneStepChatService
from chatty.core.service.rag import RagChatService
from chatty.infra.db.engine import get_session_factory

from .base import BaseChatService
from .models import ChatService
from .tools.registry import ToolRegistry, get_tool_registry

_known_agents: dict[str, type[BaseChatService]] = {
    OneStepChatService.chat_service_name: OneStepChatService,
    RagChatService.chat_service_name: RagChatService,
}


# ---------------------------------------------------------------------------
# EmbeddingClient — built by build_cron, accessed via app.state
# ---------------------------------------------------------------------------


def get_embedding_client(request: Request) -> EmbeddingClient:
    """FastAPI dependency — reads from ``app.state``."""
    return request.app.state.embedding_client


# ---------------------------------------------------------------------------
# ChatService — per-request, fully explicit Depends chain
# ---------------------------------------------------------------------------


def get_chat_service(
    llm: Annotated[BaseLanguageModel, Depends(get_gated_llm)],
    tools_registry: Annotated[
        ToolRegistry, Depends(get_tool_registry)
    ],
    config: Annotated[AppConfig, Depends(get_app_config)],
    sf: Annotated[
        async_sessionmaker[AsyncSession],
        Depends(get_session_factory),
    ],
    request: Request,
) -> ChatService:
    """Create a configured chat service per request.

    All dependencies are injected explicitly via ``Depends()`` —
    no hidden calls.
    """
    name = config.chat.agent_name
    if name not in _known_agents:
        raise NotImplementedError(
            f"Agent {name} is not implemented."
        )

    agent_cls = _known_agents[name]

    if agent_cls is OneStepChatService:
        return agent_cls(llm, tools_registry, config, sf)

    if agent_cls is RagChatService:
        embedding_client = get_embedding_client(request)
        return agent_cls(llm, config, embedding_client, sf)

    raise NotImplementedError(
        f"Agent {name} is not implemented."
    )
