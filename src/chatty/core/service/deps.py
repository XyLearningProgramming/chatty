"""FastAPI dependency factories for chat services.

``get_chat_service`` is a per-request ``Depends`` factory
with an explicit parameter chain.
"""

from typing import Annotated

from fastapi import Depends
from langchain_core.language_models import BaseLanguageModel

from chatty.configs.config import AppConfig, get_app_config
from chatty.core.embedding.cron import get_embedder, get_embedding_repository
from chatty.core.embedding.gated import GatedEmbedModel
from chatty.core.llm import get_gated_llm
from chatty.core.service.one_step import OneStepChatService
from chatty.core.service.rag import RagChatService
from chatty.infra.db import (
    ChatMessageHistoryFactory,
    get_chat_message_history_factory,
)
from chatty.infra.db.cache import CacheRepository
from chatty.infra.db.embedding import EmbeddingRepository
from chatty.infra.db.deps import get_cache_repository

from .callback import PgCallbackFactory, get_pg_callback_factory
from .models import ChatService
from .tools.registry import ToolRegistry, get_tool_registry

_known_agents: dict[str, type[ChatService]] = {
    OneStepChatService.chat_service_name: OneStepChatService,
    RagChatService.chat_service_name: RagChatService,
}


# ---------------------------------------------------------------------------
# ChatService — per-request, fully explicit Depends chain
# ---------------------------------------------------------------------------


def get_chat_service(
    llm: Annotated[BaseLanguageModel, Depends(get_gated_llm)],
    tools_registry: Annotated[ToolRegistry, Depends(get_tool_registry)],
    config: Annotated[AppConfig, Depends(get_app_config)],
    pg_callback_factory: Annotated[PgCallbackFactory, Depends(get_pg_callback_factory)],
    embedder: Annotated[GatedEmbedModel, Depends(get_embedder)],
    embedding_repository: Annotated[EmbeddingRepository, Depends(get_embedding_repository)],
    cache_repository: Annotated[CacheRepository, Depends(get_cache_repository)],
    history_factory: Annotated[
        ChatMessageHistoryFactory, Depends(get_chat_message_history_factory),
    ],
) -> ChatService:
    """Create a configured chat service per request.

    All dependencies are injected explicitly via ``Depends()`` —
    no hidden calls.
    """
    name = config.chat.agent_name
    if name not in _known_agents:
        raise NotImplementedError(f"Agent {name} is not implemented.")

    agent_cls = _known_agents[name]

    if agent_cls is OneStepChatService:
        return agent_cls(llm, tools_registry, config, pg_callback_factory)

    if agent_cls is RagChatService:
        return agent_cls(
            llm, config, embedder, embedding_repository,
            history_factory, cache_repository,
        )

    raise NotImplementedError(f"Agent {name} is not implemented.")
