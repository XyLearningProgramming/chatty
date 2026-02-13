from typing import Annotated

from fastapi import Depends
from langchain_core.language_models import BaseLanguageModel

from chatty.configs.config import AppConfig, get_app_config
from chatty.core.embedding.client import EmbeddingClient
from chatty.core.embedding.repository import EmbeddingRepository
from chatty.core.llm import get_gated_llm
from chatty.core.service.one_step import OneStepChatService
from chatty.core.service.rag import RagChatService
from chatty.infra import singleton
from chatty.infra.concurrency import get_model_semaphore
from chatty.infra.db.engine import get_async_session_factory

from .models import ChatService
from .tools.registry import ToolRegistry, get_tool_registry

_known_agents: dict[str, type] = {
    OneStepChatService.chat_service_name: OneStepChatService,
    RagChatService.chat_service_name: RagChatService,
}


@singleton
def get_embedding_client() -> EmbeddingClient:
    """Create the singleton ``EmbeddingClient``.

    Wires together the embedding config, DB repository, and the same
    ``ModelSemaphore`` used by the LLM gate.
    """
    config = get_app_config()
    repo = EmbeddingRepository(get_async_session_factory())
    semaphore = get_model_semaphore()
    return EmbeddingClient(
        config=config.embedding,
        repository=repo,
        semaphore=semaphore,
    )


@singleton
def get_chat_service(
    llm: Annotated[BaseLanguageModel, Depends(get_gated_llm)],
    tools_registry: Annotated[ToolRegistry, Depends(get_tool_registry)],
    config: Annotated[AppConfig, Depends(lambda: get_app_config())],
) -> ChatService:
    """Factory function to create a configured chat service.

    The returned service is a singleton.  Tool definitions (for
    OneStep) and section embeddings (for RAG) are loaded fresh on
    every request so that ConfigMap updates are picked up.

    The *llm* is a ``GatedChatModel`` -- every model invocation
    acquires a concurrency slot automatically, so the service does
    not need to manage concurrency itself.
    """
    name = config.chat.agent_name
    if name not in _known_agents:
        raise NotImplementedError(f"Agent {name} is not implemented.")

    agent_cls = _known_agents[name]

    if agent_cls is OneStepChatService:
        return agent_cls(llm, tools_registry, config)

    if agent_cls is RagChatService:
        embedding_client = get_embedding_client()
        return agent_cls(llm, config, embedding_client)

    # Generic fallback (should not happen with _known_agents)
    raise NotImplementedError(f"Agent {name} is not implemented.")
