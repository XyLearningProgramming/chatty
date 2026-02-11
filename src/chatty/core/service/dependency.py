from typing import Annotated

from fastapi import Depends
from langchain_core.language_models import BaseLanguageModel

from chatty.configs.config import AppConfig, get_app_config
from chatty.core.llm import get_llm
from chatty.core.service.one_step import OneStepChatService
from chatty.infra import singleton

from .models import ChatService
from .tools.registry import ToolRegistry, get_tool_registry

_known_agents = {srv.chat_service_name: srv for srv in [OneStepChatService]}


@singleton
def get_chat_service(
    llm: Annotated[BaseLanguageModel, Depends(get_llm)],
    tools_registry: Annotated[ToolRegistry, Depends(get_tool_registry)],
    config: Annotated[AppConfig, Depends(lambda: get_app_config())],
) -> ChatService:
    """Factory function to create a configured chat service.

    The returned service is a singleton, but tool definitions are loaded
    fresh on every request via ``ToolRegistry.get_tools()``.
    """
    name = config.chat.agent_name
    if name not in _known_agents:
        raise NotImplementedError(f"Agent {name} is not implemented.")

    return _known_agents[name](llm, tools_registry, config)
