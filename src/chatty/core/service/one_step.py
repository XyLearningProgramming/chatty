"""Chat service using LangGraph agent (via langchain.agents)."""

from langchain.agents import create_agent
from langchain_core.language_models import BaseLanguageModel
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from chatty.configs.config import AppConfig

from .base import GraphChatService
from .models import ChatContext
from .tools.registry import ToolRegistry


class OneStepChatService(GraphChatService):
    """Chat service powered by LangGraph ReAct agent.

    The agent graph is rebuilt on every request so that hot-reloaded tool
    definitions from the ConfigMap are picked up immediately.
    """

    chat_service_name = "one_step"

    def __init__(
        self,
        llm: BaseLanguageModel,
        tools_registry: ToolRegistry,
        config: AppConfig,
        session_factory: async_sessionmaker[AsyncSession],
    ):
        super().__init__(llm, config, session_factory)
        self._tools_registry = tools_registry

    async def _create_graph(self, ctx: ChatContext):
        """Create the LangGraph agent for this request.

        The graph is created fresh each request so that the latest tool
        definitions from YAML files (including ConfigMap) are always used.
        """
        return create_agent(
            model=self._llm,
            tools=self._tools_registry.get_tools(),
            system_prompt=self._system_prompt,
        )
