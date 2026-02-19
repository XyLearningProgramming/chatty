"""Centralized FastAPI dependency type aliases.

Import these ``*Dep`` aliases in route modules instead of manually
writing ``Annotated[T, Depends(get_xxx)]`` everywhere.  Each alias
corresponds to a single ``get_*`` factory and can be overridden in
tests via ``app.dependency_overrides[get_xxx] = ...``.
"""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from chatty.configs.config import (
    AppConfig,
    get_api_config,
    get_app_config,
    get_chat_config,
)
from chatty.configs.system import APIConfig, ChatConfig
from chatty.core.service.callback import PgCallbackFactory, get_pg_callback_factory
from chatty.core.service.deps import get_chat_service
from chatty.core.service.models import ChatService
from chatty.infra.concurrency.guards import RequestGuard, get_request_guard
from chatty.infra.concurrency.inbox import Inbox, get_inbox
from chatty.infra.concurrency.real_ip import get_real_ip
from chatty.infra.db import (
    ChatMessageHistoryFactory,
    get_chat_message_history_factory,
)
from chatty.infra.db.engine import get_async_session, get_session_factory

AppConfigDep = Annotated[AppConfig, Depends(get_app_config)]
APIConfigDep = Annotated[APIConfig, Depends(get_api_config)]
ChatConfigDep = Annotated[ChatConfig, Depends(get_chat_config)]
ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
InboxDep = Annotated[Inbox, Depends(get_inbox)]
AsyncSessionDep = Annotated[AsyncSession, Depends(get_async_session)]
SessionFactoryDep = Annotated[
    async_sessionmaker[AsyncSession], Depends(get_session_factory)
]
PgCallbackFactoryDep = Annotated[
    PgCallbackFactory, Depends(get_pg_callback_factory)
]
ChatMessageHistoryFactoryDep = Annotated[
    ChatMessageHistoryFactory, Depends(get_chat_message_history_factory)
]
RealIPDep = Annotated[str, Depends(get_real_ip)]
RequestGuardDep = Annotated[RequestGuard, Depends(get_request_guard)]
