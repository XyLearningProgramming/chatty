"""URL dispatcher tool: the model sees a single ``lookup`` tool.

The model picks a *source* (e.g. ``"resume"``, ``"current_homepage"``)
and the dispatcher fetches, post-processes, and returns plain text.
URLs and processing details are never exposed to the model.
"""

from __future__ import annotations

import logging
from typing import Any, Self, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr, create_model

from chatty.configs.persona import KnowledgeSource, ToolDeclaration
from chatty.configs.system import PromptConfig
from chatty.infra.http_utils import HttpClient
from chatty.infra.telemetry import (
    ATTR_TOOL_ERROR,
    ATTR_TOOL_SOURCE,
    SPAN_TOOL_URL_DISPATCHER,
    tracer,
)

logger = logging.getLogger(__name__)

_SCHEMA_NAME = "LookupInput"


class URLDispatcherTool(BaseTool):
    """Single ``lookup`` tool exposed to the model.

    The model sees one tool with a ``source`` argument whose allowed
    values are built dynamically from YAML config.  Internally each
    source maps to a ``KnowledgeSource`` that handles fetching and
    processing via ``get_content()``.
    """

    sources: dict[str, KnowledgeSource] = Field(default_factory=dict)
    action_processors: list[Any] = Field(default_factory=list)
    _prompt: PromptConfig = PrivateAttr()

    def __init__(self, *, prompt: PromptConfig, **data: Any) -> None:
        super().__init__(**data)
        self._prompt = prompt

    # ---- LangChain interface ---------------------------------------------

    def _run(self, source: str) -> str:
        raise NotImplementedError("Use async — call via _arun")

    async def _arun(self, source: str) -> str:
        with tracer.start_as_current_span(SPAN_TOOL_URL_DISPATCHER) as span:
            span.set_attribute(ATTR_TOOL_SOURCE, source)
            logger.debug("Tool dispatch: source=%s", source)
            src = self.sources.get(source)
            if src is None:
                span.set_attribute(ATTR_TOOL_ERROR, "invalid_source")
                valid = ", ".join(f'"{k}"' for k in self.sources)
                return self._prompt.render_tool_error(
                    source=source, valid=valid
                )
            return await src.get_content(
                HttpClient.get,
                extra_processors=self.action_processors or None,
            )

    # ---- Factory ---------------------------------------------------------

    @classmethod
    def from_declaration(
        cls,
        declaration: ToolDeclaration,
        sources: dict[str, KnowledgeSource],
        prompt: PromptConfig,
    ) -> Self:
        """Build one dispatcher from a ``ToolDeclaration``."""
        tool_sources = {
            sid: sources[sid] for sid in declaration.sources
        }

        descriptions: dict[str, str] = {}
        for sid, src in tool_sources.items():
            desc = (src.description or "").strip().split("\n")[0].strip()
            if not desc:
                desc = prompt.render_tool_source_hint(source_id=sid)
            descriptions[sid] = desc

        schema = _build_args_schema(
            prompt.render_tool_source_field(descriptions),
            list(descriptions),
        )

        return cls(
            name=declaration.name,
            description=declaration.description or prompt.tool_description,
            sources=tool_sources,
            action_processors=declaration.get_processors(),
            args_schema=schema,
            prompt=prompt,
        )


def _build_args_schema(
    source_description: str,
    source_keys: list[str],
) -> Type[BaseModel]:
    """Build a dynamic Pydantic model for the ``source`` argument."""
    return create_model(
        _SCHEMA_NAME,
        source=(
            str,
            Field(
                description=source_description,
                json_schema_extra={"enum": source_keys},
            ),
        ),
    )
