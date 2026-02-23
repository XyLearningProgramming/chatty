"""Search tool: the model sees a single tool whose ``source`` argument
selects from a set of knowledge sources.

The model picks a *source* key (e.g. ``"resume"``, ``"current_homepage"``)
and the tool internally resolves it to a URL, fetches content,
post-processes, and returns plain text.  URLs and processing details are
never exposed to the model.
"""

from __future__ import annotations

import logging
from typing import Any, Self

from pydantic import BaseModel, PrivateAttr

from chatty.configs.persona import KnowledgeSource, ToolDeclaration
from chatty.configs.system import PromptConfig
from chatty.infra.http_utils import HttpClient
from chatty.infra.telemetry import (ATTR_TOOL_ERROR, ATTR_TOOL_SOURCE,
                                    SPAN_TOOL_URL_DISPATCHER, tracer)

from .model import (FunctionDefinition, ParametersDefinition,
                    PropertyDefinition, ToolDefinition)

logger = logging.getLogger(__name__)

PARAM_SOURCE = "source"
_JSON_SCHEMA_TYPE_STRING = "string"
_ERR_INVALID_SOURCE = "invalid_source"


class SearchTool(BaseModel):
    """Single tool exposed to the LLM.

    The model sees one tool with a ``source`` argument whose allowed
    values are built dynamically from YAML config.  Internally each
    source maps to a ``KnowledgeSource`` that handles fetching and
    processing via ``get_content()``.
    """

    name: str
    description: str
    sources: dict[str, KnowledgeSource] = {}
    source_descriptions: dict[str, str] = {}
    action_processors: list[Any] = []
    _prompt: PromptConfig = PrivateAttr()

    def __init__(self, *, prompt: PromptConfig, **data: Any) -> None:
        super().__init__(**data)
        self._prompt = prompt

    # ---- OpenAI tool definition ------------------------------------------

    def to_tool_definition(self) -> ToolDefinition:
        """Return a typed OpenAI-compatible tool definition."""
        source_desc = self._prompt.render_tool_source_field(
            self.source_descriptions,
        )
        return ToolDefinition(
            function=FunctionDefinition(
                name=self.name,
                description=self.description,
                parameters=ParametersDefinition(
                    properties={
                        PARAM_SOURCE: PropertyDefinition(
                            type=_JSON_SCHEMA_TYPE_STRING,
                            description=source_desc,
                            enum=list(self.sources.keys()),
                        ),
                    },
                    required=[PARAM_SOURCE],
                ),
            ),
        )

    # ---- Execution -------------------------------------------------------

    async def execute(self, source: str, *args: Any, **kwargs: Any) -> str:
        """Resolve *source* key to a knowledge source, fetch, and return text."""
        if args or kwargs:
            logger.warning("Ignoring unexpected tool arguments: %s", args or kwargs)
        with tracer.start_as_current_span(SPAN_TOOL_URL_DISPATCHER) as span:
            span.set_attribute(ATTR_TOOL_SOURCE, source)
            logger.debug("Tool dispatch: source=%s", source)
            src = self.sources.get(source)
            if src is None:
                span.set_attribute(ATTR_TOOL_ERROR, _ERR_INVALID_SOURCE)
                valid = ", ".join(f'"{k}"' for k in self.sources)
                return self._prompt.render_tool_error(source=source, valid=valid)
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
        """Build one search tool from a ``ToolDeclaration``."""
        tool_sources = {sid: sources[sid] for sid in declaration.sources}

        descriptions: dict[str, str] = {}
        for sid, src in tool_sources.items():
            desc = (src.description or "").strip().split("\n")[0].strip()
            if not desc:
                desc = prompt.render_tool_source_hint(source_id=sid)
            descriptions[sid] = desc

        return cls(
            name=declaration.name,
            description=declaration.description or prompt.tool_description,
            sources=tool_sources,
            source_descriptions=descriptions,
            action_processors=declaration.get_processors(),
            prompt=prompt,
        )
