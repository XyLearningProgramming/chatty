"""URL dispatcher tool: the model sees a single ``lookup`` tool.

The model picks a *source* (e.g. ``"resume"``, ``"current_homepage"``)
and the dispatcher resolves it to the real URL, fetches, post-processes,
and returns plain text.  URLs and processing details are never exposed.

Supports both HTML and PDF responses.  When the server returns
``application/pdf``, the binary payload is converted to plain text
via *pymupdf* before downstream processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Self, Type

import httpx
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, create_model

from chatty.configs.persona import KnowledgeSource, ToolDeclaration

_PDF_CONTENT_TYPES = {"application/pdf", "application/x-pdf"}


# ---------------------------------------------------------------------------
# Route args
# ---------------------------------------------------------------------------


class URLToolArgs(BaseModel):
    """Typed args for a single URL route."""

    url: str = Field(default="", description="Target URL to fetch.")
    timeout: timedelta = Field(
        default=timedelta(seconds=30),
        description="HTTP request timeout.",
    )


# ---------------------------------------------------------------------------
# Route: one named destination inside the dispatcher
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Route:
    """One named URL endpoint with its args and post-processors."""

    args: URLToolArgs
    processors: list[Any] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_text_from_pdf(data: bytes) -> str:
    """Extract readable text from raw PDF bytes using pymupdf."""
    import pymupdf  # lazy import -- only needed for PDF responses

    text_parts: list[str] = []
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)


def _build_args_schema(
    resource_descriptions: dict[str, str],
) -> Type[BaseModel]:
    """Build a dynamic Pydantic model whose ``source`` field is
    constrained to the known resource keys.
    """
    options = "\n".join(
        f'  - "{key}": {desc}'
        for key, desc in resource_descriptions.items()
    )
    return create_model(
        "LookupInput",
        source=(
            str,
            Field(
                description=(
                    "The information source to query. "
                    "Pick the one key that best answers the "
                    "user's question.\n"
                    f"{options}"
                ),
                json_schema_extra={
                    "enum": list(resource_descriptions)
                },
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Dispatcher tool
# ---------------------------------------------------------------------------


class URLDispatcherTool(BaseTool):
    """Single ``lookup`` tool exposed to the model.

    The model sees one tool with a ``source`` argument whose allowed
    values are built dynamically from YAML config.  Internally each
    source maps to a hidden URL and optional post-processors.
    """

    routes: dict[str, Route] = Field(default_factory=dict)

    # ---- internal helpers ------------------------------------------------

    @staticmethod
    def _is_pdf(response: httpx.Response) -> bool:
        ct = response.headers.get("content-type", "")
        return any(pdf in ct for pdf in _PDF_CONTENT_TYPES)

    def _read_response(self, response: httpx.Response) -> str:
        if self._is_pdf(response):
            return _extract_text_from_pdf(response.content)
        return response.text

    @staticmethod
    def _apply_processors(content: str, route: Route) -> str:
        for processor in route.processors:
            content = processor.process(content)
        return content

    def _fetch_route(self, route: Route) -> str:
        """Fetch + process synchronously."""
        with httpx.Client(
            timeout=route.args.timeout.total_seconds()
        ) as client:
            response = client.get(route.args.url)
            response.raise_for_status()
            content = self._read_response(response)
            return self._apply_processors(content, route)

    async def _async_fetch_route(self, route: Route) -> str:
        """Fetch + process asynchronously."""
        async with httpx.AsyncClient(
            timeout=route.args.timeout.total_seconds()
        ) as client:
            response = await client.get(route.args.url)
            response.raise_for_status()
            content = self._read_response(response)
            return self._apply_processors(content, route)

    # ---- LangChain interface ---------------------------------------------

    def _run(self, source: str) -> str:
        route = self.routes.get(source)
        if route is None:
            valid = ", ".join(f'"{k}"' for k in self.routes)
            return (
                f"Unknown source '{source}'. "
                f"Valid options: {valid}"
            )
        return self._fetch_route(route)

    async def _arun(self, source: str) -> str:
        route = self.routes.get(source)
        if route is None:
            valid = ", ".join(f'"{k}"' for k in self.routes)
            return (
                f"Unknown source '{source}'. "
                f"Valid options: {valid}"
            )
        return await self._async_fetch_route(route)

    # ---- Factory ---------------------------------------------------------

    @classmethod
    def from_declaration(
        cls,
        declaration: ToolDeclaration,
        sources: dict[str, KnowledgeSource],
        *,
        processors: dict[str, list[Any]] | None = None,
    ) -> Self:
        """Build one dispatcher from a tool declaration.

        Parameters
        ----------
        declaration:
            ``ToolDeclaration`` with source ids and tool metadata.
        sources:
            Full ``persona.sources`` dict.
        processors:
            Mapping of source id to merged (source + action level)
            resolved ``Processor`` instances.
        """
        processors = processors or {}

        routes: dict[str, Route] = {}
        resource_descriptions: dict[str, str] = {}

        for source_id in declaration.sources:
            source = sources[source_id]
            args = URLToolArgs(
                url=source.content_url,
                timeout=source.timeout,
            )
            route_processors = processors.get(source_id, [])
            routes[source_id] = Route(
                args=args, processors=route_processors
            )

            desc = (
                (source.description or "").strip().split("\n")[0].strip()
            )
            if not desc:
                desc = f"Fetch information about '{source_id}'."
            resource_descriptions[source_id] = desc

        tool_desc = declaration.description or (
            "Look up external information that you don't "
            "already know. Call this when the user asks about "
            "something that requires live data."
        )

        return cls(
            name=declaration.name,
            description=tool_desc,
            routes=routes,
            args_schema=_build_args_schema(resource_descriptions),
        )


URLDispatcherTool.tool_type = "url_dispatcher"
