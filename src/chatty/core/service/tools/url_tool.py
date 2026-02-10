"""URL tool: fetches content from a pre-configured URL.

The URL is a hidden config arg — the model only sees the tool's
name and description.  It calls the tool with no arguments and gets
back the fetched content (optionally processed).

Supports both HTML and PDF responses.  When the server returns
``application/pdf``, the binary payload is converted to plain text
via *pymupdf* before truncation and downstream processing.
"""

from typing import Self

import httpx
from langchain_core.tools import BaseTool

from chatty.configs.tools import ToolConfig

_PDF_CONTENT_TYPES = {"application/pdf", "application/x-pdf"}


def _extract_text_from_pdf(data: bytes) -> str:
    """Extract readable text from raw PDF bytes using pymupdf."""
    import pymupdf  # lazy import – only needed for PDF responses

    text_parts: list[str] = []
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)


class FixedURLTool(BaseTool):
    """Tool that fetches content from a fixed, pre-configured URL."""

    # Hidden config — not exposed to the model
    url: str = ""
    timeout: int = 30
    max_content_length: int = 1000

    # ---- internal helpers ------------------------------------------------

    @staticmethod
    def _is_pdf(response: httpx.Response) -> bool:
        """Return *True* if the response carries PDF content."""
        content_type = response.headers.get("content-type", "")
        return any(ct in content_type for ct in _PDF_CONTENT_TYPES)

    def _read_response(self, response: httpx.Response) -> str:
        """Decode response body — handles both HTML/text and PDF."""
        if self._is_pdf(response):
            return _extract_text_from_pdf(response.content)
        return response.text

    def _truncate(self, content: str) -> str:
        if len(content) > self.max_content_length:
            return content[: self.max_content_length] + "..."
        return content

    # ---- LangChain interface ---------------------------------------------

    def _run(self, *args, **kwargs) -> str:
        """Fetch content synchronously (fallback)."""
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(self.url)
            response.raise_for_status()
            return self._truncate(self._read_response(response))

    async def _arun(self, *args, **kwargs) -> str:
        """Fetch content asynchronously."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(self.url)
            response.raise_for_status()
            return self._truncate(self._read_response(response))

    @classmethod
    def from_config(cls, config: ToolConfig) -> Self:
        """Build URL tool from YAML configuration."""
        return cls(
            name=config.name,
            description=config.description or "",
            url=config.args.get("url", ""),
            timeout=config.args.get("timeout", 30),
            max_content_length=config.args.get(
                "max_content_length", 1000
            ),
        )


# Set after class definition to avoid Pydantic interference
FixedURLTool.tool_type = "url"
