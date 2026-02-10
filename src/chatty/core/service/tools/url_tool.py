"""URL tool: fetches content from a pre-configured URL.

The URL is a hidden config arg — the model only sees the tool's
name and description.  It calls the tool with no arguments and gets
back the fetched content (optionally processed).
"""

from typing import Self

import httpx
from langchain_core.tools import BaseTool

from chatty.configs.tools import ToolConfig


class FixedURLTool(BaseTool):
    """Tool that fetches content from a fixed, pre-configured URL."""

    # Hidden config — not exposed to the model
    url: str = ""
    timeout: int = 30
    max_content_length: int = 1000

    def _run(self, *args, **kwargs) -> str:
        """Fetch content synchronously (fallback)."""
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(self.url)
            response.raise_for_status()
            return self._truncate(response.text)

    async def _arun(self, *args, **kwargs) -> str:
        """Fetch content asynchronously."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(self.url)
            response.raise_for_status()
            return self._truncate(response.text)

    def _truncate(self, content: str) -> str:
        if len(content) > self.max_content_length:
            return content[: self.max_content_length] + "..."
        return content

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
