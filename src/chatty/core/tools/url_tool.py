"""Simple URL tool implementation."""

from typing import Self

import httpx
from langchain.tools import BaseTool

from chatty.configs import PersonaToolConfig

from .base import ToolBuilder


class FixedURLTool(BaseTool):
    """Tool for fetching website content."""

    # Tool-specific configuration
    url: str | None = None
    timeout: int = 30
    max_content_length: int = 1000

    @property
    def tool_type(self) -> str:
        return "url"

    def _run(
        self,
        *args,
        **kwargs,
    ) -> str:
        """Fetch website content synchronously."""
        import requests

        response = requests.get(self.url, timeout=self.timeout)
        response.raise_for_status()

        content = response.text
        if len(content) > self.max_content_length:
            content = content[: self.max_content_length] + "..."

        return content

    async def _arun(
        self,
        *args,
        **kwargs,
    ) -> str:
        """Fetch website content asynchronously."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(self.url)
            response.raise_for_status()

            content = response.text
            if len(content) > self.max_content_length:
                content = content[: self.max_content_length] + "..."

            return content

    @staticmethod
    def from_config(config: PersonaToolConfig) -> Self:
        """Build URL tool from configuration."""
        return FixedURLTool(
            name=config.name,
            description=config.description,
            url=config.args.get("url", None),
            args_schema=config.arg_schema,
            # timeout=config.timeout,
            # max_content_length=config.max_content_length,
        )
