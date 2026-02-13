"""Configuration management for the CLI tool."""

import os

from pydantic import BaseModel, Field


class CLIConfig(BaseModel):
    """CLI configuration settings."""

    host: str = Field(
        default="localhost",
        description="Server host",
    )
    port: int = Field(
        default=8080,
        description="Server port",
    )
    api_path: str = Field(
        default="/api/v1/chatty/chat",
        description="API path for the chat endpoint",
    )

    @property
    def base_url(self) -> str:
        """Get the base URL for the API."""
        return f"http://{self.host}:{self.port}"

    @property
    def chat_url(self) -> str:
        """Get the full URL for the chat endpoint."""
        return f"{self.base_url}{self.api_path}"
