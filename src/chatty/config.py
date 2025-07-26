"""Configuration management using pydantic-settings."""

from pathlib import Path
from typing import List

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AuthorConfig(BaseModel):
    """Author persona configuration."""

    system_prompt: str = Field(description="System prompt for the chatbot")
    persona_details: str = Field(description="Detailed persona information")
    resume_uri: str = Field(default="", description="URI to resume/CV")
    blog_site_rss_uri: str = Field(default="", description="RSS feed URI")
    tools: List[dict] = Field(default_factory=list, description="Available tools")


class AppConfig(BaseSettings):
    """Application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        yaml_file=["config.yaml", "configs/config.yaml"],
        yaml_file_encoding="utf-8",
    )

    # Model server configuration
    model_server_endpoint: str = Field(
        default="http://localhost:8080/api/v1/chat/completions",
        description="Model server endpoint URL",
    )

    # Database configuration
    vector_database_uri: str = Field(
        default="postgresql://user:password@localhost:5432/chatty",
        description="Vector database connection URI",
    )

    redis_uri: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URI",
    )

    # API configuration
    host: str = Field(default="0.0.0.0", description="API server host")
    port: int = Field(default=8000, description="API server port")
    
    # Rate limiting
    rate_limit_per_minute: int = Field(
        default=60, description="API rate limit per minute"
    )

    # Memory/caching configuration
    memory_similarity_threshold: float = Field(
        default=0.95, description="Cosine similarity threshold for memory hits"
    )
    memory_admission_count: int = Field(
        default=3, description="Queries needed before caching answer"
    )
    memory_ttl_hours: int = Field(
        default=24, description="TTL for dynamic memory entries"
    )

    # Conversation configuration
    max_conversation_length: int = Field(
        default=10, description="Maximum conversation history length"
    )

    # Author configuration file path
    author_config_path: Path = Field(
        default=Path("author.yaml"),
        description="Path to author configuration file",
    )


def load_author_config(config_path: Path) -> AuthorConfig:
    """Load author configuration from YAML file."""
    import yaml

    if not config_path.exists():
        raise FileNotFoundError(f"Author config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return AuthorConfig(**data)