from pydantic import BaseModel, Field


class ThirdPartyConfig(BaseModel):
    """Configuration for third-party integrations."""

    vector_database_uri: str = Field(
        default="postgresql://user:password@localhost:5432/chatty",
        description="Vector database connection URI",
    )
    redis_uri: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URI",
    )


class APIConfig(BaseModel):
    """API configuration settings."""

    route_prefix: str = Field(
        default="chatty",
        description="Service prefix inserted between /api/v1 and route names. "
        "Set to empty string to disable.",
    )

    # Rate limiting for the main /chat endpoint.
    chat_rate_limit_per_second: int = Field(
        default=3, description="API rate limit per minute"
    )


class CacheConfig(BaseModel):
    """Cache of chat responses configuration settings."""

    enabled: bool = Field(default=True, description="Enable caching")
    max_size: int = Field(
        default=30, description="Maximum number of entries in the cache"
    )
    similarity_threshold: float = Field(
        default=0.95, description="Cosine similarity threshold for memory hits"
    )
    admission_count: int = Field(
        default=3, description="Queries needed before caching answer"
    )
    ttl_hours: int = Field(default=24, description="TTL for dynamic memory entries")


class LLMConfig(BaseModel):
    """Configuration for LLM client."""

    endpoint: str = Field(
        default="http://localhost:8000/api/v1/",
        description="Model server endpoint URL",
    )
    api_key: str = Field(
        default="", description="API key for the language model service"
    )
    model_name: str | None = Field(
        "gpt-3.5-turbo", description="Name of the language model to use"
    )
    max_tokens: int = Field(
        default=4096, description="Maximum tokens to generate in a response"
    )
    temperature: float = Field(
        default=0.1, description="Sampling temperature for model responses"
    )
    top_p: float = Field(
        default=0.9, description="Top-p sampling parameter for model responses"
    )
    timeout: float = Field(
        default=300.0, description="Timeout for model requests in seconds"
    )
    max_retries: int = Field(
        default=3, description="Maximum number of retries for model requests"
    )


class ChatConfig(BaseModel):
    """Configuration for chat agent."""

    agent_name: str = Field(
        default="one_step", description="Supported agent name to spawn as chat service"
    )

    max_conversation_length: int = Field(
        default=3, description="Maximum conversation history length"
    )
