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
    model_server_endpoint: str = Field(
        default="http://localhost:8080/api/v1/",
        description="Model server endpoint URL",
    )


class APIConfig(BaseModel):
    """API configuration settings."""

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


class ChatConfig(BaseModel):
    """Configuration for chat settings."""

    max_conversation_length: int = Field(
        default=3, description="Maximum conversation history length"
    )
    max_response_tokens: int = Field(
        default=2048, description="Maximum tokens in a single response"
    )
    temperature: float = Field(
        default=0.5, description="Sampling temperature for model responses"
    )
    top_p: float = Field(
        default=0.9, description="Top-p sampling parameter for model responses"
    )
