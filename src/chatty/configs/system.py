from datetime import timedelta

from pydantic import BaseModel, Field


class ThirdPartyConfig(BaseModel):
    """Configuration for third-party integrations."""

    postgres_uri: str = Field(
        default="postgresql+asyncpg://app:password@postgres.db.svc.cluster.local:5432/app",
        description="PostgreSQL connection URI (asyncpg driver for async FastAPI usage)",
    )
    redis_uri: str = Field(
        default="redis://:password@redis-master.db.svc.cluster.local:6379/0",
        description="Redis connection URI (redis-py asyncio compatible)",
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

    request_timeout: timedelta = Field(
        default=timedelta(minutes=5),
        description="Wall-clock timeout for the entire chat request lifecycle "
        "(queue wait + agent run). The stream is cancelled with an error "
        "event when exceeded.",
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
    ttl: timedelta = Field(
        default=timedelta(hours=24),
        description="TTL for dynamic memory entries",
    )


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
    model_timeout: timedelta = Field(
        default=timedelta(seconds=300),
        description="Timeout for model requests",
    )
    max_retries: int = Field(
        default=3, description="Maximum number of retries for model requests"
    )


class ConcurrencyConfig(BaseModel):
    """Concurrency gate settings for LLM agent runs."""

    max_concurrency: int = Field(
        default=1,
        description="Maximum number of parallel agent runs (semaphore slots)",
    )
    inbox_max_size: int = Field(
        default=10,
        description="Maximum requests admitted into the gate at once. "
        "Overflow is rejected with HTTP 429.",
    )
    acquire_timeout: timedelta = Field(
        default=timedelta(seconds=30),
        description="Maximum time a queued request waits to acquire a "
        "concurrency slot before giving up with a 'too busy' error.",
    )
    slot_timeout: timedelta = Field(
        default=timedelta(minutes=10),
        description="TTL for Redis concurrency keys (crash safety)",
    )


class ChatConfig(BaseModel):
    """Configuration for chat agent."""

    agent_name: str = Field(
        default="one_step", description="Supported agent name to spawn as chat service"
    )

    max_conversation_length: int = Field(
        default=3, description="Maximum conversation history length"
    )

    tool_timeout: timedelta = Field(
        default=timedelta(seconds=60),
        description="Per-tool execution timeout. Applied to every tool "
        "invocation (fetch + post-processing).",
    )


class EmbeddingConfig(BaseModel):
    """Configuration for the OpenAI-compatible embedding endpoint."""

    endpoint: str = Field(
        default="http://localhost:8000/api/v1/",
        description="OpenAI-compatible /embeddings base URL",
    )
    api_key: str = Field(
        default="",
        description="API key for the embedding service",
    )
    model_name: str = Field(
        default="text-embedding-ada-002",
        description="Embedding model name",
    )
    dimensions: int = Field(
        default=1536,
        description="Embedding vector dimensionality",
    )


class RagConfig(BaseModel):
    """RAG retrieval settings."""

    top_k: int = Field(
        default=3,
        description="Number of top sections to retrieve",
    )
    similarity_threshold: float = Field(
        default=0.7,
        description="Minimum cosine similarity to include a section",
    )
    cron_interval: int = Field(
        default=30,
        description="Seconds between embedding cron ticks",
    )


class TracingConfig(BaseModel):
    """OpenTelemetry tracing configuration."""

    enabled: bool = Field(
        default=False,
        description="Enable OpenTelemetry tracing export",
    )
    service_name: str = Field(
        default="chatty",
        description="Service name used in the trace provider resource",
    )
    endpoint: str = Field(
        default="",
        description="Grafana Tempo OTLP HTTP endpoint URL",
    )
    username: str = Field(
        default="",
        description="Grafana Tempo basic-auth username",
    )
    password: str = Field(
        default="",
        description="Grafana Tempo basic-auth password",
    )
    sample_rate: float = Field(
        default=0.1,
        description="Trace sampling rate (0.0-1.0)",
    )
    excluded_urls: list[str] = Field(
        default=["/metrics", "/health"],
        description="URL paths to exclude from tracing",
    )


class PromptConfig(BaseModel):
    """System prompt configuration."""

    system_prompt: str = Field(
        default="",
        description="Jinja2 template for system prompt with variables: "
        "persona_name, persona_character, persona_expertise",
    )
