from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

from jinja2 import Template
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .persona import PersonaConfig


class ThirdPartyConfig(BaseModel):
    """Configuration for third-party integrations."""

    postgres_uri: str = Field(
        default="postgresql+asyncpg://app:password@postgres.db.svc.cluster.local:5432/app",
        description="PostgreSQL connection URI (asyncpg driver)",
    )
    postgres_pool_size: int = Field(
        default=5,
        description="SQLAlchemy connection pool size",
    )
    postgres_max_overflow: int = Field(
        default=10,
        description="SQLAlchemy max overflow connections beyond pool_size",
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
        default=2,
        description="Max requests per second from a single IP to /chat",
    )
    chat_global_rate_limit: int = Field(
        default=5,
        description="Max requests per second to /chat across all clients. "
        "Set to 0 to disable.",
    )
    dedup_window: timedelta = Field(
        default=timedelta(seconds=5),
        description="Duration during which identical (IP + query) pairs are "
        "rejected as duplicates. Set to 0 to disable fingerprint dedup.",
    )

    request_timeout: timedelta = Field(
        default=timedelta(minutes=5),
        description="Wall-clock timeout for the entire chat request lifecycle "
        "(queue wait + agent run). The stream is cancelled with an error "
        "event when exceeded.",
    )

    send_traceback: bool = Field(
        default=False,
        description="Include full Python tracebacks in client-facing SSE error "
        "events. Enable only in development; errors are always logged "
        "server-side regardless of this setting.",
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
        default=timedelta(hours=72),
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
        default=512, description="Maximum tokens to generate in a response"
    )
    context_window: int = Field(
        default=2048,
        description="Server context window size in tokens. "
        "Prompt assembly will truncate input to fit within "
        "context_window - max_tokens.",
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

    rag_no_think_enabled: bool = Field(
        default=True,
        description="(RAG only) Inject /no_think for trivial first-turn queries "
        "to skip LLM reasoning and bypass the RAG pipeline.",
    )
    rag_no_think_max_chars: int = Field(
        default=15,
        description="(RAG only) Maximum query length (chars, after strip) to "
        "classify as trivial for the /no_think shortcut.",
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
        default="Qwen3-0.6B",
        description="Embedding model name",
    )
    dimensions: int = Field(
        default=1024,
        description="Embedding vector dimensionality. "
        "Must match EMBEDDING_DIMENSIONS in infra/db/models.py; "
        "changing the DDL column width requires an alembic migration.",
    )
    max_input_tokens: int = Field(
        default=512,
        description="Maximum tokens for a single embedding input. "
        "Texts exceeding this are truncated before the API call.",
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
    cron_batch_size: int = Field(
        default=1,
        description="Max hints to embed per cron tick. "
        "Keeps each tick lightweight by spreading work across ticks.",
    )


class LoggingConfig(BaseModel):
    """Structured logging configuration."""

    level: str = Field(
        default="INFO",
        description="Root log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    json_output: bool = Field(
        default=True,
        description="Emit JSON lines (prod). "
        "Set to false for human-readable dev output.",
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
    """System prompt configuration.

    All templates use Jinja2 syntax and are loaded from
    ``configs/prompt.yaml``.
    """

    # --- system / RAG prompts ---

    system_prompt: str = Field(
        default="",
        description="Jinja2 template for system prompt with variables: "
        "persona_name, persona_character, persona_expertise",
    )
    rag_system_prompt: str = Field(
        default="",
        description="Jinja2 template for the final RAG system prompt. "
        "Receives {{ base }} (rendered system_prompt) and {{ content }} "
        "(retrieved context block). Required when using the RAG chat service.",
    )
    rag_context_section: str = Field(
        default="",
        description="Jinja2 template for a single RAG context section. "
        "Receives {{ source_id }}, {{ similarity }}, and {{ content }}. "
        "Sections are joined with double newlines to form the full context block.",
    )

    # --- tool prompts ---

    tool_description: str = Field(
        default="",
        description="Fallback tool description when YAML declaration has none.",
    )
    tool_source_field: str = Field(
        default="",
        description="Jinja2 template for the source field description. "
        "Receives {{ options }} (dict of source_id → desc).",
    )
    tool_source_hint: str = Field(
        default="",
        description="Jinja2 fallback per-source description. Receives {{ source_id }}.",
    )
    tool_error_unknown_source: str = Field(
        default="",
        description="Jinja2 error template for unknown source. "
        "Receives {{ source }} and {{ valid }}.",
    )

    # --- render helpers ---

    @staticmethod
    def _render(raw: str, **kwargs: object) -> str:
        return Template(raw.strip()).render(**kwargs)

    def render_system_prompt(self, persona: PersonaConfig) -> str:
        """Render ``system_prompt`` with persona identity fields."""
        raw = (self.system_prompt or "").strip()
        if not raw:
            raise ValueError(
                "system_prompt is required. Set it in configs/prompt.yaml."
            )
        return Template(raw).render(
            persona_name=persona.name,
            persona_character=", ".join(persona.character)
            if persona.character
            else None,
            persona_expertise=", ".join(persona.expertise)
            if persona.expertise
            else None,
        )

    def render_rag_prompt(self, *, base: str, content: str) -> str:
        """Render ``rag_system_prompt`` with the base system prompt and
        retrieved context.
        """
        raw = (self.rag_system_prompt or "").strip()
        if not raw:
            raise ValueError(
                "rag_system_prompt is required. Set it in configs/prompt.yaml."
            )
        return Template(raw).render(base=base, content=content)

    def render_rag_context_section(
        self, *, source_id: str, similarity: float, content: str
    ) -> str:
        """Render a single RAG context section heading + body."""
        return self._render(
            self.rag_context_section,
            source_id=source_id,
            similarity=similarity,
            content=content,
        )

    def render_tool_source_field(self, options: dict[str, str]) -> str:
        """Render the ``source`` field description for tool schemas."""
        return self._render(self.tool_source_field, options=options)

    def render_tool_source_hint(self, source_id: str) -> str:
        """Render a fallback per-source description."""
        return self._render(self.tool_source_hint, source_id=source_id)

    def render_tool_error(self, *, source: str, valid: str) -> str:
        """Render the unknown-source error message."""
        return self._render(self.tool_error_unknown_source, source=source, valid=valid)
