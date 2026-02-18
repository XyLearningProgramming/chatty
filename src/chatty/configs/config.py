"""Configuration management using pydantic-settings.

**Not a singleton** — each call to ``get_app_config()`` re-reads config
from disk so that changes are picked up without restarting.

Priority order (highest first):

1. Environment variables (``CHATTY_`` prefix)
2. ``.env`` dotenv file
3. YAML config (``configs/config.yaml`` -- baked into the Docker image,
   replaced by the ConfigMap subPath mount in Kubernetes)
4. Init defaults / field defaults
5. File secrets
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from .persona import PersonaConfig
from .system import (
    APIConfig,
    CacheConfig,
    ChatConfig,
    ConcurrencyConfig,
    EmbeddingConfig,
    LLMConfig,
    PromptConfig,
    RagConfig,
    ThirdPartyConfig,
    TracingConfig,
)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

CONFIG_PY_PATH = Path(__file__).resolve()
PROJECT_ROOT = CONFIG_PY_PATH.parent.parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"

# YAML config file.  Baked into the Docker image at build time;
# in Kubernetes the ConfigMap is mounted over this path via subPath.
STATIC_CONFIG_FILE = CONFIG_DIR / "config.yaml"

# Prompt configuration file
PROMPT_CONFIG_FILE = CONFIG_DIR / "prompt.yml"

DOTENV_FILE_PATH = PROJECT_ROOT / ".env"
ENV_DELIMITER = "__"  # Nested environment variable delimiter
ENV_PREFIX = "CHATTY_"  # Environment variable prefix for Chatty

DEFAULT_ENCODING = "utf-8"


# ---------------------------------------------------------------------------
# Application config (re-created on every call — not a singleton)
# ---------------------------------------------------------------------------


class AppConfig(BaseSettings):
    """Application configuration."""

    model_config = SettingsConfigDict(
        env_file=DOTENV_FILE_PATH,
        env_file_encoding=DEFAULT_ENCODING,
        env_nested_delimiter=ENV_DELIMITER,
        env_prefix=ENV_PREFIX,
        case_sensitive=False,
        extra="ignore",
        yaml_file=STATIC_CONFIG_FILE,
        yaml_file_encoding=DEFAULT_ENCODING,
    )

    third_party: ThirdPartyConfig = Field(
        default_factory=ThirdPartyConfig,
        description="Third-party service configurations",
    )

    api: APIConfig = Field(
        default_factory=APIConfig,
        description="API configuration settings",
    )

    cache: CacheConfig = Field(
        default_factory=CacheConfig,
        description="Cache configuration settings",
    )

    llm: LLMConfig = Field(
        default_factory=LLMConfig,
        description="LLM client configuration settings",
    )

    chat: ChatConfig = Field(
        default_factory=ChatConfig, description="Configuration for chat agent."
    )

    concurrency: ConcurrencyConfig = Field(
        default_factory=ConcurrencyConfig,
        description="Concurrency gate settings for LLM agent runs",
    )

    persona: PersonaConfig = Field(
        default_factory=PersonaConfig,
        description="Author persona configuration (identity + knowledge)",
    )

    embedding: EmbeddingConfig = Field(
        default_factory=EmbeddingConfig,
        description="OpenAI-compatible embedding endpoint settings",
    )

    rag: RagConfig = Field(
        default_factory=RagConfig,
        description="RAG retrieval settings",
    )

    prompt: PromptConfig = Field(
        default_factory=PromptConfig,
        description="System prompt configuration",
    )

    tracing: TracingConfig = Field(
        default_factory=TracingConfig,
        description="OpenTelemetry tracing configuration",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources: list[PydanticBaseSettingsSource] = []

        # 1-2. Env vars and dotenv
        sources.append(env_settings)
        sources.append(dotenv_settings)

        # 3. YAML config (baked into image / replaced by ConfigMap in k8s)
        sources.append(YamlConfigSettingsSource(settings_cls))

        # 3.5. Prompt YAML (separate file)
        sources.append(_PromptYamlSettingsSource(settings_cls))

        # 4-5. Init defaults and file secrets
        sources.append(init_settings)
        sources.append(file_secret_settings)

        return tuple(sources)


class _PromptYamlSettingsSource(PydanticBaseSettingsSource):
    """Loads ``prompt.yml`` and validates it against :class:`PromptConfig`.

    The YAML is run through ``PromptConfig.model_validate`` so type
    errors surface immediately with clear Pydantic messages instead
    of propagating as opaque dicts.
    """

    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        self.settings_cls = settings_cls

    def __call__(self) -> dict[str, Any]:
        if not PROMPT_CONFIG_FILE.exists():
            return {}

        with open(PROMPT_CONFIG_FILE, encoding=DEFAULT_ENCODING) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            return {}

        prompt = PromptConfig.model_validate(data)
        return {"prompt": prompt.model_dump()}


def get_app_config() -> AppConfig:
    """Get the application configuration.

    Re-reads ``configs/config.yaml`` on every call so that changes are
    picked up immediately.
    """
    return AppConfig()


# ---------------------------------------------------------------------------
# Sub-config accessors for use with ``Depends()``
# ---------------------------------------------------------------------------


def get_llm_config() -> LLMConfig:
    """Return the LLM sub-config (re-read from disk)."""
    return get_app_config().llm


def get_api_config() -> APIConfig:
    """Return the API sub-config (re-read from disk)."""
    return get_app_config().api


def get_chat_config() -> ChatConfig:
    """Return the chat sub-config (re-read from disk)."""
    return get_app_config().chat


def get_embedding_config() -> EmbeddingConfig:
    """Return the embedding sub-config (re-read from disk)."""
    return get_app_config().embedding
