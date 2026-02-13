"""Configuration management using pydantic-settings.

**Not a singleton** — each call to ``get_app_config()`` re-reads config
from disk so that ConfigMap updates are picked up without restarting.

Priority order (highest first):

1. ConfigMap YAML (path from ``CHATTY_CONFIGMAP_FILE`` env var, hot-reloadable)
2. Environment variables (``CHATTY_`` prefix)
3. ``.env`` dotenv file
4. Static YAML (``configs/config.yaml`` -- baked into the Docker image)
5. Init defaults / field defaults
6. File secrets

Caveat: only the *contents* of the known config files are dynamic.
The file paths are resolved at import time; adding brand-new files
after startup requires a process restart.
"""

import os
from pathlib import Path
from typing import Any, Optional

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
)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

CONFIG_PY_PATH = Path(__file__).resolve()
PROJECT_ROOT = CONFIG_PY_PATH.parent.parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"

# Single known config file (baked into the Docker image).
STATIC_CONFIG_FILE = CONFIG_DIR / "config.yaml"

# Prompt configuration file
PROMPT_CONFIG_FILE = CONFIG_DIR / "prompt.yml"

# Optional ConfigMap override file.  The Helm chart injects the env var
# ``CHATTY_CONFIGMAP_FILE`` pointing to the mounted config file when a
# ConfigMap is deployed.  No hardcoded path — the chart decides where to
# mount.
_configmap_env = os.environ.get("CHATTY_CONFIGMAP_FILE")
CONFIGMAP_CONFIG_FILE: Optional[Path] = Path(_configmap_env) if _configmap_env else None

DOTENV_FILE_PATH = PROJECT_ROOT / ".env"
ENV_DELIMITER = "__"  # Nested environment variable delimiter
ENV_PREFIX = "CHATTY_"  # Environment variable prefix for Chatty

DEFUALT_ENCODING = "utf-8"


# ---------------------------------------------------------------------------
# Application config (re-created on every call — not a singleton)
# ---------------------------------------------------------------------------


class AppConfig(BaseSettings):
    """Application configuration."""

    model_config = SettingsConfigDict(
        env_file=DOTENV_FILE_PATH,
        env_file_encoding=DEFUALT_ENCODING,
        env_nested_delimiter=ENV_DELIMITER,
        env_prefix=ENV_PREFIX,
        case_sensitive=False,
        extra="ignore",
        yaml_file=STATIC_CONFIG_FILE,
        yaml_file_encoding=DEFUALT_ENCODING,
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

        # 1. ConfigMap YAML -- highest priority (hot-reloadable)
        if CONFIGMAP_CONFIG_FILE is not None and CONFIGMAP_CONFIG_FILE.is_file():
            sources.append(
                YamlConfigSettingsSource(
                    settings_cls,
                    yaml_file=CONFIGMAP_CONFIG_FILE,
                )
            )

        # 2-3. Env vars and dotenv
        sources.append(env_settings)
        sources.append(dotenv_settings)

        # 4. Static YAML (baked into image)
        sources.append(YamlConfigSettingsSource(settings_cls))

        # 4.5. Prompt YAML (separate file)
        sources.append(_PromptYamlSettingsSource(settings_cls))

        # 5-6. Init defaults and file secrets
        sources.append(init_settings)
        sources.append(file_secret_settings)

        return tuple(sources)


class _PromptYamlSettingsSource(PydanticBaseSettingsSource):
    """Custom settings source that loads prompt.yml file."""

    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        self.settings_cls = settings_cls

    def __call__(self) -> dict[str, Any]:
        """Load prompt config from prompt.yml file."""
        if not PROMPT_CONFIG_FILE.exists():
            return {}

        try:
            with open(PROMPT_CONFIG_FILE, encoding=DEFUALT_ENCODING) as f:
                data = yaml.safe_load(f)
                if data and "system_prompt" in data:
                    return {"prompt": {"system_prompt": data["system_prompt"]}}
        except Exception:
            pass

        return {}


def get_app_config() -> AppConfig:
    """Get the application configuration.

    Re-reads ``configs/config.yaml`` (and the ConfigMap override when
    present) on every call so that hot-reloaded values are picked up
    immediately.
    """
    return AppConfig()
