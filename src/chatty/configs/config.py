"""Configuration management using pydantic-settings."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from chatty.infra import singleton

from .persona import PersonaConfig
from .system import APIConfig, CacheConfig, ChatConfig, LLMConfig, ThirdPartyConfig
from .tools import ToolConfig

# Define paths relative to this file.

CONFIG_PY_PATH = Path(__file__).resolve()
PROJECT_ROOT = CONFIG_PY_PATH.parent.parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"

# Automatically discover all YAML files in configs directory
YAML_FILE_PATHS = sorted(CONFIG_DIR.glob("*.yaml"))
DOTENV_FILE_PATH = PROJECT_ROOT / ".env"
ENV_DELIMITER = "__"  # Nested environment variable delimiter
ENV_PREFIX = "CHATTY_"  # Environment variable prefix for Chatty

# Other constants usually not changed.

DEFUALT_ENCODING = "utf-8"


class AppConfig(BaseSettings):
    """Application configuration."""

    model_config = SettingsConfigDict(
        env_file=DOTENV_FILE_PATH,
        env_file_encoding=DEFUALT_ENCODING,
        env_nested_delimiter=ENV_DELIMITER,
        env_prefix=ENV_PREFIX,
        case_sensitive=False,
        extra="ignore",
        yaml_file=YAML_FILE_PATHS,
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

    persona: PersonaConfig = Field(
        default_factory=PersonaConfig,
        description="Author persona configuration",
    )

    tools: list[ToolConfig] = Field(
        default_factory=list,
        description="Agent tool definitions (top-level, shared)",
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
        return (
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            init_settings,
            file_secret_settings,
        )


@singleton
def get_app_config() -> AppConfig:
    """Get the application configuration."""
    return AppConfig()
