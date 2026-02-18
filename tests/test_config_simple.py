"""Simplified test for configuration reading."""

import os
from unittest.mock import patch

from chatty.configs.config import AppConfig, get_app_config


def test_config_works():
    """Test that configuration system works with environment variables."""

    env_vars = {
        "CHATTY_API__CHAT_RATE_LIMIT_PER_SECOND": "10",
        "CHATTY_CACHE__MAX_SIZE": "500",
    }

    with patch.dict(os.environ, env_vars, clear=False):
        config = AppConfig()

        assert config.api.chat_rate_limit_per_second == 10
        assert config.cache.max_size == 500


def test_singleton_get_app_config():
    """Test that get_app_config() works and returns same instance."""

    # Clear any existing singleton instance
    if hasattr(get_app_config, "_instance"):
        delattr(get_app_config, "_instance")

    config1 = get_app_config()
    config2 = get_app_config()

    # Should return same instance (singleton)
    assert config1 is config2
    assert isinstance(config1, AppConfig)

    # Verify persona loaded from YAML
    assert config1.persona.name == "Xinyu Huang"
    assert "chatty" in config1.persona.character

    # Verify new sources structure loaded
    assert "current_homepage" in config1.persona.sources
    assert "resume" in config1.persona.sources
    assert (
        config1.persona.sources["current_homepage"].content_url
        == "https://x3huang.dev"
    )

    # Verify tools loaded
    assert len(config1.persona.tools) == 1
    assert config1.persona.tools[0].name == "lookup"
    assert "current_homepage" in config1.persona.tools[0].sources

    # Verify embed loaded
    assert len(config1.persona.embed) == 2
    assert config1.persona.embed[0].source == "current_homepage"
    assert len(config1.persona.embed[0].match_hints) > 0
