"""Simplified test for configuration reading."""

import os
from unittest.mock import patch

from chatty.configs.config import AppConfig, get_app_config


class TestConfigSimple:
    """Test basic configuration functionality."""
    
    def test_config_works(self):
        """Test that configuration system works with environment variables."""
        
        env_vars = {
            'CHATTY_API__CHAT_RATE_LIMIT_PER_SECOND': '10',
            'CHATTY_CACHE__MAX_SIZE': '500',
        }
        
        with patch.dict(os.environ, env_vars, clear=False):
            config = AppConfig()
            
            assert config.api.chat_rate_limit_per_second == 10
            assert config.cache.max_size == 500

    def test_singleton_get_app_config(self):
        """Test that get_app_config() works and returns same instance."""

        # Clear any existing singleton instance
        if hasattr(get_app_config, "_instance"):
            delattr(get_app_config, "_instance")

        config1 = get_app_config()
        config2 = get_app_config()

        # Should return same instance (singleton)
        assert config1 is config2
        assert isinstance(config1, AppConfig)
        
        # Verify persona loaded from separate file
        assert config1.persona.name == "Xinyu Huang"
        assert "chatty" in config1.persona.character
