"""Test configuration reading from multiple sources."""

import os
from unittest.mock import patch

from chatty.configs.config import AppConfig


def test_config_env_vars_work(self):
    """Test that environment variables work for configuration."""

    env_vars = {
        "CHATTY_API__CHAT_RATE_LIMIT_PER_SECOND": "10",
        "CHATTY_CACHE__MAX_SIZE": "500",
    }

    with patch.dict(os.environ, env_vars, clear=False):
        config = AppConfig()

        assert config.api.chat_rate_limit_per_second == 10
        assert config.cache.max_size == 500
