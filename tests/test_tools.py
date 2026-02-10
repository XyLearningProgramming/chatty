"""Unit tests for ToolConfig, FixedURLTool, and ToolRegistry."""

from unittest.mock import AsyncMock, patch

import pytest

from chatty.configs.tools import ToolConfig
from chatty.core.service.tools.registry import ToolRegistry
from chatty.core.service.tools.url_tool import FixedURLTool

# ---------------------------------------------------------------------------
# ToolConfig
# ---------------------------------------------------------------------------


class TestToolConfig:
    """Test the Pydantic tool config model."""

    def test_minimal_config(self):
        cfg = ToolConfig(
            name="my_tool",
            description="A tool",
            tool_type="url",
        )
        assert cfg.name == "my_tool"
        assert cfg.description == "A tool"
        assert cfg.tool_type == "url"
        assert cfg.args == {}
        assert cfg.processors == []

    def test_full_config(self):
        cfg = ToolConfig(
            name="homepage",
            description="Fetch homepage",
            tool_type="url",
            args={"url": "https://example.com", "timeout": 10},
            processors=["html_head_title_meta"],
        )
        assert cfg.args["url"] == "https://example.com"
        assert cfg.args["timeout"] == 10
        assert cfg.processors == ["html_head_title_meta"]

    def test_missing_required_fields(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ToolConfig()  # type: ignore[call-arg]

        with pytest.raises(ValidationError):
            ToolConfig(name="x", description="y")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# FixedURLTool.from_config
# ---------------------------------------------------------------------------


class TestFixedURLToolFromConfig:
    """Test building FixedURLTool from ToolConfig."""

    def test_basic_from_config(self):
        cfg = ToolConfig(
            name="homepage",
            description="Get homepage",
            tool_type="url",
            args={"url": "https://example.com"},
        )
        tool = FixedURLTool.from_config(cfg)
        assert tool.name == "homepage"
        assert tool.description == "Get homepage"
        assert tool.url == "https://example.com"
        assert tool.timeout == 30  # default
        assert tool.max_content_length == 1000  # default

    def test_custom_timeout_and_max_length(self):
        cfg = ToolConfig(
            name="resume",
            description="Get resume",
            tool_type="url",
            args={
                "url": "https://example.com/resume",
                "timeout": 5,
                "max_content_length": 500,
            },
        )
        tool = FixedURLTool.from_config(cfg)
        assert tool.timeout == 5
        assert tool.max_content_length == 500

    def test_missing_url_defaults_to_empty(self):
        cfg = ToolConfig(
            name="empty",
            description="No URL",
            tool_type="url",
            args={},
        )
        tool = FixedURLTool.from_config(cfg)
        assert tool.url == ""

    def test_tool_type_attribute(self):
        assert FixedURLTool.tool_type == "url"


# ---------------------------------------------------------------------------
# FixedURLTool.truncate
# ---------------------------------------------------------------------------


class TestFixedURLToolTruncate:
    """Test content truncation logic."""

    def test_short_content_not_truncated(self):
        tool = FixedURLTool(
            name="t",
            description="d",
            url="http://x",
            max_content_length=100,
        )
        assert tool._truncate("hello") == "hello"

    def test_long_content_truncated(self):
        tool = FixedURLTool(
            name="t",
            description="d",
            url="http://x",
            max_content_length=5,
        )
        result = tool._truncate("abcdefgh")
        assert result == "abcde..."

    def test_exact_length_not_truncated(self):
        tool = FixedURLTool(
            name="t",
            description="d",
            url="http://x",
            max_content_length=5,
        )
        assert tool._truncate("abcde") == "abcde"


# ---------------------------------------------------------------------------
# FixedURLTool._arun (async, mocked HTTP)
# ---------------------------------------------------------------------------


class TestFixedURLToolArun:
    """Test async fetch with mocked httpx."""

    @pytest.mark.asyncio
    async def test_arun_fetches_url(self):
        tool = FixedURLTool(
            name="homepage",
            description="d",
            url="https://example.com",
            max_content_length=50,
        )

        mock_response = AsyncMock()
        mock_response.text = "<html><head><title>Hi</title></head></html>"
        mock_response.raise_for_status = lambda: None

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            MockClient.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await tool._arun()
            mock_client.get.assert_called_once_with("https://example.com")
            assert "<html>" in result

    @pytest.mark.asyncio
    async def test_arun_truncates_large_response(self):
        tool = FixedURLTool(
            name="t",
            description="d",
            url="https://example.com",
            max_content_length=10,
        )

        mock_response = AsyncMock()
        mock_response.text = "x" * 100
        mock_response.raise_for_status = lambda: None

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            MockClient.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await tool._arun()
            assert len(result) == 13  # 10 chars + "..."
            assert result.endswith("...")


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


class TestToolRegistry:
    """Test registry: creation from configs, error handling."""

    def test_creates_tools_from_configs(self):
        configs = [
            ToolConfig(
                name="homepage",
                description="Get homepage",
                tool_type="url",
                args={"url": "https://example.com"},
            ),
            ToolConfig(
                name="resume",
                description="Get resume",
                tool_type="url",
                args={"url": "https://example.com/resume"},
            ),
        ]
        registry = ToolRegistry(configs)
        tools = registry.get_tools()
        assert len(tools) == 2
        assert tools[0].name == "homepage"
        assert tools[1].name == "resume"

    def test_empty_configs(self):
        registry = ToolRegistry([])
        assert registry.get_tools() == []

    def test_unknown_tool_type_raises(self):
        configs = [
            ToolConfig(
                name="bad",
                description="Bad tool",
                tool_type="nonexistent",
            ),
        ]
        with pytest.raises(NotImplementedError, match="nonexistent"):
            ToolRegistry(configs)

    def test_unknown_processor_raises(self):
        configs = [
            ToolConfig(
                name="t",
                description="d",
                tool_type="url",
                args={"url": "http://x"},
                processors=["does_not_exist"],
            ),
        ]
        with pytest.raises(NotImplementedError, match="does_not_exist"):
            ToolRegistry(configs)

    def test_get_tools_returns_copy(self):
        configs = [
            ToolConfig(
                name="t",
                description="d",
                tool_type="url",
                args={"url": "http://x"},
            ),
        ]
        registry = ToolRegistry(configs)
        tools1 = registry.get_tools()
        tools2 = registry.get_tools()
        # Same contents but different list objects
        assert tools1 == tools2
        assert tools1 is not tools2

    def test_processor_applied(self):
        """When html_head_title_meta processor is specified, verify it's wired."""
        configs = [
            ToolConfig(
                name="site",
                description="d",
                tool_type="url",
                args={"url": "http://x"},
                processors=["html_head_title_meta"],
            ),
        ]
        registry = ToolRegistry(configs)
        tools = registry.get_tools()
        assert len(tools) == 1
        # Tool should still be callable (processor wraps _run/_arun)
        assert tools[0].name == "site"
