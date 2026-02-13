"""Unit tests for KnowledgeSection, URLDispatcherTool, and ToolRegistry."""

from datetime import timedelta
from unittest.mock import AsyncMock, patch

import pytest

from chatty.configs.persona import KnowledgeSection
from chatty.core.service.tools.processors import HtmlHeadTitleMeta
from chatty.core.service.tools.registry import ToolRegistry
from chatty.core.service.tools.url_tool import (
    URLDispatcherTool,
    _extract_text_from_pdf,
)

# ---------------------------------------------------------------------------
# KnowledgeSection
# ---------------------------------------------------------------------------


class TestKnowledgeSection:
    """Test the Pydantic knowledge section model."""

    def test_minimal_section(self):
        section = KnowledgeSection(title="my_section")
        assert section.title == "my_section"
        assert section.description == ""
        assert section.content == ""
        assert section.source_url == ""
        assert section.args == {}
        assert section.processors == []

    def test_full_section(self):
        section = KnowledgeSection(
            title="homepage",
            description="Fetch homepage",
            source_url="https://example.com",
            args={"timeout": 10},
            processors=["html_head_title_meta"],
        )
        assert section.source_url == "https://example.com"
        assert section.args["timeout"] == 10
        assert section.processors == ["html_head_title_meta"]

    def test_missing_required_fields(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            KnowledgeSection()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# URLDispatcherTool.from_configs
# ---------------------------------------------------------------------------


class TestURLDispatcherFromSections:
    """Test building the dispatcher from a list of KnowledgeSections."""

    def test_basic_from_sections(self):
        sections = [
            KnowledgeSection(
                title="homepage",
                description="Get homepage",
                source_url="https://example.com",
            ),
            KnowledgeSection(
                title="resume",
                description="Get resume",
                source_url="https://example.com/resume",
            ),
        ]
        tool = URLDispatcherTool.from_sections(sections)
        assert tool.name == "lookup"
        assert set(tool.routes.keys()) == {"homepage", "resume"}
        assert tool.routes["homepage"].args.url == "https://example.com"
        assert tool.routes["resume"].args.url == "https://example.com/resume"

    def test_single_section(self):
        sections = [
            KnowledgeSection(
                title="homepage",
                description="Get homepage",
                source_url="https://example.com",
            ),
        ]
        tool = URLDispatcherTool.from_sections(sections)
        assert tool.name == "lookup"
        assert set(tool.routes.keys()) == {"homepage"}

    def test_custom_timeout_and_max_length(self):
        sections = [
            KnowledgeSection(
                title="resume",
                description="Get resume",
                source_url="https://example.com/resume",
                args={"timeout": 5, "max_content_length": 500},
            ),
        ]
        tool = URLDispatcherTool.from_sections(sections)
        route = tool.routes["resume"]
        assert route.args.timeout == timedelta(seconds=5)
        assert route.args.max_content_length == 500

    def test_missing_url_defaults_to_empty(self):
        sections = [
            KnowledgeSection(
                title="empty",
                description="No URL",
            ),
        ]
        tool = URLDispatcherTool.from_sections(sections)
        assert tool.routes["empty"].args.url == ""

    def test_tool_type_attribute(self):
        assert URLDispatcherTool.tool_type == "url"

    def test_processors_stored_per_route(self):
        sections = [
            KnowledgeSection(
                title="homepage",
                description="d",
                source_url="http://x",
            ),
            KnowledgeSection(
                title="resume",
                description="d",
                source_url="http://y",
            ),
        ]
        processor = HtmlHeadTitleMeta()
        tool = URLDispatcherTool.from_sections(
            sections, processors={"homepage": [processor]}
        )
        assert len(tool.routes["homepage"].processors) == 1
        assert len(tool.routes["resume"].processors) == 0

    def test_args_schema_has_enum_and_descriptions(self):
        sections = [
            KnowledgeSection(
                title="homepage",
                description="Personal website meta info",
                source_url="http://x",
            ),
            KnowledgeSection(
                title="resume",
                description="Full resume PDF content",
                source_url="http://y",
            ),
        ]
        tool = URLDispatcherTool.from_sections(sections)
        schema = tool.args_schema.model_json_schema()
        source_schema = schema["properties"]["source"]
        # Enum lists valid values
        assert set(source_schema["enum"]) == {"homepage", "resume"}
        # Description explains what each source returns
        assert '"homepage"' in source_schema["description"]
        assert '"resume"' in source_schema["description"]
        assert "Personal website meta info" in source_schema["description"]
        assert "Full resume PDF content" in source_schema["description"]


# ---------------------------------------------------------------------------
# URLDispatcherTool._truncate
# ---------------------------------------------------------------------------


class TestURLDispatcherTruncate:
    """Test content truncation logic."""

    def test_short_content_not_truncated(self):
        assert URLDispatcherTool._truncate("hello", 100) == "hello"

    def test_long_content_truncated(self):
        result = URLDispatcherTool._truncate("abcdefgh", 5)
        assert result == "abcde..."

    def test_exact_length_not_truncated(self):
        assert URLDispatcherTool._truncate("abcde", 5) == "abcde"


# ---------------------------------------------------------------------------
# URLDispatcherTool._arun (async, mocked HTTP)
# ---------------------------------------------------------------------------


def _html_response(text: str, status_code: int = 200) -> AsyncMock:
    """Build a mock httpx.Response that looks like an HTML page."""
    resp = AsyncMock()
    resp.text = text
    resp.content = text.encode()
    resp.headers = {"content-type": "text/html; charset=utf-8"}
    resp.raise_for_status = lambda: None
    return resp


def _pdf_response(pdf_bytes: bytes, status_code: int = 200) -> AsyncMock:
    """Build a mock httpx.Response that looks like a PDF download."""
    resp = AsyncMock()
    resp.text = ""  # binary content decoded as text is useless
    resp.content = pdf_bytes
    resp.headers = {"content-type": "application/pdf"}
    resp.raise_for_status = lambda: None
    return resp


class TestURLDispatcherArun:
    """Test async fetch dispatch with mocked httpx."""

    @pytest.mark.asyncio
    async def test_arun_dispatches_by_name(self):
        sections = [
            KnowledgeSection(
                title="homepage",
                description="d",
                source_url="https://example.com",
                args={"max_content_length": 200},
            ),
        ]
        tool = URLDispatcherTool.from_sections(sections)

        mock_response = _html_response(
            "<html><head><title>Hi</title></head></html>"
        )

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            MockClient.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await tool._arun(source="homepage")
            mock_client.get.assert_called_once_with("https://example.com")
            assert "<html>" in result

    @pytest.mark.asyncio
    async def test_arun_truncates_large_response(self):
        sections = [
            KnowledgeSection(
                title="big",
                description="d",
                source_url="https://example.com",
                args={"max_content_length": 10},
            ),
        ]
        tool = URLDispatcherTool.from_sections(sections)

        mock_response = _html_response("x" * 100)

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            MockClient.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await tool._arun(source="big")
            assert len(result) == 13  # 10 chars + "..."
            assert result.endswith("...")

    @pytest.mark.asyncio
    async def test_arun_unknown_name_returns_error_message(self):
        sections = [
            KnowledgeSection(
                title="homepage",
                description="d",
                source_url="http://x",
            ),
        ]
        tool = URLDispatcherTool.from_sections(sections)
        result = await tool._arun(source="nonexistent")
        assert "Unknown source" in result
        assert "homepage" in result


# ---------------------------------------------------------------------------
# URLDispatcherTool – PDF content-type handling
# ---------------------------------------------------------------------------


class TestURLDispatcherPdf:
    """Test that PDF responses are automatically converted to text."""

    @staticmethod
    def _make_simple_pdf(text: str = "Hello from PDF") -> bytes:
        """Create a tiny valid PDF with *pymupdf* for testing."""
        import pymupdf

        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), text)
        data = doc.tobytes()
        doc.close()
        return data

    def test_is_pdf_detects_application_pdf(self):
        resp = AsyncMock()
        resp.headers = {"content-type": "application/pdf"}
        assert URLDispatcherTool._is_pdf(resp) is True

    def test_is_pdf_rejects_html(self):
        resp = AsyncMock()
        resp.headers = {"content-type": "text/html; charset=utf-8"}
        assert URLDispatcherTool._is_pdf(resp) is False

    def test_extract_text_from_pdf(self):
        pdf_bytes = self._make_simple_pdf("Resume content here")
        text = _extract_text_from_pdf(pdf_bytes)
        assert "Resume content here" in text

    @pytest.mark.asyncio
    async def test_arun_extracts_pdf_text(self):
        pdf_bytes = self._make_simple_pdf("Software Engineer")
        sections = [
            KnowledgeSection(
                title="resume",
                description="d",
                source_url="https://example.com/resume",
                args={"max_content_length": 5000},
            ),
        ]
        tool = URLDispatcherTool.from_sections(sections)

        mock_response = _pdf_response(pdf_bytes)

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            MockClient.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await tool._arun(source="resume")
            assert "Software Engineer" in result


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


class TestToolRegistry:
    """Test registry: grouping, error handling, and caching."""

    def test_creates_dispatcher_from_sections(self):
        sections = [
            KnowledgeSection(
                title="homepage",
                description="Get homepage",
                source_url="https://example.com",
            ),
            KnowledgeSection(
                title="resume",
                description="Get resume",
                source_url="https://example.com/resume",
            ),
        ]
        registry = ToolRegistry(sections)
        tools = registry.get_tools()
        # Two url sections -> one dispatcher tool
        assert len(tools) == 1
        assert tools[0].name == "lookup"

    def test_empty_sections(self):
        registry = ToolRegistry([])
        assert registry.get_tools() == []

    def test_inline_only_sections_produce_no_tools(self):
        """Sections without source_url should not produce tools."""
        sections = [
            KnowledgeSection(title="about", content="I am a dev"),
        ]
        registry = ToolRegistry(sections)
        assert registry.get_tools() == []

    def test_unknown_processor_raises(self):
        sections = [
            KnowledgeSection(
                title="t",
                description="d",
                source_url="http://x",
                processors=["does_not_exist"],
            ),
        ]
        with pytest.raises(NotImplementedError, match="does_not_exist"):
            ToolRegistry(sections)

    def test_get_tools_returns_copy(self):
        sections = [
            KnowledgeSection(
                title="t",
                description="d",
                source_url="http://x",
            ),
        ]
        registry = ToolRegistry(sections)
        tools1 = registry.get_tools()
        tools2 = registry.get_tools()
        # Same contents but different list objects
        assert tools1 == tools2
        assert tools1 is not tools2

    def test_processor_applied(self):
        """When html_head_title_meta processor is specified, verify it's wired."""
        sections = [
            KnowledgeSection(
                title="site",
                description="d",
                source_url="http://x",
                processors=["html_head_title_meta"],
            ),
        ]
        registry = ToolRegistry(sections)
        tools = registry.get_tools()
        assert len(tools) == 1
        assert tools[0].name == "lookup"
        # Verify the processor was stored in the route
        dispatcher = tools[0]
        assert isinstance(dispatcher, URLDispatcherTool)
        assert len(dispatcher.routes["site"].processors) == 1
        assert isinstance(dispatcher.routes["site"].processors[0], HtmlHeadTitleMeta)
