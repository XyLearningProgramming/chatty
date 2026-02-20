"""Unit tests for KnowledgeSource, URLDispatcherTool, and ToolRegistry."""

from datetime import timedelta
from unittest.mock import AsyncMock, patch

import pytest

from chatty.configs.persona import (
    EmbedDeclaration,
    KnowledgeSource,
    PersonaConfig,
    ProcessorWithArgs,
    ToolDeclaration,
)
from chatty.configs.system import PromptConfig
from chatty.core.service.tools.registry import ToolRegistry
from chatty.core.service.tools.url_tool import URLDispatcherTool
from chatty.infra.http_utils import PDF_CONTENT_TYPES, extract_text_from_pdf
from chatty.infra.processor_utils import (
    HtmlHeadTitleMeta,
    TruncateProcessor,
    get_processor,
)

_TEST_TOOL_TIMEOUT = timedelta(seconds=60)


def _prompt() -> PromptConfig:
    """Minimal PromptConfig with tool templates for testing."""
    return PromptConfig(
        tool_description="Default tool description",
        tool_source_field=(
            "Pick the source.\n"
            "{% for key, desc in options.items() %}"
            '  - "{{ key }}": {{ desc }}\n'
            "{% endfor %}"
        ),
        tool_source_hint="Fetch information about '{{ source_id }}'.",
        tool_error_unknown_source=(
            "Unknown source '{{ source }}'. Valid options: {{ valid }}"
        ),
    )


# ---------------------------------------------------------------------------
# KnowledgeSource
# ---------------------------------------------------------------------------


class TestKnowledgeSource:
    """Test the Pydantic knowledge source model."""

    def test_url_source(self):
        source = KnowledgeSource(
            description="A site",
            content_url="https://example.com",
        )
        assert source.content_url == "https://example.com"
        assert source.content == ""
        assert source.timeout == timedelta(seconds=30)
        assert source.processors == []

    def test_content_source(self):
        source = KnowledgeSource(
            description="Inline",
            content="Hello world",
        )
        assert source.content == "Hello world"
        assert source.content_url == ""

    def test_url_and_content_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="not both"):
            KnowledgeSource(
                content_url="https://example.com",
                content="also inline",
            )

    def test_neither_url_nor_content_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="either"):
            KnowledgeSource(description="empty")

    def test_with_processors(self):
        source = KnowledgeSource(
            content_url="https://example.com",
            processors=["html_head_title_meta"],
        )
        assert source.processors == ["html_head_title_meta"]

    def test_with_structured_processor(self):
        source = KnowledgeSource(
            content_url="https://example.com",
            processors=[ProcessorWithArgs(name="truncate", max_length=5000)],
        )
        assert len(source.processors) == 1


# ---------------------------------------------------------------------------
# PersonaConfig validation
# ---------------------------------------------------------------------------


class TestPersonaConfigValidation:
    """Test reference validation in PersonaConfig."""

    def test_valid_references(self):
        config = PersonaConfig(
            name="Test",
            sources={
                "src1": KnowledgeSource(content_url="https://example.com"),
            },
            tools=[
                ToolDeclaration(name="t", sources=["src1"]),
            ],
            embed=[
                EmbedDeclaration(source="src1"),
            ],
        )
        assert "src1" in config.sources

    def test_tool_unknown_source_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="unknown sources"):
            PersonaConfig(
                name="Test",
                sources={
                    "src1": KnowledgeSource(content_url="https://example.com"),
                },
                tools=[
                    ToolDeclaration(name="t", sources=["missing"]),
                ],
            )

    def test_embed_unknown_source_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="unknown source"):
            PersonaConfig(
                name="Test",
                sources={
                    "src1": KnowledgeSource(content_url="https://example.com"),
                },
                embed=[
                    EmbedDeclaration(source="missing"),
                ],
            )


# ---------------------------------------------------------------------------
# Processors
# ---------------------------------------------------------------------------


class TestProcessors:
    """Test get_processor and TruncateProcessor."""

    def test_get_processor_by_name(self):
        proc = get_processor("html_head_title_meta")
        assert isinstance(proc, HtmlHeadTitleMeta)

    def test_get_processor_with_kwargs(self):
        proc = get_processor("truncate", max_length=100)
        assert isinstance(proc, TruncateProcessor)

    def test_resolve_unknown_raises(self):
        with pytest.raises(NotImplementedError, match="does_not_exist"):
            get_processor("does_not_exist")

    def test_truncate_short_content(self):
        proc = TruncateProcessor(max_length=100)
        assert proc.process("hello") == "hello"

    def test_truncate_long_content(self):
        proc = TruncateProcessor(max_length=5)
        assert proc.process("abcdefgh") == "abcde..."

    def test_truncate_exact_length(self):
        proc = TruncateProcessor(max_length=5)
        assert proc.process("abcde") == "abcde"


# ---------------------------------------------------------------------------
# URLDispatcherTool.from_declaration
# ---------------------------------------------------------------------------


class TestURLDispatcherFromDeclaration:
    """Test building the dispatcher from declarations."""

    def _make_sources(self):
        return {
            "homepage": KnowledgeSource(
                description="Get homepage",
                content_url="https://example.com",
            ),
            "resume": KnowledgeSource(
                description="Get resume",
                content_url="https://example.com/resume",
            ),
        }

    def test_basic_from_declaration(self):
        sources = self._make_sources()
        decl = ToolDeclaration(
            name="lookup",
            type="url_dispatcher",
            sources=["homepage", "resume"],
        )
        tool = URLDispatcherTool.from_declaration(decl, sources, _prompt())
        assert tool.name == "lookup"
        assert set(tool.sources.keys()) == {
            "homepage",
            "resume",
        }
        assert tool.sources["homepage"].content_url == "https://example.com"
        assert tool.sources["resume"].content_url == "https://example.com/resume"

    def test_single_source(self):
        sources = self._make_sources()
        decl = ToolDeclaration(
            name="lookup",
            type="url_dispatcher",
            sources=["homepage"],
        )
        tool = URLDispatcherTool.from_declaration(decl, sources, _prompt())
        assert set(tool.sources.keys()) == {"homepage"}

    def test_custom_timeout(self):
        sources = {
            "resume": KnowledgeSource(
                description="Get resume",
                content_url="https://example.com/resume",
                timeout=5,
            ),
        }
        decl = ToolDeclaration(
            name="lookup",
            type="url_dispatcher",
            sources=["resume"],
        )
        tool = URLDispatcherTool.from_declaration(decl, sources, _prompt())
        assert tool.sources["resume"].timeout == timedelta(seconds=5)

    def test_from_declaration_returns_correct_type(self):
        sources = self._make_sources()
        decl = ToolDeclaration(
            name="lookup",
            type="url_dispatcher",
            sources=["homepage"],
        )
        tool = URLDispatcherTool.from_declaration(decl, sources, _prompt())
        assert isinstance(tool, URLDispatcherTool)

    def test_action_processors_from_declaration(self):
        sources = self._make_sources()
        decl = ToolDeclaration(
            name="lookup",
            type="url_dispatcher",
            sources=["homepage", "resume"],
            processors=[ProcessorWithArgs(name="truncate", max_length=5000)],
        )
        tool = URLDispatcherTool.from_declaration(decl, sources, _prompt())
        assert len(tool.action_processors) == 1
        assert isinstance(tool.action_processors[0], TruncateProcessor)

    def test_args_schema_has_enum_and_descriptions(self):
        sources = {
            "homepage": KnowledgeSource(
                description="Personal website meta info",
                content_url="http://x",
            ),
            "resume": KnowledgeSource(
                description="Full resume PDF content",
                content_url="http://y",
            ),
        }
        decl = ToolDeclaration(
            name="lookup",
            type="url_dispatcher",
            sources=["homepage", "resume"],
        )
        tool = URLDispatcherTool.from_declaration(decl, sources, _prompt())
        schema = tool.args_schema.model_json_schema()
        source_schema = schema["properties"]["source"]
        assert set(source_schema["enum"]) == {
            "homepage",
            "resume",
        }
        assert '"homepage"' in source_schema["description"]
        assert '"resume"' in source_schema["description"]
        assert "Personal website meta info" in source_schema["description"]
        assert "Full resume PDF content" in source_schema["description"]

    def test_declaration_description_used(self):
        sources = self._make_sources()
        decl = ToolDeclaration(
            name="lookup",
            type="url_dispatcher",
            sources=["homepage"],
            description="Custom tool description",
        )
        tool = URLDispatcherTool.from_declaration(decl, sources, _prompt())
        assert tool.description == "Custom tool description"

    def test_fallback_description_from_prompt(self):
        sources = self._make_sources()
        decl = ToolDeclaration(
            name="lookup",
            type="url_dispatcher",
            sources=["homepage"],
        )
        tool = URLDispatcherTool.from_declaration(decl, sources, _prompt())
        assert tool.description == "Default tool description"


# ---------------------------------------------------------------------------
# URLDispatcherTool._arun (async, mocked HTTP)
# ---------------------------------------------------------------------------


def _html_response(text: str, status_code: int = 200) -> AsyncMock:
    resp = AsyncMock()
    resp.text = text
    resp.content = text.encode()
    resp.headers = {"content-type": "text/html; charset=utf-8"}
    resp.raise_for_status = lambda: None
    return resp


def _pdf_response(pdf_bytes: bytes, status_code: int = 200) -> AsyncMock:
    resp = AsyncMock()
    resp.text = ""
    resp.content = pdf_bytes
    resp.headers = {"content-type": "application/pdf"}
    resp.raise_for_status = lambda: None
    return resp


class TestURLDispatcherArun:
    """Test async fetch dispatch with mocked httpx."""

    def _make_tool(self, sources, decl):
        return URLDispatcherTool.from_declaration(decl, sources, _prompt())

    @pytest.mark.asyncio
    async def test_arun_dispatches_by_name(self):
        sources = {
            "homepage": KnowledgeSource(
                description="d",
                content_url="https://example.com",
            ),
        }
        decl = ToolDeclaration(
            name="lookup",
            type="url_dispatcher",
            sources=["homepage"],
        )
        tool = self._make_tool(sources, decl)

        mock_response = _html_response("<html><head><title>Hi</title></head></html>")

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await tool._arun(source="homepage")
            mock_client.get.assert_called_once_with("https://example.com")
            assert "<html>" in result

    @pytest.mark.asyncio
    async def test_arun_with_action_truncate(self):
        sources = {
            "big": KnowledgeSource(
                description="d",
                content_url="https://example.com",
            ),
        }
        decl = ToolDeclaration(
            name="lookup",
            type="url_dispatcher",
            sources=["big"],
            processors=[ProcessorWithArgs(name="truncate", max_length=10)],
        )
        tool = URLDispatcherTool.from_declaration(decl, sources, _prompt())

        mock_response = _html_response("x" * 100)

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await tool._arun(source="big")
            assert len(result) == 13  # 10 chars + "..."
            assert result.endswith("...")

    @pytest.mark.asyncio
    async def test_arun_unknown_name_returns_error_message(self):
        sources = {
            "homepage": KnowledgeSource(
                description="d",
                content_url="http://x",
            ),
        }
        decl = ToolDeclaration(
            name="lookup",
            type="url_dispatcher",
            sources=["homepage"],
        )
        tool = self._make_tool(sources, decl)
        result = await tool._arun(source="nonexistent")
        assert "Unknown source" in result
        assert "homepage" in result

    @pytest.mark.asyncio
    async def test_arun_applies_source_and_action_processors(
        self,
    ):
        """Both source-level and action-level processors run."""
        sources = {
            "site": KnowledgeSource(
                description="d",
                content_url="https://example.com",
                processors=["html_head_title_meta"],
            ),
        }
        decl = ToolDeclaration(
            name="lookup",
            type="url_dispatcher",
            sources=["site"],
            processors=[ProcessorWithArgs(name="truncate", max_length=50)],
        )
        tool = URLDispatcherTool.from_declaration(decl, sources, _prompt())

        html = (
            "<html><head><title>My Site</title>"
            '<meta name="description" content="desc">'
            "</head><body>" + "x" * 500 + "</body></html>"
        )
        mock_response = _html_response(html)

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await tool._arun(source="site")
            assert "My Site" in result
            assert len(result) <= 53  # 50 + "..."


# ---------------------------------------------------------------------------
# URLDispatcherTool -- PDF handling
# ---------------------------------------------------------------------------


class TestURLDispatcherPdf:
    """Test that PDF responses are automatically converted to text."""

    @staticmethod
    def _make_simple_pdf(
        text: str = "Hello from PDF",
    ) -> bytes:
        import pymupdf

        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), text)
        data = doc.tobytes()
        doc.close()
        return data

    def test_is_pdf_detects_application_pdf(self):
        ct = "application/pdf"
        assert any(pdf in ct for pdf in PDF_CONTENT_TYPES)

    def test_is_pdf_rejects_html(self):
        ct = "text/html; charset=utf-8"
        assert not any(pdf in ct for pdf in PDF_CONTENT_TYPES)

    def test_extract_text_from_pdf(self):
        pdf_bytes = self._make_simple_pdf("Resume content here")
        text = extract_text_from_pdf(pdf_bytes)
        assert "Resume content here" in text

    @pytest.mark.asyncio
    async def test_arun_extracts_pdf_text(self):
        pdf_bytes = self._make_simple_pdf("Software Engineer")
        sources = {
            "resume": KnowledgeSource(
                description="d",
                content_url="https://example.com/resume",
            ),
        }
        decl = ToolDeclaration(
            name="lookup",
            type="url_dispatcher",
            sources=["resume"],
        )
        tool = URLDispatcherTool.from_declaration(decl, sources, _prompt())

        mock_response = _pdf_response(pdf_bytes)

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await tool._arun(source="resume")
            assert "Software Engineer" in result


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


class TestToolRegistry:
    """Test registry: building, error handling, and caching."""

    def _make_registry(self, sources, tools):
        return ToolRegistry(
            tools=tools,
            sources=sources,
            prompt=_prompt(),
            tool_timeout=_TEST_TOOL_TIMEOUT,
        )

    def test_creates_dispatcher_from_declarations(self):
        sources = {
            "homepage": KnowledgeSource(
                description="Get homepage",
                content_url="https://example.com",
            ),
            "resume": KnowledgeSource(
                description="Get resume",
                content_url="https://example.com/resume",
            ),
        }
        tools = [
            ToolDeclaration(
                name="lookup",
                type="url_dispatcher",
                sources=["homepage", "resume"],
            ),
        ]
        registry = self._make_registry(sources, tools)
        result = registry.get_tools()
        assert len(result) == 1
        assert result[0].name == "lookup"

    def test_empty_tools(self):
        registry = self._make_registry({}, [])
        assert registry.get_tools() == []

    def test_all_declarations_become_url_dispatcher(self):
        sources = {
            "s": KnowledgeSource(
                description="d",
                content_url="http://x",
            ),
        }
        tools = [
            ToolDeclaration(
                name="t",
                type="url_dispatcher",
                sources=["s"],
            ),
        ]
        registry = self._make_registry(sources, tools)
        result = registry.get_tools()
        assert len(result) == 1
        assert isinstance(result[0], URLDispatcherTool)

    def test_unknown_action_processor_raises(self):
        sources = {
            "s": KnowledgeSource(
                content_url="http://x",
            ),
        }
        tools = [
            ToolDeclaration(
                name="t",
                type="url_dispatcher",
                sources=["s"],
                processors=["does_not_exist"],
            ),
        ]
        with pytest.raises(NotImplementedError, match="does_not_exist"):
            self._make_registry(sources, tools)

    def test_get_tools_returns_copy(self):
        sources = {
            "s": KnowledgeSource(
                description="d",
                content_url="http://x",
            ),
        }
        tools = [
            ToolDeclaration(
                name="t",
                type="url_dispatcher",
                sources=["s"],
            ),
        ]
        registry = self._make_registry(sources, tools)
        tools1 = registry.get_tools()
        tools2 = registry.get_tools()
        assert tools1 == tools2
        assert tools1 is not tools2

    def test_action_processors_stored_on_tool(self):
        sources = {
            "site": KnowledgeSource(
                description="d",
                content_url="http://x",
            ),
        }
        tools = [
            ToolDeclaration(
                name="lookup",
                type="url_dispatcher",
                sources=["site"],
                processors=[ProcessorWithArgs(name="truncate", max_length=5000)],
            ),
        ]
        registry = self._make_registry(sources, tools)
        result = registry.get_tools()
        assert len(result) == 1
        dispatcher = result[0]
        assert isinstance(dispatcher, URLDispatcherTool)
        assert len(dispatcher.action_processors) == 1
        assert isinstance(dispatcher.action_processors[0], TruncateProcessor)
