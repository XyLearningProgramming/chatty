"""Content processors for extracting structured data from various sources."""

from abc import abstractmethod
from typing import Protocol

from bs4 import BeautifulSoup


class Processor(Protocol):
    """Abstract base class for content processors."""

    @abstractmethod
    def process(self, content: str) -> str:
        """Process the given content and return the modified content."""
        pass


def with_processors(*processors: Processor):
    """Decorator to wrap a tool with content processors."""

    def decorator(tool_class):
        """Class decorator that wraps _run and _arun methods."""

        # Store original methods
        original_run = tool_class._run
        original_arun = tool_class._arun

        def _run_with_processors(
            self,
            *args,
            **kwargs,
        ) -> str:
            """Wrapped _run method that applies processors to the result."""
            # Call original _run method
            result = original_run(self, *args, **kwargs)

            # Apply processors in sequence
            for processor in processors:
                result = processor.process(result)

            return result

        async def _arun_with_processors(
            self,
            *args,
            **kwargs,
        ) -> str:
            """Wrapped _arun method that applies processors to the result."""
            # Call original _arun method
            result = await original_arun(self, *args, **kwargs)

            # Apply processors in sequence
            for processor in processors:
                result = processor.process(result)

            return result

        # Replace the methods on the class
        tool_class._run = _run_with_processors
        tool_class._arun = _arun_with_processors

        return tool_class

    return decorator


class HtmlHeadTitleMeta(Processor):
    """Processor for extracting title and meta information from HTML head section."""

    # Predetermined meta attributes to extract
    meta_attrs = {
        "description",
        "keywords",
        "author",
        "og:url",
        "og:image",
        "og:title",
        "og:description",
    }

    def process(self, html_content: str) -> str:
        """Extract title and meta information from HTML content by preserving original elements.

        Args:
            html_content: Raw HTML content

        Returns:
            Clean HTML containing only title and wanted meta tags
        """
        soup = BeautifulSoup(html_content, "html.parser")

        # Create new head element to hold filtered content
        new_head = soup.new_tag("head")

        # Preserve title element if it exists
        title_tag = soup.find("title")
        if title_tag:
            new_head.append(title_tag.extract())

        # Preserve wanted meta tags
        meta_tags = soup.find_all("meta")
        for meta in meta_tags:
            # Check if this meta tag has attributes we want to preserve
            name = meta.get("name")
            prop = meta.get("property")

            if (name and name in self.meta_attrs) or (prop and prop in self.meta_attrs):
                new_head.append(meta.extract())

        # Create minimal HTML structure
        html_tag = soup.new_tag("html")
        html_tag.append(new_head)

        return str(html_tag)
