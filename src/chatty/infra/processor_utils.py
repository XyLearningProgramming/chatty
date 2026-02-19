"""Text processor protocol, built-in implementations, and registry.

Lives in infra so that both config models and the service layer can
import without circular dependencies.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Protocol

from bs4 import BeautifulSoup


class TextProcessor(Protocol):
    """Structural protocol — anything with ``.process(str) -> str``."""

    @property
    def processor_name(self) -> str:
        return ""

    @abstractmethod
    def process(self, content: str) -> str: ...


# ---------------------------------------------------------------------------
# Built-in processors
# ---------------------------------------------------------------------------


class HtmlHeadTitleMeta(TextProcessor):
    """Extract title and meta information from HTML head section."""

    processor_name = "html_head_title_meta"

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
        soup = BeautifulSoup(html_content, "html.parser")

        new_head = soup.new_tag("head")

        title_tag = soup.find("title")
        if title_tag:
            new_head.append(title_tag.extract())

        meta_tags = soup.find_all("meta")
        for meta in meta_tags:
            name = meta.get("name")
            prop = meta.get("property")
            if (name and name in self.meta_attrs) or (prop and prop in self.meta_attrs):
                new_head.append(meta.extract())

        html_tag = soup.new_tag("html")
        html_tag.append(new_head)

        return str(html_tag)


class TruncateProcessor(TextProcessor):
    """Truncate content to a maximum character length."""

    processor_name = "truncate"

    def __init__(self, max_length: int = 8000) -> None:
        self._max_length = max_length

    def process(self, content: str) -> str:
        if len(content) > self._max_length:
            return content[: self._max_length] + "..."
        return content


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_KNOWN_PROCESSORS: dict[str, type] = {
    HtmlHeadTitleMeta.processor_name: HtmlHeadTitleMeta,
    TruncateProcessor.processor_name: TruncateProcessor,
}


def get_processor(name: str, **kwargs: Any) -> TextProcessor:
    """Look up a processor by name and return an instance."""
    cls = _KNOWN_PROCESSORS.get(name)
    if cls is None:
        raise NotImplementedError(f"Processor '{name}' is not supported.")
    return cls(**kwargs)
