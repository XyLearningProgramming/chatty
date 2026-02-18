"""Content processors for transforming fetched content.

Each processor implements the ``Processor`` protocol and is registered
by ``processor_name`` so that YAML configs can reference them by string.

The ``resolve_processors`` helper converts a list of ``ProcessorRef``
values (plain strings or structured dicts) into instantiated
``Processor`` objects.  It is shared by the tool registry and the
embedding cron.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol

from bs4 import BeautifulSoup

if TYPE_CHECKING:
    from chatty.configs.persona import ProcessorRef


class Processor(Protocol):
    """Protocol for content processors."""

    @property
    def processor_name(self) -> str:
        return ""

    @abstractmethod
    def process(self, content: str) -> str:
        """Process the given content and return the modified content."""
        ...


# ---------------------------------------------------------------------------
# Built-in processors
# ---------------------------------------------------------------------------


class HtmlHeadTitleMeta:
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
            if (name and name in self.meta_attrs) or (
                prop and prop in self.meta_attrs
            ):
                new_head.append(meta.extract())

        html_tag = soup.new_tag("html")
        html_tag.append(new_head)

        return str(html_tag)


class TruncateProcessor:
    """Truncate content to a maximum character length.

    Replaces the hardcoded ``_truncate()`` helpers that previously
    lived in ``url_tool.py`` and ``cron.py``.
    """

    processor_name = "truncate"

    def __init__(self, max_length: int = 8000) -> None:
        self._max_length = max_length

    def process(self, content: str) -> str:
        if len(content) > self._max_length:
            return content[: self._max_length] + "..."
        return content


# ---------------------------------------------------------------------------
# Processor registry + resolver
# ---------------------------------------------------------------------------

_KNOWN_PROCESSORS: dict[str, type] = {
    HtmlHeadTitleMeta.processor_name: HtmlHeadTitleMeta,
    TruncateProcessor.processor_name: TruncateProcessor,
}


def resolve_processors(
    refs: list[ProcessorRef],
) -> list[Processor]:
    """Convert a list of ``ProcessorRef`` values to instantiated processors.

    Raises ``NotImplementedError`` for unknown processor names.
    """
    from chatty.configs.persona import ProcessorWithArgs

    processors: list[Processor] = []
    for ref in refs:
        if isinstance(ref, str):
            cls = _KNOWN_PROCESSORS.get(ref)
            if cls is None:
                raise NotImplementedError(
                    f"Processor '{ref}' is not supported."
                )
            processors.append(cls())
        elif isinstance(ref, ProcessorWithArgs):
            cls = _KNOWN_PROCESSORS.get(ref.name)
            if cls is None:
                raise NotImplementedError(
                    f"Processor '{ref.name}' is not supported."
                )
            kwargs: dict = {}
            if ref.max_length is not None:
                kwargs["max_length"] = ref.max_length
            processors.append(cls(**kwargs))
        else:
            raise TypeError(
                f"Invalid processor reference type: {type(ref)}"
            )
    return processors


def with_processors(*processors: Processor):
    """Decorator to wrap a tool with content processors."""

    def decorator(tool_class):  # type: ignore[no-untyped-def]
        original_run = tool_class._run
        original_arun = tool_class._arun

        def _run_with_processors(
            self,
            *args,
            **kwargs,
        ) -> str:
            result = original_run(self, *args, **kwargs)
            for processor in processors:
                result = processor.process(result)
            return result

        async def _arun_with_processors(
            self,
            *args,
            **kwargs,
        ) -> str:
            result = await original_arun(self, *args, **kwargs)
            for processor in processors:
                result = processor.process(result)
            return result

        tool_class._run = _run_with_processors
        tool_class._arun = _arun_with_processors
        return tool_class

    return decorator
