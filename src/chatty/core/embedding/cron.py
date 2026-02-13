"""Background cron that opportunistically embeds persona sections.

Runs as an ``asyncio.Task`` started in the FastAPI lifespan.  Each tick:

1. Reads ``persona.sections`` from the (hot-reloadable) config.
2. For ``source_url`` sections, fetches and caches the resolved content.
3. For all sections with non-empty content, checks the DB for missing
   embeddings.
4. For each missing embedding, races for the ``ModelSemaphore`` via
   ``try_embed`` (instant-timeout).  If the slot is busy, the section
   is skipped until the next tick.
"""

import asyncio
import logging

import httpx

from chatty.configs.config import get_app_config
from chatty.configs.persona import KnowledgeSection
from chatty.core.service.tools.processors import HtmlHeadTitleMeta, Processor
from chatty.core.service.tools.url_tool import _extract_text_from_pdf, _PDF_CONTENT_TYPES

from .client import EmbeddingClient
from .repository import text_hash

logger = logging.getLogger(__name__)

# Registry of known processors (mirrors ToolRegistry)
_KNOWN_PROCESSORS: dict[str, type[Processor]] = {
    HtmlHeadTitleMeta.processor_name: HtmlHeadTitleMeta,
}

# In-memory cache of resolved source_url content keyed by (title, url)
_resolved_content: dict[tuple[str, str], str] = {}


async def _fetch_section_content(section: KnowledgeSection) -> str | None:
    """Fetch and post-process content from a section's ``source_url``.

    Returns ``None`` on failure (logged, not raised).
    """
    url = section.source_url
    max_length = section.args.get("max_content_length", 8000)
    timeout = section.args.get("timeout", 30)

    try:
        async with httpx.AsyncClient(timeout=float(timeout)) as client:
            response = await client.get(url)
            response.raise_for_status()

            ct = response.headers.get("content-type", "")
            if any(pdf in ct for pdf in _PDF_CONTENT_TYPES):
                content = _extract_text_from_pdf(response.content)
            else:
                content = response.text

            # Truncate
            if len(content) > max_length:
                content = content[:max_length]

            # Apply processors
            for proc_name in section.processors:
                proc_cls = _KNOWN_PROCESSORS.get(proc_name)
                if proc_cls:
                    content = proc_cls().process(content)

            return content
    except Exception:
        logger.warning(
            "Cron: failed to fetch content for section '%s' from %s",
            section.title,
            url,
            exc_info=True,
        )
        return None


def _section_text(section: KnowledgeSection) -> str | None:
    """Return the embeddable text for a section (resolved or inline)."""
    if section.source_url:
        key = (section.title, section.source_url)
        return _resolved_content.get(key)
    return section.content or None


async def embedding_cron_loop(embedding_client: EmbeddingClient) -> None:
    """Run the embedding cron indefinitely.

    Should be started as an ``asyncio.Task`` and cancelled on shutdown.
    """
    while True:
        try:
            config = get_app_config()
            interval = config.rag.cron_interval
            sections = config.persona.sections

            # Step 1: Resolve source_url content (cached in-memory)
            for section in sections:
                if not section.source_url:
                    continue
                key = (section.title, section.source_url)
                if key not in _resolved_content:
                    content = await _fetch_section_content(section)
                    if content:
                        _resolved_content[key] = content
                        logger.info(
                            "Cron: resolved content for section '%s' "
                            "(%d chars)",
                            section.title,
                            len(content),
                        )

            # Step 2: Find sections that need embedding
            texts_to_embed: list[tuple[KnowledgeSection, str]] = []
            for section in sections:
                text = _section_text(section)
                if not text:
                    continue
                cached = await embedding_client.get_cached(text)
                if cached is None:
                    texts_to_embed.append((section, text))

            if texts_to_embed:
                logger.info(
                    "Cron: %d section(s) need embedding", len(texts_to_embed)
                )

            # Step 3: Try to embed each missing section
            for section, text in texts_to_embed:
                result = await embedding_client.try_embed(text)
                if result is not None:
                    logger.info(
                        "Cron: embedded section '%s' (%d dims)",
                        section.title,
                        len(result),
                    )
                else:
                    logger.debug(
                        "Cron: semaphore busy, skipping section '%s'",
                        section.title,
                    )

        except asyncio.CancelledError:
            logger.info("Embedding cron cancelled, shutting down.")
            return
        except Exception:
            logger.exception("Embedding cron tick failed")

        await asyncio.sleep(interval)
