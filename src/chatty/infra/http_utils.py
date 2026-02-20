"""HTTP utilities with PDF support.

Pure infra — no domain imports.
"""

from __future__ import annotations

import httpx

PDF_CONTENT_TYPES = {"application/pdf", "application/x-pdf"}
_PDF_FILETYPE = "pdf"
_CONTENT_TYPE_HEADER = "content-type"


def extract_text_from_pdf(data: bytes) -> str:
    """Extract readable text from raw PDF bytes using pymupdf."""
    import pymupdf  # lazy — only needed for PDF responses

    parts: list[str] = []
    with pymupdf.open(stream=data, filetype=_PDF_FILETYPE) as doc:
        for page in doc:
            parts.append(page.get_text())
    return "\n".join(parts)


def _read_response(response: httpx.Response) -> str:
    ct = response.headers.get(_CONTENT_TYPE_HEADER, "")
    if any(pdf in ct for pdf in PDF_CONTENT_TYPES):
        return extract_text_from_pdf(response.content)
    return response.text


class HttpClient:
    """Async HTTP GET with PDF support."""

    @staticmethod
    async def get(url: str, timeout: float) -> str:
        """Fetch *url* and return text (auto-extracts PDF)."""
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            return _read_response(response)
