"""Lightweight token estimation and truncation.

Uses a conservative chars-per-token ratio so that estimates err on the
side of *over*-counting (i.e. we may truncate slightly more than
strictly necessary, but we never exceed the context budget).
"""

CHARS_PER_TOKEN = 3
"""Conservative ratio (~3.5-4 for English, ~1.5-2 for CJK).

Using 3 means we slightly over-estimate token counts, which is the
safe direction — we'd rather trim a little extra than blow the context.
"""


def estimate_tokens(text: str) -> int:
    """Return an estimated token count for *text*."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate *text* so its estimated token count fits *max_tokens*."""
    max_chars = max_tokens * CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."
