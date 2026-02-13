"""Embedding infrastructure -- client, repository, and background cron."""

from .client import EmbeddingClient
from .repository import EmbeddingRepository

__all__ = [
    "EmbeddingClient",
    "EmbeddingRepository",
]
