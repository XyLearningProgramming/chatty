"""Singleton async Redis client.

``Redis.from_url()`` is synchronous (no TCP connection until the first
command), so it fits the regular ``@singleton`` decorator.  The caller
is responsible for verifying the connection (e.g. ``await client.ping()``)
and for closing it on shutdown (``await client.aclose()``).
"""

from redis.asyncio import Redis

from chatty.configs.config import get_app_config
from chatty.infra.singleton import singleton


@singleton
def get_redis_client() -> Redis:
    """Create and cache a ``Redis`` client from application config.

    The client is **not** connected yet — the first real command
    triggers the TCP handshake.  Call ``await client.ping()`` to
    verify reachability.
    """
    uri = get_app_config().third_party.redis_uri
    return Redis.from_url(uri, decode_responses=True)
