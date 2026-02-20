"""Real client IP extraction for reverse-proxy deployments.

Deployment chain: Client -> Cloudflare -> Traefik -> FastAPI.

``request.client.host`` is always Traefik's pod IP, so we extract the
real IP from proxy headers in priority order:

1. ``CF-Connecting-IP`` — set by Cloudflare, unspoofable
2. ``X-Real-IP`` — set by some proxy configs
3. ``X-Forwarded-For`` — leftmost entry (trusted because Traefik only
   accepts forwarded headers from Cloudflare IPs)
4. ``request.client.host`` — last resort (local dev / direct access)
"""

from __future__ import annotations

from fastapi import Request

_REAL_IP_HEADER_NAMES = [
    "cf-connecting-ip",
    "x-real-ip",
    "x-forwarded-for",
]

_DEFAULT_UNKNOWN_IP = "unknown"


def get_real_ip(request: Request) -> str:
    """Extract the real client IP from the request.

    Usable as a FastAPI dependency::

        real_ip: str = Depends(get_real_ip)
    """
    for header in _REAL_IP_HEADER_NAMES:
        value = request.headers.get(header)
        if value:
            return value.split(",")[0].strip()

    if request.client:
        return request.client.host

    return _DEFAULT_UNKNOWN_IP
