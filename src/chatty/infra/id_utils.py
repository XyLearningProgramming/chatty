"""Prefixed ID generation for the Chatty application.

All public-facing IDs use a ``{prefix}_{random}`` format so that any
ID can be visually identified by its origin:

- ``conv_a8Kx3nQ9mP2r``  — conversation
- ``trace_L7wBd4Fj9Ks2``  — single agent request
- ``msg_kJ3pW7mD4bNx``   — message (human / system / tool)

Provider-generated IDs (OpenAI ``chatcmpl-xxx``, ``call_xxx``) are
used as-is for AI messages and tool calls because they are already
prefixed.
"""

import secrets
import string

_ALPHABET = string.ascii_letters + string.digits  # a-z A-Z 0-9
_DEFAULT_LENGTH = 12  # ~71 bits of entropy


def generate_id(prefix: str, length: int = _DEFAULT_LENGTH) -> str:
    """Generate a prefixed random ID.

    Args:
        prefix: Short descriptor (e.g. ``"conv"``, ``"trace"``, ``"msg"``).
        length: Number of random alphanumeric characters after the prefix.

    Returns:
        ``"{prefix}_{random}"`` string.
    """
    suffix = "".join(secrets.choice(_ALPHABET) for _ in range(length))
    return f"{prefix}_{suffix}"
