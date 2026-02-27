"""ReasoningChatOpenAI — ChatOpenAI subclass that preserves ``reasoning_content``.

Why this exists
---------------
Some OpenAI-compatible model servers (e.g. our slm-server with Qwen3)
emit a **custom** ``reasoning_content`` field inside the streaming
``delta`` object::

    data: {"choices": [{"delta": {"reasoning_content": "Let me think…"}}]}

This field is *not* part of the official OpenAI Chat Completions spec,
so ``langchain-openai``'s ``_convert_delta_to_message_chunk`` silently
drops it — only ``content``, ``function_call``, ``tool_calls``, and
``role`` are extracted from the delta dict.

The OpenAI Python SDK *does* keep the field (``ChoiceDelta`` uses
``extra="allow"``), and ``model_dump()`` includes it, but
``langchain-openai`` never reads it.

This subclass overrides a single method —
``_convert_chunk_to_generation_chunk`` — to rescue
``reasoning_content`` from the raw chunk dict and inject it into
``AIMessageChunk.additional_kwargs["reasoning_content"]``, where the
downstream ``_reasoning_from_chunk`` helper already looks for it.

Compatibility
-------------
* When the upstream server is standard OpenAI (no ``reasoning_content``
  in deltas), the override is a no-op — ``additional_kwargs`` is left
  untouched and behaviour is identical to vanilla ``ChatOpenAI``.
* All other ``ChatOpenAI`` features (tool calling, structured output,
  response headers, Responses API, etc.) are inherited as-is.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessageChunk
from langchain_openai import ChatOpenAI

# -- OpenAI Chat Completions chunk keys ------------------------------------
_KEY_CHOICES = "choices"
_KEY_DELTA = "delta"

# -- Custom delta field emitted by slm-server (Qwen3, DeepSeek, etc.) -----
_KEY_REASONING_CONTENT = "reasoning_content"


class ReasoningChatOpenAI(ChatOpenAI):
    """``ChatOpenAI`` that propagates ``delta.reasoning_content`` to LangChain messages.

    Drop-in replacement for ``ChatOpenAI``.  The only behavioural
    difference is that streaming chunks whose delta carries
    ``reasoning_content`` will have it stored in
    ``AIMessageChunk.additional_kwargs["reasoning_content"]``.
    """

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> Any:  # returns ChatGenerationChunk | None
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if generation_chunk is None:
            return None

        # --- Rescue reasoning_content from the raw delta ---------------
        choices = chunk.get(_KEY_CHOICES) or []
        if choices:
            delta = choices[0].get(_KEY_DELTA) or {}
            reasoning = delta.get(_KEY_REASONING_CONTENT)
            if reasoning and isinstance(generation_chunk.message, AIMessageChunk):
                generation_chunk.message.additional_kwargs[_KEY_REASONING_CONTENT] = (
                    reasoning
                )

        return generation_chunk
