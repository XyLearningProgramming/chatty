"""Unit tests for ReasoningChatOpenAI — reasoning_content preservation.

``chatty.core.llm.__init__`` re-exports FastAPI Depends helpers that
trigger a circular import at test-collection time.  The
``ReasoningChatOpenAI`` class itself only depends on langchain_openai
and langchain_core, so we load the source file directly to sidestep
the package ``__init__``.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessageChunk

# ---------------------------------------------------------------------------
# Direct-load reasoning.py (avoid chatty.core.llm.__init__ cycle)
# ---------------------------------------------------------------------------

_MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "chatty"
    / "core"
    / "llm"
    / "reasoning.py"
)
_spec = importlib.util.spec_from_file_location("_reasoning_standalone", _MODULE_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
ReasoningChatOpenAI = _mod.ReasoningChatOpenAI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(delta: dict, **top_level: object) -> dict:
    """Build a minimal ChatCompletionChunk-style dict."""
    return {
        "id": "chatcmpl_test",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": None,
                "logprobs": None,
            }
        ],
        **top_level,
    }


@pytest.fixture()
def llm() -> ReasoningChatOpenAI:
    return ReasoningChatOpenAI(api_key="test", model="test")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReasoningContentExtraction:
    """Verify reasoning_content flows from raw delta into AIMessageChunk."""

    def test_reasoning_content_injected_into_additional_kwargs(
        self, llm: ReasoningChatOpenAI
    ):
        chunk = _make_chunk(
            {"role": "assistant", "reasoning_content": "Let me think about this."}
        )
        gen = llm._convert_chunk_to_generation_chunk(chunk, AIMessageChunk, {})

        assert gen is not None
        assert isinstance(gen.message, AIMessageChunk)
        assert gen.message.additional_kwargs["reasoning_content"] == (
            "Let me think about this."
        )

    def test_content_still_extracted_normally(self, llm: ReasoningChatOpenAI):
        chunk = _make_chunk({"role": "assistant", "content": "Hello!"})
        gen = llm._convert_chunk_to_generation_chunk(chunk, AIMessageChunk, {})

        assert gen is not None
        assert gen.message.content == "Hello!"
        # No reasoning_content -> key should be absent
        assert "reasoning_content" not in gen.message.additional_kwargs

    def test_both_reasoning_and_content(self, llm: ReasoningChatOpenAI):
        chunk = _make_chunk(
            {
                "role": "assistant",
                "content": "The answer is 4.",
                "reasoning_content": "2+2=4",
            }
        )
        gen = llm._convert_chunk_to_generation_chunk(chunk, AIMessageChunk, {})

        assert gen is not None
        assert gen.message.content == "The answer is 4."
        assert gen.message.additional_kwargs["reasoning_content"] == "2+2=4"

    def test_empty_reasoning_content_not_injected(self, llm: ReasoningChatOpenAI):
        chunk = _make_chunk(
            {"role": "assistant", "content": "hi", "reasoning_content": ""}
        )
        gen = llm._convert_chunk_to_generation_chunk(chunk, AIMessageChunk, {})

        assert gen is not None
        assert "reasoning_content" not in gen.message.additional_kwargs

    def test_null_reasoning_content_not_injected(self, llm: ReasoningChatOpenAI):
        chunk = _make_chunk(
            {"role": "assistant", "content": "hi", "reasoning_content": None}
        )
        gen = llm._convert_chunk_to_generation_chunk(chunk, AIMessageChunk, {})

        assert gen is not None
        assert "reasoning_content" not in gen.message.additional_kwargs

    def test_no_choices_returns_generation_chunk(self, llm: ReasoningChatOpenAI):
        """Empty choices list should still return a valid chunk (usage-only frame)."""
        chunk = {
            "id": "chatcmpl_test",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "test-model",
            "choices": [],
        }
        gen = llm._convert_chunk_to_generation_chunk(chunk, AIMessageChunk, {})
        # Should not crash; may return a chunk with empty content
        assert gen is not None

    def test_tool_calls_unaffected(self, llm: ReasoningChatOpenAI):
        """Tool call deltas pass through unchanged."""
        chunk = _make_chunk(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    }
                ],
            }
        )
        gen = llm._convert_chunk_to_generation_chunk(chunk, AIMessageChunk, {})

        assert gen is not None
        assert gen.message.tool_call_chunks
        assert gen.message.tool_call_chunks[0]["name"] == "get_weather"
        assert "reasoning_content" not in gen.message.additional_kwargs

    def test_standard_openai_chunk_without_reasoning_field(
        self, llm: ReasoningChatOpenAI
    ):
        """Standard OpenAI chunk (no reasoning_content key at all) works fine."""
        chunk = _make_chunk({"role": "assistant", "content": "Sure!"})
        assert "reasoning_content" not in chunk["choices"][0]["delta"]

        gen = llm._convert_chunk_to_generation_chunk(chunk, AIMessageChunk, {})
        assert gen is not None
        assert gen.message.content == "Sure!"
        assert "reasoning_content" not in gen.message.additional_kwargs
