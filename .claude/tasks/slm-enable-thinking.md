# SLM Server: enable_thinking Hard Switch

**Status:** Planned — not yet implemented
**Created:** 2026-02-20

## Goal

Add `enable_thinking` support to `slm_server` so the Qwen3 chat
template's hard switch actually fires. Currently `llama-cpp-python`
v0.3.13's `Jinja2ChatFormatter` never passes `enable_thinking` into
the Jinja2 render context, so the template block:

```jinja2
{%- if enable_thinking is defined and enable_thinking is false %}
    {{- '<think>\n\n</think>\n\n' }}
{%- endif %}
```

...is dead code. The `/no_think` soft switch is unreliable on small
models (Qwen3-0.6B). The hard switch forces an empty think block
before generation, which is deterministic.

---

## Problem Chain

```
chatty NoThinkChatModel
  → sends "hello /no_think" to slm_server
    → llama-cpp-python Jinja2ChatFormatter renders template
      → enable_thinking is undefined (never passed)
        → model sees /no_think in user text but ignores it (0.6B)
          → <think> block emitted anyway
```

---

## Changes in slm-server

### 1. model.py — Add enable_thinking to ChatCompletionRequest

```python
enable_thinking: bool | None = Field(
    default=None,
    description="Hard switch for model thinking. "
    "When false, injects an empty <think> block via the chat template. "
    "None (default) preserves the model's natural behavior.",
    exclude=True,  # Not a llama-cpp-python parameter
)
```

`exclude=True` ensures `req.model_dump()` (which is spread into
`llm.create_chat_completion(**req.model_dump())`) does not pass the
unknown kwarg to llama-cpp.

### 2. app.py — Patch the formatter before calling the LLM

Before `llm.create_chat_completion`, if `req.enable_thinking is not
None`, temporarily override the Llama instance's chat handler with
one whose Jinja2 render context includes `enable_thinking`.

The cleanest approach: subclass `Jinja2ChatFormatter` to accept and
forward extra template kwargs.

```python
from llama_cpp.llama_chat_format import Jinja2ChatFormatter

class ThinkingAwareChatFormatter(Jinja2ChatFormatter):
    """Extends Jinja2ChatFormatter to pass enable_thinking to the template."""

    def __init__(self, base: Jinja2ChatFormatter, enable_thinking: bool):
        # Reuse the already-compiled template from the base formatter
        self._environment = base._environment
        self.eos_token = base.eos_token
        self.bos_token = base.bos_token
        self.add_generation_prompt = base.add_generation_prompt
        self.stop_token_ids = base.stop_token_ids
        self._enable_thinking = enable_thinking

    def __call__(self, *, messages, **kwargs):
        # Render with enable_thinking in the Jinja2 context
        ...
```

Then in `run_llm_streaming` / `run_llm_non_streaming`, pop
`enable_thinking` from the request before the LLM call and set a
patched chat handler on the Llama instance for the duration of the
call.

### 3. Alternative: context-manager approach

Since `max_concurrency=1`, there is no race on `llm.chat_handler`.
A simple context manager can swap the handler and restore it:

```python
@contextmanager
def override_thinking(llm, enable_thinking):
    if enable_thinking is None:
        yield
        return
    original = llm.chat_handler
    # Build patched handler from the default template handler
    ...
    llm.chat_handler = patched
    try:
        yield
    finally:
        llm.chat_handler = original
```

---

## Changes in chatty

### 1. no_think.py — Switch from prompt suffix to extra_body

Replace the `_inject` method (which appends ` /no_think` to the
user message) with `model_kwargs` that sends
`enable_thinking: false` in the OpenAI API request body.

`ChatOpenAI` (via the openai SDK) forwards unknown body fields
through `extra_body`, which FastAPI's `ChatCompletionRequest` will
now accept.

---

## Files Changed

| Repo | File | Change |
|------|------|--------|
| slm_server | slm_server/model.py | Add `enable_thinking` field (excluded from model_dump) |
| slm_server | slm_server/app.py | Subclass formatter, wire enable_thinking through to Jinja2 render |
| chatty | src/chatty/core/llm/no_think.py | Send `enable_thinking=false` via extra_body instead of appending suffix |
