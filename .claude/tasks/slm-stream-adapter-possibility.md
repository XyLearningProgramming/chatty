# SLM-Server Stream Adapter Possibility

**Status:** Reference — implement only if ChatOpenAI does not pass through slm-server fields  
**Created:** 2026-02-23  
**Related:** Plan "Align stream with slm-server"; upstream `../slm-server` schema (reasoning_content, full tool_calls).

## Goal

Document when and why we might add an **adapter** between slm-server’s SSE stream and chatty’s stream processor, so that thinking and tool calls are reliably passed out even if the default LLM client does not expose the new schema.

## Context

- **slm-server** now emits:
  - `delta.reasoning_content` for thinking (no `<think>` in content when postprocessor is used).
  - Full `delta.tool_calls` (index, id, type, function.name, function.arguments) per chunk.
- **Chatty** uses LangChain’s `ChatOpenAI` (with slm-server’s base_url) and maps `AIMessageChunk` → domain `StreamEvent`s in `stream.py`.

## When an Adapter Might Be Needed

1. **Reasoning not visible**
   - LangChain’s `ChatOpenAI` targets the official OpenAI API. Third-party fields like `reasoning_content` are **not** guaranteed to be mapped into chunks; they may be dropped or appear in `additional_kwargs` depending on client version.
   - If we never see `reasoning_content` (or equivalent) on chunks after implementing “prefer reasoning_content” in `stream.py`, thinking will not be “properly passed out” when using slm-server with thinking enabled.

2. **Tool calls shape or timing**
   - Less likely: if the client always sends partial/incremental tool_call_chunks and never exposes full tool_calls in stream, we might need to merge by `index` ourselves or consume a different source (e.g. only from the accumulated message at end of stream). Current plan already handles “iterate all tool_call_chunks” and full-argument parsing; if the client merges upstream’s full deltas into one chunk per tool, that should work without an adapter.

## Adapter Options (if we need one)

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **A) Custom HTTP stream → AIMessageChunk-like** | A thin client that reads slm-server SSE, parses `delta.reasoning_content`, `delta.content`, `delta.tool_calls`, and builds objects that look like `AIMessageChunk` (with e.g. `reasoning_content` in `additional_kwargs` and correct `tool_call_chunks`). Feed these into existing `map_llm_stream`. | Reuses existing stream mapper and one_step loop; minimal API/CLI change. | Need to maintain SSE parsing and chunk construction; must match LangChain’s accumulation semantics for tool_calls. |
| **B) Custom HTTP stream → StreamEvents** | Parse slm-server SSE and emit domain `StreamEvent`s (ThinkingEvent, ContentEvent, ToolCallEvent) directly. Bypass LangChain for the stream path; still call the model for the actual HTTP request but use our own stream parser for the response body. | Full control; no dependency on client’s chunk shape. | Larger refactor: one_step (or a new “stream-only” path) would consume events + still need accumulated message for tool execution (e.g. request a non-streaming call for tool rounds, or accumulate from our parsed stream). |
| **C) Custom BaseChatModel** | Subclass or wrap so that `_astream` uses our own HTTP/SSE handling and yields `ChatGenerationChunk` with `reasoning_content` and tool_call data set as we need. | Fits into existing LangChain pipeline; gated/no_think wrappers unchanged. | Same parsing complexity as A; more surface (custom model class) to maintain. |
| **D) Callback that sees raw response** | If LangChain’s client exposes a hook that receives raw SSE lines or response chunks, we could attach a callback that pushes reasoning/tool_calls into a side channel (e.g. a queue) that `map_llm_stream` reads. | No replacement of the HTTP client. | Depends on LangChain offering such a hook; may be fragile across versions. |

## Recommendation

- **First:** Implement stream processor and normalization as in the plan; test against slm-server. If `reasoning_content` appears on chunks (e.g. in `additional_kwargs`), no adapter is needed for thinking.
- **If reasoning is missing:** Prefer **Option A** (custom HTTP stream building AIMessageChunk-like objects with `reasoning_content` and correct tool_call data). It keeps the rest of the stack unchanged and localizes SSE knowledge in one place. Option C is equivalent in effect; choose A if we want to avoid a custom model class, C if we want to keep “all LLM I/O” behind a LangChain model interface.
- **Avoid** duplicating slm-server’s postprocessor logic (e.g. <think> / &lt;tool_call&gt; parsing) in chatty; the adapter should consume the **postprocessed** stream (reasoning_content, structured tool_calls) and only map it into the shape our stream processor expects.

## Where to Implement

- **Option A:** New module under `src/chatty/core/llm/` or `src/chatty/infra/llm/` that implements an async generator `(request) → AsyncIterator[AIMessageChunk]` using httpx (or aiohttp) and the same base_url as the current LLM config. The existing `get_llm` or a new factory would return a small “StreamAdapterChatModel” that uses this generator in `_astream` and delegates non-stream calls to the original `ChatOpenAI`.
- **Option B:** New stream-only path in the service layer that takes the HTTP response stream and produces `AsyncIterator[StreamEvent]`; one_step would need to obtain the accumulated message for tool rounds via a separate non-streaming call or by accumulating from our parsed stream.
- **Option C:** New class in `src/chatty/core/llm/` (e.g. `SlmStreamChatModel(BaseChatModel)`) that in `_astream` does HTTP + SSE parsing and yields `ChatGenerationChunk` with the right content, additional_kwargs, and tool_call_chunks.

No code in this task file; this document is for reasoning and future implementation only.
