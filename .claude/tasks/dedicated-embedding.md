# Dedicated Embedding Model in SLM Server

**Status:** Planned — not yet implemented
**Created:** 2026-02-12

## Context

slm_server (../slm_server) currently uses a single Qwen3-0.6B-Q4_K_M (chat model)
for both chat completions AND embeddings via llama-cpp-python. This is suboptimal:

1. **Quality**: The chat model was never trained for embedding tasks. Qwen3-Embedding-0.6B
   scores 70.70 on MTEB English vs the chat model which is untrained for retrieval.
2. **KV cache thrashing**: Embedding calls wipe the LLM's KV cache, forcing full
   system-prompt re-prefill on the next chat completion. On 1 CPU this is slow.
3. **Semaphore contention**: Both operations share MAX_CONCURRENCY=1, so an embedding
   call blocks all chat completions.

## Constraints

- slm_server pod: 1 CPU, 1 GiB memory limit (200m CPU / 600 Mi request)
- Qwen3-0.6B-Q4_K_M already uses ~550-600 MB at runtime
- Cannot fit a second llama.cpp Llama instance (~350+ MB more)
- chatty pod: 200m CPU, 256 Mi memory limit

## Goal

Load a tiny dedicated embedding model (~22-33M params, ~90-130 MB) inside slm_server
to serve `/api/v1/embeddings`, keeping the Llama instance exclusively for chat.
This eliminates KV cache thrashing and improves embedding quality.

## Recommended Embedding Model

**all-MiniLM-L6-v2** via sentence-transformers (ONNX backend):
- 22M params, ~90 MB RAM, 384-dim vectors
- ~5ms per embedding on CPU
- MTEB English retrieval: ~49 (far better than untrained chat model extraction)
- Apache 2.0 license

Alternative: **BGE-small-en-v1.5** (33M params, ~130 MB, 384-dim, slightly better quality).

Pick whichever fits the memory budget. Profile actual RSS after loading both models
to confirm it stays under 1 GiB.

## Changes Required in slm_server

### 1. config.py — Add embedding model settings

Add an `EmbeddingSettings` nested model to `Settings`:

```python
class EmbeddingSettings(BaseModel):
    model_name: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model ID for the dedicated embedding model.",
    )
    device: str = Field("cpu", description="Device for embedding inference.")
    # If using ONNX backend:
    # backend: str = Field("onnx", description="Backend: 'onnx' or 'torch'.")

class Settings(BaseSettings):
    ...
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
```

### 2. app.py — Separate embedding model from Llama

a) Remove `embedding=True` from `get_llm()` — the Llama instance only does chat now.

b) Add a new singleton for the embedding model:

```python
from sentence_transformers import SentenceTransformer

def get_embedding_model(
    settings: Annotated[Settings, Depends(get_settings)],
) -> SentenceTransformer:
    if not hasattr(get_embedding_model, "_instance"):
        get_embedding_model._instance = SentenceTransformer(
            settings.embedding.model_name,
            device=settings.embedding.device,
        )
    return get_embedding_model._instance
```

c) Update the `/api/v1/embeddings` endpoint to use the dedicated model.
   The embedding model is tiny and fast (~5ms), so it does NOT need the
   llm semaphore — it can run concurrently with chat completions:

```python
@app.post("/api/v1/embeddings")
async def create_embeddings(
    req: EmbeddingRequest,
    model: Annotated[SentenceTransformer, Depends(get_embedding_model)],
    __: Annotated[None, Depends(raise_as_http_exception)],
):
    """Create embeddings using the dedicated embedding model."""
    with slm_embedding_span(req) as span:
        inputs = req.input if isinstance(req.input, list) else [req.input]
        vectors = await asyncio.to_thread(model.encode, inputs, normalize_embeddings=True)
        # Build OpenAI-compatible response
        data = [
            {"object": "embedding", "embedding": vec.tolist(), "index": i}
            for i, vec in enumerate(vectors)
        ]
        result = {
            "object": "list",
            "data": data,
            "model": req.model or model.get_modules()[0].auto_model.name_or_path,
            "usage": {"prompt_tokens": sum(len(t.split()) for t in inputs), "total_tokens": sum(len(t.split()) for t in inputs)},
        }
        set_attribute_response_embedding(span, result)
        return result
```

### 3. pyproject.toml — Add dependency

```toml
dependencies = [
    ...
    "sentence-transformers>=3.0.0",
]
```

Note: sentence-transformers pulls in torch (~200 MB in the Docker image).
To keep the image smaller, consider using the ONNX backend instead:

```toml
dependencies = [
    ...
    "sentence-transformers[onnx]>=3.0.0",
]
```

Then load with: `SentenceTransformer(model_name, backend="onnx")`.
The ONNX backend has a much smaller runtime footprint than full PyTorch.

### 4. scripts/download.sh — Pre-download the embedding model

Add a step to pre-download the embedding model into the Docker image or
init container, so the first request doesn't trigger a slow HuggingFace download:

```bash
# Download embedding model for offline use
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

Or in the Dockerfile:

```dockerfile
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

### 5. deploy/helm/values.yaml — Adjust resource limits

The embedding model adds ~90-130 MB RAM. Adjust:

```yaml
resources:
  limits:
    memory: 1Gi    # keep as-is, but monitor closely
  requests:
    memory: 750Mi  # bump from 600Mi to account for embedding model
```

### 6. chatty (this repo) — Update embedding config

After slm_server ships the dedicated model, update chatty's config to match
the new embedding dimensions:

- configs/config.yaml: `embedding.dimensions: 384` (was 1024)
- configs/config.yaml: `embedding.model_name: "all-MiniLM-L6-v2"`
- pgvector column width may need a migration if currently fixed at 1024

### 7. Tests in slm_server

- Update test_embedding.py: mock SentenceTransformer instead of Llama.create_embedding
- Update test_app.py: embedding endpoint no longer depends on llm or llm semaphore
- Add test: embedding requests don't block chat completions (no semaphore contention)
- Add test: verify OpenAI-compatible response format from new endpoint

## Memory Budget Estimate

| Component                        | RAM       |
|----------------------------------|-----------|
| Qwen3-0.6B Q4_K_M weights       | ~484 MB   |
| KV cache @ n_ctx=2048            | ~50-80 MB |
| all-MiniLM-L6-v2 (ONNX)         | ~90 MB    |
| Python + FastAPI + overhead      | ~60 MB    |
| **Total**                        | **~700-720 MB** |

Fits within the 1 GiB limit with ~280 MB headroom.

## Migration Path

1. Implement and ship the slm_server changes (separate PR in ../slm_server)
2. Deploy new slm_server with both models
3. Run chatty pgvector migration to change vector dimension from 1024 to 384
4. Update chatty embedding config (dimensions, model_name)
5. Deploy updated chatty

## Key Benefit Summary

- Embedding quality: untrained chat extraction → purpose-built retrieval model
- Embedding latency: ~seconds (0.6B forward pass) → ~5ms (22M ONNX model)
- Chat latency: KV cache stays warm between completions (system prompt prefix reuse)
- Concurrency: embeddings no longer block chat (separate model, no shared semaphore)
