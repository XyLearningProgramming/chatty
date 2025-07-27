# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chatty is a persona-driven chatbot that impersonates professional personas using a multi agent pipeline. The system is designed as a stateless, scalable service that responds to queries based on predefined knowledge bases while maintaining conversation context.

## Commands

### Linting and Code Quality
- **Linting**: `uv run ruff check src/` - Run linting checks using Ruff
- **Type checking**: `uv run mypy src/chatty` - Run type checking with MyPy  
- **Code formatting**: `uv run ruff format src/` - Format code with Ruff
- **Testing**: `uv run pytest tests/` - Run test suite with pytest

### Development
- **Run application**: `uv run python src/chatty/app.py` - Start the main application
- **Install dependencies**: `uv sync` - Install project dependencies using uv

## Architecture

### Core Components
The system follows a multi-agent architecture with three main agents:

1. **Relevancy Check Agent** - Determines if queries are relevant to the persona
2. **Knowledge Agent (ReAct)** - Uses tools to gather context from various sources
3. **Synthesis Agent** - Synthesizes information into coherent responses

### Directory Structure
```
src/chatty/
├── app.py              # Main FastAPI application entry point
├── api/                # HTTP API layer
│   ├── chat.py         # Chat endpoint implementation
│   └── models.py       # Pydantic request/response models
├── core/               # Business logic
│   ├── knowledge.py    # Knowledge Agent with retrieval tools
│   ├── synthesis.py    # Synthesis Agent implementation
│   ├── memory.py       # Semantic caching logic (admission, eviction, similarity)
│   └── generation.py   # Text generation and token processing
└── infra/              # Infrastructure abstractions
    └── vector_db.py    # Vector database operations
```

### Key Technologies
- **Backend**: FastAPI with Python 3.13+
- **Vector Database**: PostgreSQL with pgvector extension
- **Caching**: Redis for semantic caching
- **Model**: Qwen3 0.6B (production), Qwen2-0.5B-Instruct (CI/testing)
- **Framework**: LangChain for RAG pipeline

### API Design
- **Endpoint**: `POST /api/v1/chat`
- **Response**: Server-Sent Events (SSE) stream with JSON objects
- **Event Types**: `token` (streaming text), `structured_data` (JSON), `end_of_stream`

### Memory Strategy
Two-tier semantic memory system managed by core/memory.py:
- **Tier 1**: Golden questions (static, permanent memory)
- **Tier 2**: Auto-discovered questions (dynamic, with TTL and LFU eviction)

Business logic includes:
- Admission policy: Remember after 3+ similar queries
- Similarity threshold: Cosine similarity > 0.95 for memory hits
- Frequency tracking and LFU eviction policies

### Configuration
External YAML configuration files for:
- `author.yaml`: System prompt, persona details, knowledge source URIs, all as tools
- `config.yaml`: Service endpoints, database connections
Use `pydantic_settings` to read config.
Use fastapi dependency to inject settings, service objects. Example of dependency injection:
```python
def get_settings() -> Settings:
    pass

async def handler(settings: Annotated[Settings,Depends(get_settings)]):
    pass
```


### Testing Strategy
- **Golden Dataset**: Curated Q&A pairs for regression testing
- **CI Testing**: Uses lightweight Qwen2-0.5B-Instruct model with caching
- **Quality Gates**: Semantic similarity and keyword matching against golden answers
- **Test Standards**: Tests should be simple, but covering most codes, with less strict condition checks; do not extensively check functions from third-party.

## Development Notes

### Code Style
- Line length: 88 characters (Ruff configuration)
- Exclude tests from Ruff linting
- Use type hints consistently
- Follow existing patterns in the codebase
- Use python3.13 compatible type formatting, eg. `list | None` instead of `Optional[List]`

### Resource Constraints
The system is designed for minimal resource usage:
- Chatbot service: ~0.5 core, ~400 MiB
- Model server: ~3 cores, ~800 MiB (existing)
- Vector DB: ~0.5 core, ~500 MiB (shared)
- Redis: ~0.25 core, ~200 MiB (shared)

### State Management
- Stateless service design (no user sessions stored)
- Conversation context via sliding window approach
- No PII storage for privacy compliance