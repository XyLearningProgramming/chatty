# Chatty

A persona-driven chatbot API that impersonates you (or anyone) professionally with rich context, tools, and streaming responses.

## Configuration

### Non-secrets — `configs/config.yaml`

Tune LLM parameters, API settings, cache behaviour, persona, and tool definitions here.
This file is baked into the Docker image and optionally overridden at runtime via a Kubernetes ConfigMap.

### Secrets — `.env`

Copy the example and fill in real values:

```bash
cp .env.example .env
```

The app reads env vars with the `CHATTY_` prefix and `__` nested delimiter (via pydantic-settings).
See `.env.example` for all available variables (LLM endpoint/key, Postgres URI, Redis URI, etc.).

## Local Development

### 1. Prerequisites

- Python ≥ 3.13
- [uv](https://docs.astral.sh/uv/)
- Docker (for Postgres & Redis)

### 2. Start infrastructure

```bash
docker compose -f deploy/docker/docker-compose.dev.yaml up -d
```

This pulls up **PostgreSQL** (with pgvector) on port `5432` and **Redis** on port `6379`, accessible via `localhost` or `host.docker.internal`.

### 3. Install dependencies

```bash
make install        # or: uv sync
```

### 4. Run the dev server

```bash
make dev            # or: uv run uvicorn chatty.app:app --host 0.0.0.0 --port 8080 --reload
```

The API is then available at `http://localhost:8080`. Health check: `GET /health`.

### 5. Run tests & checks

```bash
make test           # unit + e2e
make check          # lint + typecheck
```

## Deployment

Production deployment uses Helmfile to a Kubernetes cluster.
See [`deploy/helmfile/README.md`](deploy/helmfile/README.md) for full instructions on required secrets, helmfile commands, and helpers.
