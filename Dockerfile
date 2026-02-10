FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies first (cached layer)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy source and config, then install the project itself
COPY src/ src/
COPY configs/ configs/
RUN uv sync --frozen --no-dev

# --- Runtime stage ---
FROM python:3.13-slim

WORKDIR /app

# Copy the virtual environment and config from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/configs /app/configs

ENV PATH="/app/.venv/bin:$PATH"
ENV PORT=8080
ENV HOST=0.0.0.0

EXPOSE 8080

CMD ["uvicorn", "chatty.app:app", "--host", "0.0.0.0", "--port", "8080"]
