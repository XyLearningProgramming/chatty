
.PHONY: help install test test-unit test-e2e test-golden lint format typecheck check dev cli db-upgrade dev-up clean

# Project root (directory containing this Makefile). Ensures .env is loaded from repo root
# even when make is invoked from a subdirectory.
ROOT := $(dir $(realpath $(firstword $(MAKEFILE_LIST))))

# Default target
help:
	@echo "Available targets:"
	@echo "  install     - Install dependencies"
	@echo "  test        - Run all tests"
	@echo "  test-unit   - Run unit tests only"
	@echo "  test-e2e    - Run end-to-end tests"
	@echo "  test-golden - Run golden dataset tests"
	@echo "  lint        - Run linting checks"
	@echo "  format      - Format code"
	@echo "  typecheck   - Run type checking"
	@echo "  check       - Run all code quality checks (lint + typecheck)"
	@echo "  dev         - Start development server with reload"
	@echo "  cli         - Start interactive CLI (cli/)"
	@echo "  db-upgrade  - Run Alembic migrations to head"
	@echo "  dev-up      - Start dev stack (deploy/docker/docker-compose.dev.yaml) in detached mode"
	@echo "  clean       - Clean up generated files"

# Installation
install:
	uv sync

# Testing
test: test-unit test-e2e

test-unit:
	uv run pytest tests/ -v --ignore=tests/e2e/

test-e2e:
	uv run pytest tests/e2e/ -v

test-golden:
	uv run pytest tests/e2e/test_streaming_agent.py::TestStreamingAgent::test_relevance_filtering_golden_cases -v -s

# Code quality
lint:
	uv run ruff check src/

format:
	uv run ruff format src/

typecheck:
	uv run mypy src/chatty

check: lint typecheck

# Development
dev:
	uv run --env-file $(ROOT).env uvicorn chatty.app:app --host 0.0.0.0 --port 8080 --reload

cli:
	cd $(ROOT) && uv run --env-file $(ROOT).env python -m cli

# Database
db-upgrade:
	uv run --env-file $(ROOT).env alembic upgrade head

# Docker dev stack
dev-up:
	docker compose -f $(ROOT)deploy/docker/docker-compose.dev.yaml up -d

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/
	rm -rf src/*.egg-info/
