
.PHONY: help install test test-unit test-e2e test-golden lint format typecheck check dev clean

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
	uv run uvicorn chatty.app:app --host 0.0.0.0 --port 8080 --reload

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/
	rm -rf src/*.egg-info/
