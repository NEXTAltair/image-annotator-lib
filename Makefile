# Image Annotator Lib Makefile
# Development task automation

.PHONY: help test lint format install install-dev clean run-example typecheck test-unit test-integration test-webapi test-scorer test-tagger test-cov setup

# Default target
help:
	@echo "Image Annotator Lib - Available Commands:"
	@echo ""
	@echo "Development:"
	@echo "  setup        Setup cross-platform development environment"
	@echo "  install      Install project dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo "  run-example  Run example script"
	@echo "  test         Run all tests"
	@echo "  test-unit    Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-webapi  Run Web API tests only"
	@echo "  test-scorer  Run scorer model tests only"
	@echo "  test-tagger  Run tagger model tests only"
	@echo "  test-cov     Run tests with coverage report"
	@echo "  lint         Run code linting (ruff)"
	@echo "  format       Format code (ruff format)"
	@echo "  typecheck    Run type checking (mypy)"
	@echo "  clean        Clean build artifacts"

# Setup target
setup:
	@echo "Setting up development environment..."
	./scripts/setup.sh

# Development targets
install:
	@echo "Installing project dependencies..."
	UV_PROJECT_ENVIRONMENT=.venv_linux uv sync

install-dev:
	@echo "Installing development dependencies..."
	UV_PROJECT_ENVIRONMENT=.venv_linux uv sync --dev

run-example:
	@echo "Running example script..."
	UV_PROJECT_ENVIRONMENT=.venv_linux uv run python example/example_lib.py

# Testing targets
test:
	@echo "Running all tests..."
	UV_PROJECT_ENVIRONMENT=.venv_linux uv run pytest -n auto

test-unit:
	@echo "Running unit tests..."
	UV_PROJECT_ENVIRONMENT=.venv_linux uv run pytest -m unit -n auto

test-integration:
	@echo "Running integration tests..."
	UV_PROJECT_ENVIRONMENT=.venv_linux uv run pytest -m integration

test-webapi:
	@echo "Running Web API tests..."
	UV_PROJECT_ENVIRONMENT=.venv_linux uv run pytest -m webapi

test-scorer:
	@echo "Running scorer model tests..."
	UV_PROJECT_ENVIRONMENT=.venv_linux uv run pytest -m scorer

test-tagger:
	@echo "Running tagger model tests..."
	UV_PROJECT_ENVIRONMENT=.venv_linux uv run pytest -m tagger



test-cov:
	@echo "Running tests with coverage..."
	UV_PROJECT_ENVIRONMENT=.venv_linux uv run pytest --cov=src --cov-report=html --cov-report=xml

# Code quality targets
lint:
	@echo "Running code linting..."
	UV_PROJECT_ENVIRONMENT=.venv_linux uv run ruff check

format:
	@echo "Formatting code..."
	UV_PROJECT_ENVIRONMENT=.venv_linux uv run ruff format
	UV_PROJECT_ENVIRONMENT=.venv_linux uv run ruff check --fix


typecheck:
	@echo "Running type checking..."
	UV_PROJECT_ENVIRONMENT=.venv_linux uv run mypy src/

# Cleanup target
clean:
	@echo "Cleaning build artifacts..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf build/ dist/ *.egg-info
	@echo "Build artifacts cleaned."
