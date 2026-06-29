# Image Annotator Lib Makefile
# Development task automation

.PHONY: help test lint format install install-dev clean run-example typecheck test-unit test-integration test-webapi test-scorer test-tagger test-cov setup adr-okf adr-index docs-okf

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
	@echo "  adr-okf      Validate ADR frontmatter + check index is up to date (ADR 0010)"
	@echo "  adr-index    Regenerate ADR README table + index.md from frontmatter (ADR 0010)"
	@echo "  docs-okf     Validate docs OKF frontmatter (lazy: --skip-missing, ADR 0010)"
	@echo "  clean        Clean build artifacts"

# Setup target
setup:
	@echo "Setting up development environment..."
	./scripts/setup.sh

# Development targets
install:
	@echo "Installing project dependencies..."
	uv sync

install-dev:
	@echo "Installing development dependencies..."
	uv sync --dev

run-example:
	@echo "Running example script..."
	uv run python example/example_lib.py

# Testing targets
test:
	@echo "Running all tests..."
	uv run pytest -n auto

test-unit:
	@echo "Running unit tests..."
	uv run pytest -m unit -n auto

test-integration:
	@echo "Running integration tests..."
	uv run pytest -m integration

test-webapi:
	@echo "Running Web API tests..."
	uv run pytest -m webapi

test-scorer:
	@echo "Running scorer model tests..."
	uv run pytest -m scorer

test-tagger:
	@echo "Running tagger model tests..."
	uv run pytest -m tagger

test-cov:
	@echo "Running tests with coverage..."
	uv run pytest --cov=src --cov-report=html --cov-report=xml

# Code quality targets
lint:
	@echo "Running code linting..."
	uv run ruff check

format:
	@echo "Formatting code..."
	uv run ruff format
	uv run ruff check --fix

typecheck:
	@echo "Running type checking..."
	uv run mypy src/

# OKF ドキュメント検証・索引生成 (ADR 0010)
OKF := .agents/skills/okf-bundle/scripts
DOCS_OKF_EXCLUDE := README.md,CHANGELOG.md,CLAUDE.md,AGENTS.md,GEMINI.md,SKILL.md

adr-index:
	@echo "Regenerating ADR index from frontmatter..."
	python3 $(OKF)/okf_index.py --bundle-root docs/decisions \
		--table --columns id,title,timestamp,status --headers "ADR,タイトル,日付,ステータス" \
		--link-column id --exclude README.md --table-output docs/decisions/README.md
	python3 $(OKF)/okf_index.py --bundle-root docs/decisions \
		--index --index-output docs/decisions/index.md \
		--index-title "Architecture Decision Records" --exclude README.md

adr-okf:
	@echo "Validating ADR frontmatter (OKF)..."
	python3 $(OKF)/okf_validate.py --bundle-root docs/decisions \
		--require type,title,status,timestamp --exclude README.md
	python3 $(OKF)/okf_index.py --bundle-root docs/decisions \
		--table --columns id,title,timestamp,status --headers "ADR,タイトル,日付,ステータス" \
		--link-column id --exclude README.md --table-output docs/decisions/README.md --check
	python3 $(OKF)/okf_index.py --bundle-root docs/decisions \
		--index --index-output docs/decisions/index.md \
		--index-title "Architecture Decision Records" --exclude README.md --check

docs-okf:
	@echo "Validating documentation OKF frontmatter (lazy migration, ADR 0010)..."
	python3 $(OKF)/okf_validate.py --bundle-root docs \
		--skip-missing --exclude $(DOCS_OKF_EXCLUDE)

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
