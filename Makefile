# Makefile for langchain-viscribe

.PHONY: install lint type-check test test-unit test-integration format build clean pre-commit

# Variables
PACKAGE_NAME = langchain_viscribe
TEST_DIR = tests

# Default target
all: lint type-check test

# Install project dependencies
install:
	uv sync

# Linting and Formatting Checks
lint:
	uv run ruff check $(PACKAGE_NAME) $(TEST_DIR)
	uv run black --check $(PACKAGE_NAME) $(TEST_DIR)
	uv run isort --check-only $(PACKAGE_NAME) $(TEST_DIR)

# Format code
format:
	uv run ruff check --fix $(PACKAGE_NAME) $(TEST_DIR)
	uv run black $(PACKAGE_NAME) $(TEST_DIR)
	uv run isort $(PACKAGE_NAME) $(TEST_DIR)

# Type Checking with MyPy
type-check:
	uv run mypy $(PACKAGE_NAME) $(TEST_DIR)

# Run all tests
test:
	uv run pytest tests/unit_tests/ --disable-socket -v

# Run unit tests only
test-unit:
	uv run pytest tests/unit_tests/ --disable-socket -v

# Run integration tests (requires API key)
test-integration:
	uv run pytest tests/integration_tests/ -v

# Run tests with coverage
test-cov:
	uv run pytest tests/unit_tests/ --disable-socket --cov=$(PACKAGE_NAME) --cov-report=xml --cov-report=html

# Run Pre-Commit Hooks
pre-commit:
	uv run pre-commit run --all-files

# Clean Up Generated Files
clean:
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Build the Package
build:
	uv build

# Install pre-commit hooks
install-hooks:
	uv run pre-commit install
