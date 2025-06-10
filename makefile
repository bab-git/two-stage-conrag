# Makefile for Two-Stage RAG System

# Help target: Lists all make commands
help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Available targets:"
	@echo "  run       Run Streamlit app (frontend/app.py)"
	@echo "  env       Set up virtual environment and install requirements"
	@echo "  lint      Lint backend and frontend using Ruff"
	@echo "  format    Auto-format code using Black"
	@echo "  test      Run unit tests with pytest"
	@echo "  clean     Remove __pycache__, .pyc files, and virtual environment"

# Phony targets: These targets are not files, so make will always run them
.PHONY: run ui lint format test clean env

# Launch the Streamlit UI
run:
	streamlit run frontend/app.py --server.fileWatcherType none

# Set up virtual environment
env-dev:
	@echo "Setting up virtual environment for development"
	python -m venv .venv
	. .venv/bin/activate && \
	pip install -r backend/requirements.txt && \
	pip install -r frontend/requirements.txt && \
	pip install -r requirements-dev.txt && \
	pip install -e .

# Set up virtual environment for production
env:
	@echo "Setting up virtual environment for production"
	python -m venv .venv
	. .venv/bin/activate && \
	pip install -r backend/requirements.txt && \
	pip install -r frontend/requirements.txt && \
	pip install .

# Code linting
lint:
	ruff check backend/ frontend/ scripts/ --fix

# Auto-format with Black
format:
	black backend frontend scripts

# Run all unit tests
# run your pytest suite with coverage reporting
test:
	@echo "Running tests with pytest…"
	pytest \
		--strict-markers \
		--tb=short \
		--disable-warnings \
		--maxfail=1 \
		--cov=backend \
		--cov-report=term-missing

# Clean pyc, cache, logs, etc.
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache .mypy_cache .venv
