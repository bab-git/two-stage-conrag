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
env:
	python -m venv .venv && source .venv/bin/activate && \
	pip install -r backend/requirements.txt && \
	pip install -r frontend/requirements.txt

# Code linting
lint:
	ruff check backend/ frontend/ scripts/ --fix

# Auto-format with Black
format:
	black backend frontend scripts

# Run all unit tests
test:
	pytest tests/

# Clean pyc, cache, logs, etc.
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache .mypy_cache .venv
