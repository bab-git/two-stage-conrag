# Makefile for Two-Stage RAG System

# Help target: Lists all make commands
help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Available targets:"
	@echo "  install         Set up virtual environment and install requirements"
	@echo "  install-dev     Set up virtual environment and install development requirements"
	@echo ""
	@echo "  export-local    Export requirements-local.txt (with llama-cpp-python)"
	@echo "  export-cloud    Export requirements.txt (without llama-cpp-python)"
	@echo ""
	@echo "  run             Run Streamlit app (frontend/app.py)"
	@echo ""
	@echo "  lint            Lint backend and frontend using Ruff"
	@echo "  format          Auto-format code using Black"
	@echo "  test            Run unit tests with pytest"
	@echo ""
	@echo "  docker-build    Build Docker image"
	@echo "  docker-run      Run Docker container"
	@echo ""
	@echo "  clean           Remove __pycache__, .pyc files, and virtual environment"



IMAGE_NAME := two-stage-conrag
IMAGE_TAG  := latest

# Phony targets: These targets are not files, so make will always run them
.PHONY: help run install-dev install install-cloud lint format test clean export-local export-cloud docker-build docker-run docker-stop

# Launch the Streamlit UI inside Poetry's venv
run:
	poetry run streamlit run frontend/app.py


# Create venv & install all dependencies (including dev)
install-dev:
	@echo "Installing all dependencies (dev + prod) via Poetry"
	poetry config virtualenvs.in-project true --local
	poetry install --with local

# Create venv & install only production dependencies
install:
	@echo "Installing production dependencies (including llama-cpp-python)"
	poetry config virtualenvs.in-project true --local
	poetry install --without dev --with local

# Create venv & install for cloud deployment (without llama-cpp-python)
install-cloud:
	@echo "Installing dependencies for cloud deployment (without llama-cpp-python)"
	poetry config virtualenvs.in-project true --local
	poetry install --without dev --without local --with cloud

# Code linting (using Ruff inside the venv)
lint:
	poetry run ruff check backend frontend scripts --fix

# Auto-format with Black
format:
	poetry run black backend frontend scripts

# Run all unit tests with pytest + coverage
test:
	@echo "Running tests with pytest…"
	poetry run pytest \
		--strict-markers \
		--tb=short \
		--disable-warnings \
		--maxfail=1 \
		--cov=backend \
		--cov-report=term-missing

# Export requirements.txt
export-local:
	@echo "Exporting requirements for local deployment (with llama-cpp-python)"
	poetry export -f requirements.txt --output requirements-local.txt --without-hashes --with local

export-cloud:
	@echo "Exporting requirements for cloud deployment (with pysqlite3-binary, without llama-cpp-python)"
	poetry export -f requirements.txt --output requirements.txt --without-hashes --without local --with cloud


# Docker-related targets
docker-build:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

docker-run:
	docker run --rm \
	  --env-file .env \
	  -p 8501:8501 \
	  $(IMAGE_NAME):$(IMAGE_TAG)

docker-stop:
	@docker ps -q --filter ancestor=$(IMAGE_NAME):$(IMAGE_TAG) | xargs -r docker stop


# Clean pyc, cache, logs, etc.
clean:
	find . -type d -name "__pycache__" -exec rm -r {} + 
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	rm -rf .venv
	@echo "Cleaned up cache files and virtual environment"