# Makefile for Two-Stage RAG System

# Help target: Lists all make commands
help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Available targets:"
	@echo "  install         Set up virtual environment and install requirements"
	@echo "  install-dev     Set up virtual environment and install development requirements"
	@echo "  export-reqs     Export pinned requirements.txt from Poetry lock"
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
.PHONY: help run install-dev install lint format test clean export-reqs docker-build docker-run docker-stop

# Launch the Streamlit UI inside Poetry's venv
run:
	poetry run streamlit run frontend/app.py --server.fileWatcherType none


# Create venv & install all dependencies (including dev)
install-dev:
	@echo "Installing all dependencies (dev + prod) via Poetry"
	poetry config virtualenvs.in-project true --local
	poetry install

# Create venv & install only production dependencies
install:
	@echo "Installing production dependencies only"
	poetry config virtualenvs.in-project true --local
	poetry install --without dev

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
export-reqs:
	poetry export -f requirements.txt --output requirements.txt --without-hashes

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