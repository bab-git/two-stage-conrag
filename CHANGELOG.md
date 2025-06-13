# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] Refactored backend with Docker support

### Added
- `Dockerfile`: simplified Docker build using `requirements.txt` exported from Poetry
- `Makefile` target: `export-reqs` to export pinned dependencies from `poetry.lock`
- `requirements_fallback.txt` as a fallback for environments without Poetry

### Changed
- Dockerfile installs dependencies via `pip install -r requirements.txt` for faster and more compatible image builds
- Removed runtime dependency on Poetry inside the Docker container

## [0.1.0] Initial implementation of two-stage RAG PDF QA system

### Added
- Initial implementation of two-stage RAG PDF QA system
- Poetry-based dependency management (`pyproject.toml`, `poetry.lock`)
- Streamlit frontend interface (`frontend/app.py`)
- Basic unit tests, logging setup, and project structure
