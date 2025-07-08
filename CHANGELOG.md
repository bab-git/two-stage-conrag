# Changelog

All notable changes to this project will be documented in this file.

## [1.2.0] Loading Screen Implementation

### Added
- **Loading Screen**: Implemented elegant loading indicators for both local and Modal deployments
  - Simple `st.info()` loading message that appears during app initialization
  - Automatic clearing of loading message once all components are loaded
  - Improved user experience during cold starts and app startup
  
## [1.1.0] Groq Models Support and UX Enhancements

### Added
- More Sample PDFs: Added more sample PDFs and functionality to show relevant chunks and documents.

### Changed
- **UX Improvements**: 
  - Free LLMs are now ordered for better user experience.
  - Model categories have been updated for improved navigation.
  - Multi-model options and API key per model have been introduced.
  - Model selection refactor: Sidebar choices have been improved.

### Fixed
- **SQLite**: 
  - Replace `sqlite3` with `pysqlite3-binary` to overcome the limitation on streamlit cloud.

### Testing
- **LLM Manager**: Updated mock tests for LLM manager.

### Miscellaneous
- **Streamlit**: Now using Streamlit secrets for configuration.
- **GROQ Models**: Updated GROQ models.
- **Default Model**: Set default model to openai-4o-mini.

## [1.0.0] Local Models Support and Enhanced User Experience

### Added
- **Local Model Support**: Integrated `llama-cpp-python` for running local LLaMA models.
- **API Key Input**: Streamlit sidebar input for OpenAI API keys with session-based storage.
- **Model Selection UI**: Dropdown to choose between Local LLaMA and OpenAI GPT models.

### Changed
- **Dependencies**: Added `llama-cpp-python ^0.3.9`.
- **Configuration**: Updated `config.yaml` for local model settings.
- **App Architecture**: Refactored for dynamic model switching and session management.


## [0.3.0] UX Enhancements and Upload Features

### Added
- Upload support in Streamlit app for user-provided PDFs (`pdf_uploader_ui`)
- Submit buttons for both "Upload PDFs" and sample file trigger
- PDF upload processing integrated into app initialization
- Vector store cleanup with feature flag support

### Changed
- Streamlit header layout and spacing for better UX

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
