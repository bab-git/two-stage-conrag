# Backend Components

This directory contains the core business logic and processing pipeline for the Two-Stage Consecutive RAG System.

## Structure

### `my_lib/` – Modular Backend Pipeline Components:
- **`pdf_manager.py`**: 
  - **Purpose**: Handles PDF loading, chunking, and vector store creation.
  - **Key Functions**: 
    - `load_pdfs()`: Loads PDF documents from a specified directory.
    - `chunk_documents()`: Splits PDFs into manageable chunks for processing.
    - `create_vectorstore()`: Builds a vector store for efficient retrieval.

- **`retrievers.py`**: 
  - **Purpose**: Sets up and executes BM25 + semantic retrievers.
  - **Key Functions**: 
    - `setup_retrievers()`: Initializes BM25, CrossEncoder, and retrieval configs.
    - `retrieve_small_chunks()`: Performs keyword-based retrieval with reranking.
    - `calculate_drs()`: Computes normalized document-level relevance scores.
    - `retrieve_large_chunks()`: Performs semantic search with relevance filtering.
    - `score_aggregate()`: Combines small and large chunk scores.


- **`qa_chains.py`**: 
  - **Purpose**: Runs the full QA workflow — question shortening, context retrieval, answer generation.
  - **Key Functions**:
    - `shorten_question()`: Reformulates verbose queries for keyword search.
    - `retrieve_context()`: Uses retrievers to gather and score relevant document chunks.
    - `generate_answer()`: Calls the LLM to synthesize an answer from the selected context.


- **`hybrid_retrieval.py`**: 
  - **Purpose**: Optional fusion of keyword and semantic results.
  - **Key Functions**: 
    - `fuse_results()`: Combines results from different retrieval strategies for improved accuracy.

### Other Files:
- **`settings.py`**
  - **Purpose**: Loads API keys and checks execution context.
  - **Details**:
    - Loads `.env` using `dotenv` and validates `OPENAI_API_KEY`.
    - Optionally logs `LANGSMITH_API_KEY` status.
    - Includes `is_streamlit_running()` to detect Streamlit vs CLI mode.


## Usage
1. **Environment Setup**: Create a `.env` file with required secrets (e.g., `OPENAI_API_KEY`).
2. **Configuration**: Review or adjust settings in `settings.py` or `configs/config.yaml`.
3. **Execution**: The backend is orchestrated via the Streamlit app (`frontend/app.py`) or CLI scripts in `/scripts/`.


## Notes

- The modular design allows easy replacement of components such as LLMs, retrievers, and rerankers.
- Ensure all dependencies are installed via `requirements.txt` before running the system.
