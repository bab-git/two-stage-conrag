# tests/conftest.py
import os
import shutil
import pytest
from omegaconf import OmegaConf

"""Pytest configuration and shared fixtures for the test suite."""

# ====================================
# Create temporary PDF directory fixture
# ====================================
@pytest.fixture
def tmp_pdf_dir(tmp_path):
    """
    Creates a temporary directory with one zero-byte PDF file.
    """
    d = tmp_path / "pdfs"
    d.mkdir()
    (d / "empty.pdf").write_bytes(b"%%EOF")
    return str(d)

# ====================================
# Create minimal configuration fixture
# ====================================
@pytest.fixture
def config(tmp_path):
    """
    A minimal OmegaConf config matching your pdf_manager signature.
    """
    return OmegaConf.create({
        "llm": {"embed_model_id": "test-model", "openai_modelID": "test-model"},
        "Vectorstore": {
            "persist_directory": str(tmp_path / "persist"),
            "collection_name": "test-collection"
        },
        "splitter": {
            "small_chunk_size": 400,
            "large_chunk_size": 2000,
            "paragraph_separator": "\n\n"
        },
        "Retrieval": {
            "top_k_BM25": 2,
            "top_k_semantic": 2,
            "top_k_documents": 1,
            "semantic_CE_model": "cross-encoder/test",
            "keyword_CE_model": "cross-encoder/test",
            "top_k_final": 1
        },
        "settings": {"verbose": False}
    })
