"""Unit tests for Retrievers class functionality."""

import pytest
from backend.my_lib.retrievers import Retrievers
from backend.my_lib.pdf_manager import PDFManager
from langchain_core.documents import Document

# ====================================
# Mock vector store for testing
# ====================================
class DummyVS:
    """Mock vector store class for testing."""
    
    def similarity_search(self, query, k, filter):
        """Mock similarity search method."""
        return []

# ====================================
# Test retriever setup and small chunk retrieval
# ====================================
def test_setup_and_retrieve_small(config, tmp_pdf_dir, monkeypatch):
    """Test retriever setup and small chunk retrieval with mocked dependencies."""
    # prepare PDFManager with proper Document objects
    mgr = PDFManager(tmp_pdf_dir, config)
    # Create proper Document objects instead of dummy objects
    mgr.small_chunks = [
        Document(page_content="foo", metadata={"name": "test.pdf", "page": 1})
    ]
    
    retr = Retrievers(mgr, config)
    
    # Mock the BM25Retriever.from_documents to avoid the actual setup
    def mock_from_documents(documents, k):
        mock_retriever = type("MockRetriever", (), {
            "invoke": lambda self, q: documents
        })()
        return mock_retriever
    
    # Mock CrossEncoder to avoid loading actual models
    def mock_cross_encoder(model_name):
        return type("MockCE", (), {
            "predict": lambda self, pairs: [1.0] * len(pairs)
        })()
    
    monkeypatch.setattr("backend.my_lib.retrievers.BM25Retriever.from_documents", mock_from_documents)
    monkeypatch.setattr("backend.my_lib.retrievers.CrossEncoder", mock_cross_encoder)
    
    retr.setup_retrievers()
    
    # Now test retrieve_small_chunks
    chunks = retr.retrieve_small_chunks("foo")
    assert chunks is not None
    assert len(chunks) > 0
    assert hasattr(chunks[0].metadata, "__getitem__")
    assert "score" in chunks[0].metadata

# ====================================
# Test large chunk retrieval functionality
# ====================================
def test_retrieve_large(config, tmp_pdf_dir):
    """Test large chunk retrieval with dummy vector store."""
    mgr = PDFManager(tmp_pdf_dir, config)
    retr = Retrievers(mgr, config)
    # inject dummy vectorstore
    retr.vectorstore = DummyVS()
    # Fix the mock CrossEncoder rank method signature
    retr.CE_model_semantic = type("CE", (), {
        "rank": lambda self, question, passages: []  # Fixed: now takes question and passages
    })()
    result = retr.retrieve_large_chunks("q", ["some.pdf"])
    assert result == []
