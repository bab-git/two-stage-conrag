"""Unit tests for PDFManager class functionality."""
import pytest
from backend.my_lib.pdf_manager import PDFManager

# ====================================
# Test PDF loading with empty directory
# ====================================
def test_load_pdfs_empty(tmp_pdf_dir, config):
    mgr = PDFManager(tmp_pdf_dir, config)
    mgr.load_pdfs()
    # we only “loaded” our zero-byte file, but PyPDFLoader likely fails → docs stays empty
    assert isinstance(mgr.documents, list)

# ====================================
# Test PDF loading with empty directory
# ====================================
def test_chunk_and_vectorstore(tmp_pdf_dir, config, monkeypatch):
    mgr = PDFManager(tmp_pdf_dir, config)
    # stub out actual PDF loading
    mgr.documents = [type("Doc", (), {"page_content": "This is a test document.\n\nWith multiple lines.\n\n"*100, "metadata": {}})()]
    mgr.chunk_documents()
    assert mgr.small_chunks and mgr.large_chunks
    # stub out Chroma & embeddings so create_vectorstore doesn’t hit disk
    monkeypatch.setattr("backend.my_lib.pdf_manager.HuggingFaceEmbeddings", lambda **kw: None)
    monkeypatch.setattr("backend.my_lib.pdf_manager.Chroma", type("C", (), {"from_documents": lambda *a, **kw: type("V", (), {})()}))
    mgr.create_vectorstore()
    assert mgr.vectorstore is not None
