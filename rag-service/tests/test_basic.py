"""
Basic tests for the RAG service.
Run with: pytest tests/ -v
"""
import pytest
from rag_service.utils import get_domain, is_same_domain, chunk_text, clean_text

def test_get_domain():
    """Test domain extraction."""
    assert get_domain("https://example.com/path") == "example.com"
    assert get_domain("https://www.example.com/path") == "example.com"
    assert get_domain("http://sub.example.com:8080/path") == "sub.example.com"

def test_is_same_domain():
    """Test domain comparison."""
    assert is_same_domain("https://example.com/page1", "https://example.com/page2")
    assert is_same_domain("https://www.example.com/page1", "https://example.com/page2")
    assert not is_same_domain("https://example.com/page1", "https://different.com/page2")

def test_chunk_text():
    """Test text chunking."""
    text = "This is a test. It has multiple sentences. We want to chunk it properly."
    chunks = chunk_text(text, chunk_size=30, overlap=10)
    
    assert len(chunks) > 1
    assert all(len(chunk) <= 40 for chunk in chunks)  # Allow some flexibility
    
def test_clean_text():
    """Test text cleaning."""
    dirty_text = "  This   has    extra    spaces   and\n\nnewlines  "
    clean = clean_text(dirty_text)
    
    assert "extra" in clean
    assert "    " not in clean  # No multiple spaces
    assert clean.strip() == clean  # No leading/trailing spaces

# Test configuration loading
def test_config_import():
    """Test that we can import config without errors."""
    from rag_service.config import DEFAULT_CHUNK_SIZE, DEFAULT_TOP_K
    
    assert isinstance(DEFAULT_CHUNK_SIZE, int)
    assert DEFAULT_CHUNK_SIZE > 0
    assert isinstance(DEFAULT_TOP_K, int)
    assert DEFAULT_TOP_K > 0