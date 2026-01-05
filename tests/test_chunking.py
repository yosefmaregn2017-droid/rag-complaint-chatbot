# =============================================================================
# test_chunking.py - Tests for Text Chunking
# =============================================================================
"""
Tests for the chunking module.

These tests ensure:
1. Chunking produces at least one chunk per document
2. Chunk overlap works correctly
3. Metadata is preserved through chunking

Run with: pytest tests/test_chunking.py -v
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document
from src.chunking import (
    create_text_splitter,
    chunk_documents,
    get_chunk_stats,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def short_document():
    """A document shorter than chunk_size (should stay as one chunk)."""
    return Document(
        page_content="This is a short document about a billing issue.",
        metadata={"ticket_id": "1", "product": "TV"}
    )


@pytest.fixture
def long_document():
    """A document longer than chunk_size (should be split)."""
    # Create a ~1000 character document
    content = """
    Subject: Major technical issue with smart TV
    
    Description: I purchased this smart TV last month and have been experiencing 
    multiple issues. First, the TV randomly turns off during viewing. This happens 
    about 3-4 times per day, usually when watching streaming content. Second, the 
    WiFi connection keeps dropping, which makes it impossible to use any smart 
    features. I've tried resetting the router, updating the TV firmware, and even 
    factory resetting the TV, but nothing has worked.
    
    The TV model is XYZ-2000 and I bought it on January 15th. I've contacted support 
    twice before but the issue persists. This is very frustrating as I paid premium 
    price for this product and expected better quality.
    
    Resolution: After extensive troubleshooting, we determined the issue was with 
    the main board. A replacement unit was sent to the customer and the issue was 
    resolved. Customer confirmed the new TV is working properly.
    """
    return Document(
        page_content=content.strip(),
        metadata={"ticket_id": "2", "product": "Smart TV", "priority": "High"}
    )


@pytest.fixture
def sample_documents(short_document, long_document):
    """List of sample documents for testing."""
    return [short_document, long_document]


# =============================================================================
# TEXT SPLITTER TESTS
# =============================================================================

class TestTextSplitter:
    """Tests for text splitter creation."""
    
    def test_create_splitter_default_params(self):
        """Should create splitter with default parameters."""
        splitter = create_text_splitter()
        assert splitter is not None
        assert splitter._chunk_size == 500  # Default from config
        assert splitter._chunk_overlap == 50  # Default from config
    
    def test_create_splitter_custom_params(self):
        """Should create splitter with custom parameters."""
        splitter = create_text_splitter(chunk_size=300, chunk_overlap=30)
        assert splitter._chunk_size == 300
        assert splitter._chunk_overlap == 30


# =============================================================================
# CHUNKING TESTS
# =============================================================================

class TestChunking:
    """Tests for document chunking."""
    
    def test_chunk_returns_list(self, sample_documents):
        """chunk_documents should return a list."""
        chunks = chunk_documents(sample_documents)
        assert isinstance(chunks, list)
    
    def test_chunk_returns_documents(self, sample_documents):
        """chunk_documents should return Document objects."""
        chunks = chunk_documents(sample_documents)
        assert all(isinstance(c, Document) for c in chunks)
    
    def test_at_least_one_chunk_per_doc(self, sample_documents):
        """Each input document should produce at least one chunk."""
        chunks = chunk_documents(sample_documents)
        # We have 2 input docs, should have at least 2 chunks
        assert len(chunks) >= len(sample_documents)
    
    def test_short_doc_single_chunk(self, short_document):
        """Short documents should remain as single chunk."""
        chunks = chunk_documents([short_document], chunk_size=500)
        # Short doc should produce exactly 1 chunk
        assert len(chunks) == 1
    
    def test_long_doc_multiple_chunks(self, long_document):
        """Long documents should be split into multiple chunks."""
        chunks = chunk_documents([long_document], chunk_size=300)
        # Long doc (~1000 chars) with chunk_size=300 should produce multiple chunks
        assert len(chunks) > 1
    
    def test_metadata_preserved(self, sample_documents):
        """Metadata should be preserved through chunking."""
        chunks = chunk_documents(sample_documents)
        
        # Find chunks from the long document (ticket_id="2")
        long_doc_chunks = [c for c in chunks if c.metadata.get("ticket_id") == "2"]
        
        # All chunks should have the original metadata
        for chunk in long_doc_chunks:
            assert chunk.metadata.get("product") == "Smart TV"
            assert chunk.metadata.get("priority") == "High"
    
    def test_chunk_index_added(self, long_document):
        """Chunks should have chunk_index in metadata."""
        chunks = chunk_documents([long_document], chunk_size=300)
        
        # Each chunk should have a chunk_index
        for chunk in chunks:
            assert "chunk_index" in chunk.metadata
    
    def test_chunk_indices_sequential(self, long_document):
        """Chunk indices should be sequential starting from 0."""
        chunks = chunk_documents([long_document], chunk_size=300)
        
        indices = [c.metadata["chunk_index"] for c in chunks]
        expected = list(range(len(chunks)))
        assert indices == expected


# =============================================================================
# OVERLAP TESTS
# =============================================================================

class TestChunkOverlap:
    """Tests for chunk overlap behavior."""
    
    def test_overlap_creates_shared_content(self, long_document):
        """Consecutive chunks should share some content due to overlap."""
        chunks = chunk_documents([long_document], chunk_size=200, chunk_overlap=50)
        
        if len(chunks) >= 2:
            # Get content of first two chunks
            chunk1_content = chunks[0].page_content
            chunk2_content = chunks[1].page_content
            
            # The end of chunk1 should appear at the start of chunk2
            # (approximately, due to how splitter finds boundaries)
            # We just check that they're not completely disjoint
            chunk1_words = set(chunk1_content.split())
            chunk2_words = set(chunk2_content.split())
            
            # There should be some overlap in words
            overlap = chunk1_words.intersection(chunk2_words)
            assert len(overlap) > 0, "Chunks should have some overlapping content"
    
    def test_zero_overlap_allowed(self, long_document):
        """Should work with zero overlap."""
        chunks = chunk_documents([long_document], chunk_size=200, chunk_overlap=0)
        assert len(chunks) >= 1


# =============================================================================
# CHUNK STATS TESTS
# =============================================================================

class TestChunkStats:
    """Tests for chunk statistics."""
    
    def test_stats_returns_dict(self, sample_documents):
        """get_chunk_stats should return a dictionary."""
        chunks = chunk_documents(sample_documents)
        stats = get_chunk_stats(chunks)
        assert isinstance(stats, dict)
    
    def test_stats_has_required_keys(self, sample_documents):
        """Stats should have all required keys."""
        chunks = chunk_documents(sample_documents)
        stats = get_chunk_stats(chunks)
        
        required_keys = ["total_chunks", "min_length", "max_length", "mean_length", "median_length"]
        for key in required_keys:
            assert key in stats, f"Missing key: {key}"
    
    def test_stats_values_sensible(self, sample_documents):
        """Stats values should be sensible."""
        chunks = chunk_documents(sample_documents)
        stats = get_chunk_stats(chunks)
        
        assert stats["total_chunks"] == len(chunks)
        assert stats["min_length"] <= stats["max_length"]
        assert stats["min_length"] <= stats["mean_length"] <= stats["max_length"]
    
    def test_empty_chunks_handled(self):
        """Should handle empty chunk list."""
        stats = get_chunk_stats([])
        assert "error" in stats


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_document(self):
        """Should handle empty document content."""
        empty_doc = Document(page_content="", metadata={"ticket_id": "empty"})
        chunks = chunk_documents([empty_doc])
        # Empty doc might produce 0 or 1 chunk depending on implementation
        assert isinstance(chunks, list)
    
    def test_whitespace_only_document(self):
        """Should handle whitespace-only document."""
        ws_doc = Document(page_content="   \n\n   ", metadata={"ticket_id": "ws"})
        chunks = chunk_documents([ws_doc])
        assert isinstance(chunks, list)
    
    def test_very_long_word(self):
        """Should handle documents with very long words."""
        long_word = "a" * 1000  # 1000 character "word"
        doc = Document(page_content=long_word, metadata={"ticket_id": "long"})
        chunks = chunk_documents([doc], chunk_size=500)
        # Should still produce chunks (splitter falls back to character split)
        assert len(chunks) >= 1


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
