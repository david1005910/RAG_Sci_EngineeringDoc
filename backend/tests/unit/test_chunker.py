"""Tests for TextChunker"""

import pytest
from src.services.embedding.chunker import TextChunker, TextChunk


class TestTextChunker:
    """Test cases for TextChunker"""

    @pytest.fixture
    def chunker(self):
        return TextChunker(chunk_size=100, chunk_overlap=20)

    def test_clean_text_removes_references(self, chunker):
        """Test that reference numbers are removed"""
        text = "This is a test [1] with references [2,3] in it."
        cleaned = chunker.clean_text(text)
        assert "[1]" not in cleaned
        assert "[2,3]" not in cleaned

    def test_clean_text_normalizes_whitespace(self, chunker):
        """Test that whitespace is normalized"""
        text = "This   has   multiple    spaces"
        cleaned = chunker.clean_text(text)
        assert "  " not in cleaned

    def test_chunk_by_tokens_creates_chunks(self, chunker):
        """Test that chunking creates multiple chunks for long text"""
        long_text = " ".join(["word"] * 500)
        chunks = chunker.chunk_by_tokens(long_text)
        assert len(chunks) > 1
        assert all(isinstance(c, TextChunk) for c in chunks)

    def test_chunk_by_tokens_has_correct_indices(self, chunker):
        """Test that chunk indices are sequential"""
        long_text = " ".join(["word"] * 500)
        chunks = chunker.chunk_by_tokens(long_text)
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_empty_text_returns_empty_list(self, chunker):
        """Test that empty text returns no chunks"""
        chunks = chunker.chunk_by_tokens("")
        assert chunks == []

    def test_chunk_paper_includes_abstract_section(self, chunker):
        """Test that paper chunking labels abstract section"""
        chunks = chunker.chunk_paper(
            title="Test Title",
            abstract="This is a test abstract."
        )
        assert len(chunks) >= 1
        assert chunks[0].section == "abstract"

    def test_estimate_tokens(self, chunker):
        """Test token estimation"""
        text = "This is a test sentence with some words."
        tokens = chunker.estimate_tokens(text)
        assert tokens > 0
        # Roughly 1.3 tokens per word, 8 words
        assert 8 <= tokens <= 15
