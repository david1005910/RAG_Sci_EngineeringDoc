"""Text Chunking Utilities"""

import re
import logging
from typing import List, Optional
from dataclasses import dataclass

from src.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Represents a text chunk"""
    text: str
    index: int
    section: Optional[str] = None
    token_count: Optional[int] = None
    metadata: Optional[dict] = None


class TextChunker:
    """
    Text chunking for embedding and retrieval.

    Supports:
    - Token-based chunking with overlap
    - Section-based chunking for papers
    - Text preprocessing and cleaning
    """

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        """
        Initialize chunker.

        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove reference numbers like [1], [2,3], etc.
        text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)

        # Remove figure/table references
        text = re.sub(r'\((?:Fig(?:ure)?|Table)\s*\.?\s*\d+[a-zA-Z]?\)', '', text, flags=re.IGNORECASE)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)

        # Strip and return
        return text.strip()

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count (approximate).

        Uses simple word-based estimation.
        For accuracy, use actual tokenizer.
        """
        # Rough estimate: ~1.3 tokens per word
        words = len(text.split())
        return int(words * 1.3)

    def chunk_by_tokens(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None
    ) -> List[TextChunk]:
        """
        Split text into overlapping chunks by token count.

        Args:
            text: Input text
            chunk_size: Target chunk size (uses default if None)
            overlap: Overlap size (uses default if None)

        Returns:
            List of TextChunk objects
        """
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap or self.chunk_overlap

        # Clean text first
        text = self.clean_text(text)

        if not text:
            return []

        # Split into words (simple tokenization)
        words = text.split()

        if not words:
            return []

        # Estimate words per chunk (assuming ~1.3 tokens per word)
        words_per_chunk = int(chunk_size / 1.3)
        words_overlap = int(overlap / 1.3)

        chunks = []
        start = 0

        while start < len(words):
            end = min(start + words_per_chunk, len(words))
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)

            chunks.append(TextChunk(
                text=chunk_text,
                index=len(chunks),
                token_count=self.estimate_tokens(chunk_text)
            ))

            # Move start position, considering overlap
            start = end - words_overlap
            if start >= len(words) - words_overlap:
                break

        logger.debug(f"Created {len(chunks)} chunks from text")
        return chunks

    def chunk_by_sentences(
        self,
        text: str,
        max_chunk_size: Optional[int] = None
    ) -> List[TextChunk]:
        """
        Split text into chunks by sentences, respecting max size.

        Args:
            text: Input text
            max_chunk_size: Maximum tokens per chunk

        Returns:
            List of TextChunk objects
        """
        max_chunk_size = max_chunk_size or self.chunk_size

        # Clean text
        text = self.clean_text(text)

        if not text:
            return []

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_tokens = self.estimate_tokens(sentence)

            if current_size + sentence_tokens > max_chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(TextChunk(
                    text=chunk_text,
                    index=len(chunks),
                    token_count=self.estimate_tokens(chunk_text)
                ))
                current_chunk = []
                current_size = 0

            current_chunk.append(sentence)
            current_size += sentence_tokens

        # Add remaining
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(TextChunk(
                text=chunk_text,
                index=len(chunks),
                token_count=self.estimate_tokens(chunk_text)
            ))

        return chunks

    def chunk_paper(
        self,
        title: str,
        abstract: str,
        sections: Optional[dict] = None
    ) -> List[TextChunk]:
        """
        Chunk a paper with section awareness.

        Args:
            title: Paper title
            abstract: Paper abstract
            sections: Optional dict of section_name -> text

        Returns:
            List of TextChunk objects with section metadata
        """
        chunks = []

        # Title + Abstract as first chunk(s)
        combined = f"{title}. {abstract}" if abstract else title
        abstract_chunks = self.chunk_by_tokens(combined)

        for chunk in abstract_chunks:
            chunk.section = "abstract"
            chunk.index = len(chunks)
            chunks.append(chunk)

        # Process additional sections if provided
        if sections:
            for section_name, section_text in sections.items():
                section_chunks = self.chunk_by_tokens(section_text)
                for chunk in section_chunks:
                    chunk.section = section_name.lower()
                    chunk.index = len(chunks)
                    chunks.append(chunk)

        return chunks
