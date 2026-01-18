"""Paper and Chunk Models"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Text, Integer, Date, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from src.core.database import Base


class Paper(Base):
    """Paper metadata model"""

    __tablename__ = "papers"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pmid = Column(String(20), unique=True, nullable=False, index=True)
    title = Column(Text, nullable=False)
    abstract = Column(Text, nullable=True)
    authors = Column(JSONB, default=list)  # List of author names
    journal = Column(String(255), nullable=True)
    publication_date = Column(Date, nullable=True, index=True)
    doi = Column(String(100), nullable=True)
    keywords = Column(JSONB, default=list)  # Author-provided keywords
    mesh_terms = Column(JSONB, default=list)  # MeSH terms
    citation_count = Column(Integer, default=0)
    pdf_url = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    chunks = relationship("Chunk", back_populates="paper", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Paper PMID:{self.pmid}>"


class Chunk(Base):
    """Text chunk model for vector search"""

    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_id = Column(UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    text = Column(Text, nullable=False)
    section = Column(String(50), nullable=True)  # 'abstract', 'introduction', 'methods', etc.
    chunk_index = Column(Integer, nullable=False)  # Order within paper
    token_count = Column(Integer, nullable=True)
    embedding_id = Column(String(100), nullable=True)  # ID in vector store
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    paper = relationship("Paper", back_populates="chunks")

    def __repr__(self):
        return f"<Chunk paper={self.paper_id} index={self.chunk_index}>"
