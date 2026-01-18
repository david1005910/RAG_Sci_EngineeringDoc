"""User Library Model - Saved Papers"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from src.core.database import Base


class SavedPaper(Base):
    """User's saved paper model"""

    __tablename__ = "saved_papers"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    pmid = Column(String(20), nullable=False, index=True)
    title = Column(Text, nullable=False)
    abstract = Column(Text, nullable=True)
    tags = Column(JSONB, default=list)  # User-defined tags
    notes = Column(Text, nullable=True)  # User notes
    saved_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="saved_papers")

    def __repr__(self):
        return f"<SavedPaper PMID:{self.pmid} by user {self.user_id}>"
