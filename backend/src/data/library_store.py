"""Persistent library storage for saved papers with JSON file backend"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import json
import os
import logging

logger = logging.getLogger(__name__)

# Data directory for library storage
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
LIBRARY_FILE = os.path.join(DATA_DIR, "library.json")


@dataclass
class SavedPaperData:
    """Saved paper data"""
    id: str
    user_id: str
    pmid: str
    title: str
    abstract: str
    authors: List[str] = field(default_factory=list)
    journal: str = ""
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    saved_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "pmid": self.pmid,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "journal": self.journal,
            "tags": self.tags,
            "notes": self.notes,
            "saved_at": self.saved_at if isinstance(self.saved_at, str) else self.saved_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SavedPaperData":
        """Create SavedPaperData from dictionary"""
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            pmid=data["pmid"],
            title=data["title"],
            abstract=data["abstract"],
            authors=data.get("authors", []),
            journal=data.get("journal", ""),
            tags=data.get("tags", []),
            notes=data.get("notes"),
            saved_at=data.get("saved_at", datetime.utcnow().isoformat())
        )


class LibraryStore:
    """Persistent library storage with JSON file backend"""

    def __init__(self, library_file: str = LIBRARY_FILE):
        self._library_file = library_file
        # user_id -> list of saved papers
        self._papers: Dict[str, List[SavedPaperData]] = {}
        self._ensure_data_dir()
        self._load_library()

    def _ensure_data_dir(self):
        """Ensure data directory exists"""
        os.makedirs(os.path.dirname(self._library_file), exist_ok=True)

    def _load_library(self):
        """Load library from JSON file"""
        if os.path.exists(self._library_file):
            try:
                with open(self._library_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for user_id, papers_data in data.get("library", {}).items():
                        self._papers[user_id] = [
                            SavedPaperData.from_dict(p) for p in papers_data
                        ]
                total_papers = sum(len(p) for p in self._papers.values())
                logger.info(f"Loaded {total_papers} papers for {len(self._papers)} users from {self._library_file}")
            except Exception as e:
                logger.error(f"Failed to load library: {e}")
                self._papers = {}

    def _save_library(self):
        """Save library to JSON file"""
        try:
            data = {
                "library": {
                    user_id: [p.to_dict() for p in papers]
                    for user_id, papers in self._papers.items()
                },
                "updated_at": datetime.utcnow().isoformat()
            }
            with open(self._library_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug(f"Saved library to {self._library_file}")
        except Exception as e:
            logger.error(f"Failed to save library: {e}")

    def save_paper(
        self,
        user_id: str,
        pmid: str,
        title: str,
        abstract: str,
        authors: List[str] = None,
        journal: str = "",
        tags: List[str] = None,
        notes: Optional[str] = None
    ) -> SavedPaperData:
        """Save a paper to user's library and persist to file"""
        if user_id not in self._papers:
            self._papers[user_id] = []

        # Check if paper already saved
        for paper in self._papers[user_id]:
            if paper.pmid == pmid:
                # Update existing paper
                paper.tags = tags or paper.tags
                paper.notes = notes if notes is not None else paper.notes
                self._save_library()  # Persist changes
                return paper

        # Create new saved paper
        paper = SavedPaperData(
            id=str(uuid.uuid4()),
            user_id=user_id,
            pmid=pmid,
            title=title,
            abstract=abstract,
            authors=authors or [],
            journal=journal,
            tags=tags or [],
            notes=notes
        )

        self._papers[user_id].append(paper)
        self._save_library()  # Persist to file
        logger.info(f"Saved paper {pmid} for user {user_id}")
        return paper

    def get_papers(
        self,
        user_id: str,
        tag: Optional[str] = None,
        limit: int = 20,
        offset: int = 0
    ) -> tuple[int, List[SavedPaperData]]:
        """Get user's saved papers"""
        if user_id not in self._papers:
            return 0, []

        papers = self._papers[user_id]

        # Filter by tag if provided
        if tag:
            papers = [p for p in papers if tag in p.tags]

        # Sort by saved_at (most recent first)
        papers = sorted(papers, key=lambda p: p.saved_at, reverse=True)

        total = len(papers)
        papers = papers[offset:offset + limit]

        return total, papers

    def get_paper(self, user_id: str, paper_id: str) -> Optional[SavedPaperData]:
        """Get a specific saved paper"""
        if user_id not in self._papers:
            return None

        for paper in self._papers[user_id]:
            if paper.id == paper_id:
                return paper
        return None

    def get_paper_by_pmid(self, user_id: str, pmid: str) -> Optional[SavedPaperData]:
        """Get a saved paper by PMID"""
        if user_id not in self._papers:
            return None

        for paper in self._papers[user_id]:
            if paper.pmid == pmid:
                return paper
        return None

    def update_paper(
        self,
        user_id: str,
        paper_id: str,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None
    ) -> Optional[SavedPaperData]:
        """Update a saved paper and persist changes"""
        paper = self.get_paper(user_id, paper_id)
        if not paper:
            return None

        if tags is not None:
            paper.tags = tags
        if notes is not None:
            paper.notes = notes

        self._save_library()  # Persist changes
        return paper

    def delete_paper(self, user_id: str, paper_id: str) -> bool:
        """Remove a paper from library and persist changes"""
        if user_id not in self._papers:
            return False

        for i, paper in enumerate(self._papers[user_id]):
            if paper.id == paper_id:
                pmid = paper.pmid
                del self._papers[user_id][i]
                self._save_library()  # Persist changes
                logger.info(f"Deleted paper {pmid} for user {user_id}")
                return True
        return False

    def get_tags(self, user_id: str) -> List[str]:
        """Get all unique tags for a user"""
        if user_id not in self._papers:
            return []

        tags = set()
        for paper in self._papers[user_id]:
            tags.update(paper.tags)

        return sorted(list(tags))

    def is_paper_saved(self, user_id: str, pmid: str) -> bool:
        """Check if a paper is already saved"""
        return self.get_paper_by_pmid(user_id, pmid) is not None


# Global library store instance
library_store = LibraryStore()
