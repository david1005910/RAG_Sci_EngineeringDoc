"""VectorDB Metadata Store - Full paper metadata storage for indexed papers

Stores complete paper metadata when papers are indexed to VectorDB,
ensuring full abstracts and all authors are preserved.
"""

import os
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Data directory for metadata storage
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
METADATA_FILE = os.path.join(DATA_DIR, "vectordb_metadata.json")


class VectorDBMetadataStore:
    """Persistent storage for full paper metadata indexed in VectorDB"""

    def __init__(self, metadata_file: str = METADATA_FILE):
        self._metadata_file = metadata_file
        self._papers: Dict[str, dict] = {}  # pmid -> full paper data
        self._ensure_data_dir()
        self._load_metadata()

    def _ensure_data_dir(self):
        """Ensure data directory exists"""
        os.makedirs(os.path.dirname(self._metadata_file), exist_ok=True)

    def _load_metadata(self):
        """Load metadata from JSON file"""
        if os.path.exists(self._metadata_file):
            try:
                with open(self._metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._papers = data.get("papers", {})
                logger.info(f"Loaded {len(self._papers)} paper metadata from {self._metadata_file}")
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
                self._papers = {}

    def _save_metadata(self):
        """Save metadata to JSON file"""
        try:
            data = {
                "papers": self._papers,
                "updated_at": datetime.utcnow().isoformat(),
                "total_count": len(self._papers)
            }
            with open(self._metadata_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug(f"Saved {len(self._papers)} paper metadata to {self._metadata_file}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def save_paper(
        self,
        pmid: str,
        title: str,
        abstract: str,
        authors: List[str],
        journal: str = "",
        publication_date: Optional[str] = None,
        keywords: List[str] = None
    ) -> dict:
        """Save or update paper metadata"""
        paper_data = {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "authors": authors or [],
            "journal": journal,
            "publication_date": publication_date or "",
            "keywords": keywords or [],
            "indexed_at": datetime.utcnow().isoformat()
        }

        # Update or add paper
        self._papers[pmid] = paper_data
        self._save_metadata()

        return paper_data

    def save_papers_batch(self, papers: List[dict]) -> int:
        """Save multiple papers at once (batch operation)"""
        saved_count = 0
        for paper in papers:
            pmid = paper.get("pmid")
            if pmid:
                self._papers[pmid] = {
                    "pmid": pmid,
                    "title": paper.get("title", ""),
                    "abstract": paper.get("abstract", ""),
                    "authors": paper.get("authors", []),
                    "journal": paper.get("journal", ""),
                    "publication_date": paper.get("publication_date", ""),
                    "keywords": paper.get("keywords", []),
                    "indexed_at": datetime.utcnow().isoformat()
                }
                saved_count += 1

        self._save_metadata()
        logger.info(f"Batch saved {saved_count} paper metadata")
        return saved_count

    def get_paper(self, pmid: str) -> Optional[dict]:
        """Get paper metadata by PMID"""
        return self._papers.get(pmid)

    def get_all_papers(self) -> List[dict]:
        """Get all paper metadata"""
        return list(self._papers.values())

    def get_papers_count(self) -> int:
        """Get total count of papers"""
        return len(self._papers)

    def delete_paper(self, pmid: str) -> bool:
        """Delete paper metadata"""
        if pmid in self._papers:
            del self._papers[pmid]
            self._save_metadata()
            return True
        return False

    def clear_all(self):
        """Clear all paper metadata"""
        self._papers = {}
        self._save_metadata()

    def search_papers(self, query: str, limit: int = 50) -> List[dict]:
        """Simple text search in paper titles and abstracts"""
        query_lower = query.lower()
        results = []

        for paper in self._papers.values():
            title = paper.get("title", "").lower()
            abstract = paper.get("abstract", "").lower()

            if query_lower in title or query_lower in abstract:
                results.append(paper)

            if len(results) >= limit:
                break

        return results


# Global metadata store instance
vectordb_metadata_store = VectorDBMetadataStore()
