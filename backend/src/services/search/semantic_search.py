"""Semantic Search Service"""

import logging
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from src.services.embedding.generator import EmbeddingGenerator, get_embedding_generator
from src.services.storage.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result item"""
    pmid: str
    title: str
    abstract: str
    relevance_score: float
    authors: List[str]
    journal: str
    publication_date: Optional[str] = None
    keywords: List[str] = None


@dataclass
class SearchResponse:
    """Search response"""
    total: int
    took_ms: int
    results: List[SearchResult]


class SemanticSearchService:
    """
    Semantic search service for papers.

    Features:
    - Vector similarity search
    - Query expansion
    - Result filtering
    - Chunk-to-paper aggregation
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None
    ):
        """
        Initialize search service.

        Args:
            vector_store: Vector store instance
            embedding_generator: Embedding generator instance
        """
        self.vector_store = vector_store or get_vector_store()
        self.embedding_generator = embedding_generator or get_embedding_generator()

    async def search(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None
    ) -> SearchResponse:
        """
        Search for papers using semantic similarity.

        Args:
            query: Search query (natural language)
            limit: Maximum results to return
            offset: Offset for pagination
            filters: Optional filters (year, journal, etc.)

        Returns:
            SearchResponse with results
        """
        start_time = time.time()

        # 1. Generate query embedding
        query_embedding = self.embedding_generator.encode(query)

        # 2. Build filter
        filter_dict = self._build_filter(filters) if filters else None

        # 3. Search vector store (get more for aggregation)
        search_limit = (limit + offset) * 3  # Get more chunks for paper aggregation
        raw_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=search_limit,
            filter_dict=filter_dict
        )

        # 4. Aggregate chunks by paper
        paper_results = self._aggregate_by_paper(raw_results)

        # 5. Apply pagination
        total = len(paper_results)
        paginated = paper_results[offset:offset + limit]

        # 6. Format results
        results = [
            SearchResult(
                pmid=p['pmid'],
                title=p['title'],
                abstract=p['abstract'],
                relevance_score=p['score'],
                authors=p.get('authors', []),
                journal=p.get('journal', ''),
                publication_date=p.get('publication_date'),
                keywords=p.get('keywords', [])
            )
            for p in paginated
        ]

        took_ms = int((time.time() - start_time) * 1000)

        return SearchResponse(
            total=total,
            took_ms=took_ms,
            results=results
        )

    def _build_filter(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build vector store filter from search filters."""
        filter_dict = {}

        if 'year_from' in filters and filters['year_from']:
            # Note: Qdrant filter handling would need range queries
            pass

        if 'journals' in filters and filters['journals']:
            filter_dict['journal'] = filters['journals']

        return filter_dict if filter_dict else None

    def _aggregate_by_paper(self, results: List[Dict]) -> List[Dict]:
        """
        Aggregate chunk results by paper PMID.

        Takes the highest scoring chunk for each paper.
        """
        paper_map = {}

        for result in results:
            pmid = result['metadata'].get('pmid')
            if not pmid:
                continue

            if pmid not in paper_map:
                paper_map[pmid] = {
                    'pmid': pmid,
                    'title': result['metadata'].get('title', ''),
                    'abstract': result['text'],
                    'score': result['score'],
                    'authors': result['metadata'].get('authors', []),
                    'journal': result['metadata'].get('journal', ''),
                    'publication_date': result['metadata'].get('publication_date'),
                    'keywords': result['metadata'].get('keywords', [])
                }
            else:
                # Update with higher score if found
                if result['score'] > paper_map[pmid]['score']:
                    paper_map[pmid]['score'] = result['score']

        # Sort by score and return
        papers = list(paper_map.values())
        papers.sort(key=lambda x: x['score'], reverse=True)

        return papers

    async def search_similar(
        self,
        pmid: str,
        limit: int = 5
    ) -> List[SearchResult]:
        """
        Find papers similar to a given paper.

        Args:
            pmid: PMID of the reference paper
            limit: Number of similar papers to return

        Returns:
            List of similar papers
        """
        # Get the paper's chunks
        paper_chunks = self.vector_store.search(
            query_embedding=None,  # We need to get by filter
            top_k=1,
            filter_dict={'pmid': pmid}
        )

        if not paper_chunks:
            return []

        # Use first chunk's text to find similar
        reference_text = paper_chunks[0]['text']
        reference_embedding = self.embedding_generator.encode(reference_text)

        # Search excluding the original paper
        all_results = self.vector_store.search(
            query_embedding=reference_embedding,
            top_k=limit * 3
        )

        # Filter out the original paper and aggregate
        filtered = [r for r in all_results if r['metadata'].get('pmid') != pmid]
        papers = self._aggregate_by_paper(filtered)[:limit]

        return [
            SearchResult(
                pmid=p['pmid'],
                title=p['title'],
                abstract=p['abstract'],
                relevance_score=p['score'],
                authors=p.get('authors', []),
                journal=p.get('journal', '')
            )
            for p in papers
        ]


# Singleton instance
_search_service_instance: Optional[SemanticSearchService] = None


def get_search_service() -> SemanticSearchService:
    """Get or create singleton search service."""
    global _search_service_instance
    if _search_service_instance is None:
        _search_service_instance = SemanticSearchService()
    return _search_service_instance
