"""Paper Recommendation Service"""

import logging
from typing import List, Optional
from dataclasses import dataclass

import numpy as np

from src.services.embedding.generator import EmbeddingGenerator, get_embedding_generator
from src.services.storage.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class RecommendedPaper:
    """Recommended paper"""
    pmid: str
    title: str
    similarity_score: float
    common_keywords: List[str]


class PaperRecommender:
    """
    Paper recommendation service.

    Uses embedding similarity to find related papers.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None
    ):
        """
        Initialize recommender.

        Args:
            vector_store: Vector store instance
            embedding_generator: Embedding generator instance
        """
        self.vector_store = vector_store or get_vector_store()
        self.embedding_generator = embedding_generator or get_embedding_generator()

    async def recommend_similar(
        self,
        pmid: str,
        top_k: int = 5
    ) -> List[RecommendedPaper]:
        """
        Recommend papers similar to a given paper.

        Args:
            pmid: PMID of the reference paper
            top_k: Number of recommendations

        Returns:
            List of recommended papers
        """
        # Get reference paper chunks
        reference_results = self.vector_store.search(
            query_embedding=np.zeros(768),  # Dummy, we'll filter by PMID
            top_k=10,
            filter_dict={'pmid': pmid}
        )

        if not reference_results:
            logger.warning(f"Paper {pmid} not found in vector store")
            return []

        # Get paper metadata
        reference_paper = reference_results[0]['metadata']
        reference_keywords = set(reference_paper.get('keywords', []))

        # Combine chunk texts and create embedding
        combined_text = " ".join([r['text'] for r in reference_results])
        paper_embedding = self.embedding_generator.encode(combined_text)

        # Search for similar papers
        similar_results = self.vector_store.search(
            query_embedding=paper_embedding,
            top_k=top_k * 3  # Get more for filtering
        )

        # Aggregate by paper and filter out reference
        recommendations = []
        seen_pmids = {pmid}

        for result in similar_results:
            result_pmid = result['metadata'].get('pmid')
            if not result_pmid or result_pmid in seen_pmids:
                continue

            seen_pmids.add(result_pmid)

            # Find common keywords
            result_keywords = set(result['metadata'].get('keywords', []))
            common = list(reference_keywords & result_keywords)

            recommendations.append(RecommendedPaper(
                pmid=result_pmid,
                title=result['metadata'].get('title', 'Unknown'),
                similarity_score=round(result['score'], 3),
                common_keywords=common[:5]  # Limit common keywords
            ))

            if len(recommendations) >= top_k:
                break

        return recommendations

    async def recommend_by_keywords(
        self,
        keywords: List[str],
        top_k: int = 10,
        exclude_pmids: Optional[List[str]] = None
    ) -> List[RecommendedPaper]:
        """
        Recommend papers based on keywords.

        Args:
            keywords: List of keywords to search for
            top_k: Number of recommendations
            exclude_pmids: PMIDs to exclude from results

        Returns:
            List of recommended papers
        """
        exclude_pmids = set(exclude_pmids or [])

        # Create query from keywords
        query = " ".join(keywords)
        query_embedding = self.embedding_generator.encode(query)

        # Search
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k * 2
        )

        # Aggregate and filter
        recommendations = []
        seen_pmids = set()

        for result in results:
            result_pmid = result['metadata'].get('pmid')
            if not result_pmid or result_pmid in seen_pmids or result_pmid in exclude_pmids:
                continue

            seen_pmids.add(result_pmid)

            result_keywords = set(result['metadata'].get('keywords', []))
            common = list(set(keywords) & result_keywords)

            recommendations.append(RecommendedPaper(
                pmid=result_pmid,
                title=result['metadata'].get('title', 'Unknown'),
                similarity_score=round(result['score'], 3),
                common_keywords=common
            ))

            if len(recommendations) >= top_k:
                break

        return recommendations


# Singleton instance
_recommender_instance: Optional[PaperRecommender] = None


def get_recommender() -> PaperRecommender:
    """Get or create singleton recommender."""
    global _recommender_instance
    if _recommender_instance is None:
        _recommender_instance = PaperRecommender()
    return _recommender_instance
