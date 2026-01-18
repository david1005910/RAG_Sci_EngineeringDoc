"""RAG (Retrieval-Augmented Generation) Service"""

import logging
import re
import time
from typing import List, Optional, Literal
from dataclasses import dataclass

import openai
from sentence_transformers import CrossEncoder

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

from src.core.config import settings
from src.services.embedding.generator import EmbeddingGenerator, get_embedding_generator
from src.services.storage.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class RAGSource:
    """Source document information"""
    pmid: str
    title: str
    relevance: float
    excerpt: str


@dataclass
class RAGResponse:
    """RAG query response"""
    answer: str
    sources: List[RAGSource]
    confidence: float
    processing_time_ms: int


class RAGService:
    """
    Retrieval-Augmented Generation service for biomedical Q&A.

    Pipeline:
    1. Query embedding
    2. Vector search
    3. Re-ranking (optional)
    4. Context building
    5. LLM generation
    6. Response validation
    """

    SYSTEM_PROMPT = """You are an expert biomedical researcher assistant with deep knowledge of medical and life science research.

CRITICAL RULES:
1. ONLY use information from the provided research papers context
2. ALWAYS cite sources using [PMID: XXXXX] format when making claims
3. If the context doesn't contain sufficient information to answer the question, clearly state: "Based on the provided papers, I cannot find sufficient information to answer this question."
4. Be precise, factual, and avoid speculation
5. Use technical terminology appropriately but explain complex concepts when needed
6. Structure your response clearly with relevant subheadings if the answer is complex

FORMAT:
- Use bullet points for lists
- Cite each claim with the relevant PMID
- Summarize key findings at the end if multiple papers are referenced"""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        openai_api_key: Optional[str] = None,
        model: Optional[str] = None,
        use_reranking: bool = True,
        reranker_type: Literal["cohere", "crossencoder"] = "cohere"
    ):
        """
        Initialize RAG service.

        Args:
            vector_store: Vector store instance
            embedding_generator: Embedding generator instance
            openai_api_key: OpenAI API key
            model: OpenAI model to use
            use_reranking: Whether to use reranking
            reranker_type: Type of reranker ("cohere" or "crossencoder")
        """
        self.vector_store = vector_store or get_vector_store()
        self.embedding_generator = embedding_generator or get_embedding_generator()
        self.model = model or settings.OPENAI_MODEL

        # Initialize OpenAI client
        api_key = openai_api_key or settings.OPENAI_API_KEY
        self.openai_client = openai.AsyncOpenAI(api_key=api_key)

        # Initialize reranker
        self.use_reranking = use_reranking
        self.reranker_type = reranker_type
        self.reranker = None
        self.cohere_client = None

        if use_reranking:
            # Try Cohere first if specified
            if reranker_type == "cohere" and COHERE_AVAILABLE and settings.COHERE_API_KEY:
                try:
                    self.cohere_client = cohere.Client(settings.COHERE_API_KEY)
                    logger.info("Cohere reranker initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize Cohere: {e}")
                    self.cohere_client = None

            # Fallback to CrossEncoder
            if self.cohere_client is None:
                try:
                    self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
                    logger.info("CrossEncoder reranker initialized (fallback)")
                except Exception as e:
                    logger.warning(f"Failed to load CrossEncoder reranker: {e}")
                    self.reranker = None

    async def query(
        self,
        question: str,
        top_k: int = 5,
        rerank: bool = True,
        context_pmids: Optional[List[str]] = None
    ) -> RAGResponse:
        """
        Answer a question using RAG.

        Args:
            question: User's question
            top_k: Number of documents to retrieve
            rerank: Whether to apply reranking
            context_pmids: Optional list of PMIDs to restrict search to

        Returns:
            RAGResponse with answer, sources, and confidence
        """
        start_time = time.time()

        # 1. Generate query embedding
        query_embedding = self.embedding_generator.encode(question)

        # 2. Search for relevant documents
        filter_dict = None
        if context_pmids:
            filter_dict = {"pmid": context_pmids}

        # Retrieve more documents if reranking
        has_reranker = self.cohere_client or self.reranker
        search_k = top_k * 2 if rerank and has_reranker else top_k
        search_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=search_k,
            filter_dict=filter_dict
        )

        if not search_results:
            return RAGResponse(
                answer="I couldn't find any relevant papers to answer your question. Please try rephrasing or asking about a different topic.",
                sources=[],
                confidence=0.0,
                processing_time_ms=int((time.time() - start_time) * 1000)
            )

        # 3. Rerank if enabled
        if rerank and (self.cohere_client or self.reranker):
            search_results = self._rerank(question, search_results)[:top_k]

        # 4. Build context
        context = self._build_context(search_results)

        # 5. Generate answer
        answer = await self._generate_answer(question, context)

        # 6. Validate and calculate confidence
        confidence = self._validate_response(answer, search_results)

        # 7. Format sources
        sources = self._format_sources(search_results)

        processing_time = int((time.time() - start_time) * 1000)

        return RAGResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            processing_time_ms=processing_time
        )

    def _rerank(
        self,
        question: str,
        results: List[dict]
    ) -> List[dict]:
        """Rerank search results using Cohere or CrossEncoder."""
        if not results:
            return results

        # Try Cohere reranking first
        if self.cohere_client:
            try:
                return self._rerank_cohere(question, results)
            except Exception as e:
                logger.warning(f"Cohere reranking failed: {e}, falling back to CrossEncoder")

        # Fallback to CrossEncoder
        if self.reranker:
            return self._rerank_crossencoder(question, results)

        return results

    def _rerank_cohere(
        self,
        question: str,
        results: List[dict]
    ) -> List[dict]:
        """Rerank using Cohere Rerank API."""
        documents = [r['text'] for r in results]

        response = self.cohere_client.rerank(
            model="rerank-multilingual-v3.0",  # Supports multiple languages including Korean
            query=question,
            documents=documents,
            top_n=len(documents),
            return_documents=False
        )

        # Sort results by Cohere relevance score
        reranked = []
        for item in response.results:
            result = results[item.index].copy()
            result['rerank_score'] = item.relevance_score
            reranked.append(result)

        logger.info(f"Cohere reranked {len(reranked)} results")
        return reranked

    def _rerank_crossencoder(
        self,
        question: str,
        results: List[dict]
    ) -> List[dict]:
        """Rerank using CrossEncoder."""
        # Create query-document pairs
        pairs = [(question, r['text']) for r in results]

        # Get reranking scores
        scores = self.reranker.predict(pairs)

        # Sort by reranking score
        reranked = sorted(
            zip(scores, results),
            key=lambda x: x[0],
            reverse=True
        )

        return [r for _, r in reranked]

    def _build_context(self, results: List[dict]) -> str:
        """Build context string from search results."""
        context_parts = []

        for i, result in enumerate(results, 1):
            pmid = result['metadata'].get('pmid', 'N/A')
            title = result['metadata'].get('title', 'Unknown Title')
            text = result['text']

            context_parts.append(
                f"[Paper {i}]\n"
                f"PMID: {pmid}\n"
                f"Title: {title}\n"
                f"Content: {text}\n"
            )

        return "\n---\n".join(context_parts)

    async def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using LLM."""
        try:
            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Based on the following research papers:\n\n{context}\n\nQuestion: {question}"
                    }
                ],
                temperature=0.1,
                max_tokens=1500
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return f"I encountered an error while generating the response. Please try again later."

    def _validate_response(
        self,
        answer: str,
        sources: List[dict]
    ) -> float:
        """
        Validate response and calculate confidence score.

        Checks:
        - Cited PMIDs exist in sources
        - Response doesn't contain hallucination indicators
        """
        # Extract cited PMIDs from answer
        cited_pmids = re.findall(r'PMID:\s*(\d+)', answer)

        # Get source PMIDs
        source_pmids = [
            s['metadata'].get('pmid', '')
            for s in sources
        ]

        # Check citation validity
        if cited_pmids:
            valid_citations = all(pmid in source_pmids for pmid in cited_pmids)
            citation_score = 1.0 if valid_citations else 0.5
        else:
            citation_score = 0.3  # No citations is concerning

        # Check for uncertainty indicators
        uncertainty_phrases = [
            "cannot find",
            "no information",
            "not enough",
            "unclear",
            "uncertain"
        ]
        has_uncertainty = any(phrase in answer.lower() for phrase in uncertainty_phrases)

        if has_uncertainty:
            return max(0.2, citation_score * 0.5)

        # Base confidence on citation validity and source relevance
        avg_relevance = sum(s['score'] for s in sources) / len(sources) if sources else 0
        confidence = (citation_score * 0.6) + (avg_relevance * 0.4)

        return round(min(confidence, 1.0), 2)

    def _format_sources(self, results: List[dict]) -> List[RAGSource]:
        """Format search results as source objects."""
        return [
            RAGSource(
                pmid=r['metadata'].get('pmid', ''),
                title=r['metadata'].get('title', 'Unknown'),
                relevance=round(r['score'], 3),
                excerpt=r['text'][:300] + "..." if len(r['text']) > 300 else r['text']
            )
            for r in results
        ]


# Singleton instance
_rag_service_instance: Optional[RAGService] = None


async def get_rag_service() -> RAGService:
    """Get or create singleton RAG service."""
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = RAGService()
    return _rag_service_instance
