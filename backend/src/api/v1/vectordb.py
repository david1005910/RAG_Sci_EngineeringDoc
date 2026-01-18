"""VectorDB API Endpoints for paper storage and retrieval

Implements Hybrid Search combining:
- Dense Search: Qdrant vector similarity using OpenAI embeddings
- Sparse Search: SPLADE-inspired keyword expansion with BM25
"""

import time
import logging
import uuid
import re
import math
from typing import List, Optional, Dict, Set, Tuple
from collections import Counter
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import aiohttp
import numpy as np

from src.core.config import settings
from src.data.vectordb_metadata_store import vectordb_metadata_store

logger = logging.getLogger(__name__)

router = APIRouter()


# ============== Schemas ==============

class PaperForVectorDB(BaseModel):
    """Paper data for vector storage"""
    pmid: str
    title: str
    abstract: str
    authors: List[str] = []
    journal: str = ""
    publication_date: Optional[str] = None
    keywords: List[str] = []


class SavePapersRequest(BaseModel):
    """Request to save multiple papers"""
    papers: List[PaperForVectorDB]


class SavePapersResponse(BaseModel):
    """Response after saving papers"""
    saved_count: int
    total_chunks: int
    processing_time_ms: int
    paper_ids: List[str]


class VectorDBStatsResponse(BaseModel):
    """Vector DB statistics"""
    collection_name: str
    vectors_count: int
    status: str
    search_mode: str
    dense_engine: Optional[str] = None
    sparse_engine: Optional[str] = None
    splade_indexed: Optional[bool] = None
    splade_vocab_size: Optional[int] = None
    with_embeddings: Optional[int] = None
    qdrant_status: Optional[str] = None


class SearchVectorDBRequest(BaseModel):
    """Request to search vector DB"""
    query: str
    top_k: int = 5
    search_mode: str = "hybrid"  # "dense", "sparse", or "hybrid"
    dense_weight: float = 0.7  # Weight for dense search in hybrid mode


class VectorSearchResult(BaseModel):
    """Single vector search result"""
    pmid: str
    title: str
    text: str
    score: float
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    section: str


class SearchVectorDBResponse(BaseModel):
    """Response from vector search"""
    results: List[VectorSearchResult]
    took_ms: int
    search_mode: str


# ============== SPLADE-inspired Sparse Search ==============

class SPLADESearch:
    """
    SPLADE-inspired sparse search with query expansion.

    Uses OpenAI to expand queries with related terms (similar to SPLADE's learned expansion)
    and BM25 for scoring.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs: Dict[str, int] = {}
        self.doc_lens: List[int] = []
        self.avg_doc_len: float = 0.0
        self.n_docs: int = 0
        self.idf: Dict[str, float] = {}
        self.term_weights: Dict[str, Dict[str, float]] = {}  # doc_id -> {term: weight}

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize with biomedical-aware preprocessing"""
        text = text.lower()
        # Keep hyphens in compound words (e.g., "CRISPR-Cas9")
        text = re.sub(r'[^\w\s\-]', ' ', text)
        tokens = text.split()
        return [t for t in tokens if len(t) > 1]

    async def _expand_query(self, query: str) -> List[Tuple[str, float]]:
        """
        Expand query using OpenAI (SPLADE-like expansion).
        Returns list of (term, weight) pairs.

        IMPORTANT: Original query terms get highest weight (2.0) to ensure relevance.
        """
        # Always include original query terms with highest weight
        original_tokens = self._tokenize(query)
        original_terms = [(t, 2.0) for t in original_tokens]  # High weight for original terms

        if not settings.OPENAI_API_KEY:
            return original_terms

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                }

                prompt = f"""Expand this biomedical search query with closely related scientific terms.
IMPORTANT: Focus on synonyms and directly related terms. Do NOT include unrelated concepts.

Query: "{query}"

Rules:
1. Include the EXACT original query terms first
2. Add only synonyms and closely related terms
3. Do NOT add tangentially related concepts
4. Maximum 15 terms total

Example:
Query: "CRISPR gene editing"
Output: CRISPR, gene, editing, Cas9, genome, nuclease, guide RNA, genetic, modification, CRISPR-Cas9, gene therapy

Your output (terms only, comma-separated):"""

                payload = {
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 100
                }

                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        expanded_text = data["choices"][0]["message"]["content"].strip()
                        # Parse expanded terms with decreasing weights
                        terms = [t.strip().lower() for t in expanded_text.split(",")]
                        weighted_terms = []

                        # Add original terms with highest weight first
                        seen_terms = set()
                        for term, weight in original_terms:
                            if term not in seen_terms:
                                weighted_terms.append((term, weight))
                                seen_terms.add(term)

                        # Add expanded terms with decreasing weights
                        for i, term in enumerate(terms):
                            term = term.strip().lower()
                            if term and term not in seen_terms:
                                # Weight decreases for expanded terms (max 1.0)
                                weight = 1.0 / (1 + 0.15 * i)
                                weighted_terms.append((term, weight))
                                seen_terms.add(term)

                        return weighted_terms[:20]  # Limit total terms
                    else:
                        return original_terms
        except Exception as e:
            logger.error(f"Query expansion error: {e}")
            return original_terms

    def fit(self, documents: List[str], doc_ids: List[str]):
        """Build SPLADE-style sparse index"""
        self.n_docs = len(documents)
        self.doc_lens = []
        self.doc_freqs = {}
        self.term_weights = {}

        for doc_id, doc in zip(doc_ids, documents):
            tokens = self._tokenize(doc)
            self.doc_lens.append(len(tokens))

            # Calculate term frequencies and weights
            term_freq = Counter(tokens)
            total_terms = len(tokens)

            self.term_weights[doc_id] = {}
            for term, freq in term_freq.items():
                # TF-IDF style weight (will be adjusted with IDF later)
                tf = freq / total_terms if total_terms > 0 else 0
                self.term_weights[doc_id][term] = tf

            # Update document frequencies
            unique_terms = set(tokens)
            for term in unique_terms:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        self.avg_doc_len = sum(self.doc_lens) / self.n_docs if self.n_docs > 0 else 0

        # Calculate IDF
        for term, df in self.doc_freqs.items():
            self.idf[term] = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)

    def score(
        self,
        query_terms: List[Tuple[str, float]],
        doc_idx: int,
        doc_text: str
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate SPLADE-style score with weighted query terms.
        Returns (total_score, term_scores_dict)
        """
        doc_tokens = self._tokenize(doc_text)
        doc_len = self.doc_lens[doc_idx] if doc_idx < len(self.doc_lens) else len(doc_tokens)
        term_freqs = Counter(doc_tokens)

        score = 0.0
        term_scores = {}

        for term, query_weight in query_terms:
            # Handle multi-word terms
            term_tokens = term.split()
            if len(term_tokens) > 1:
                # Check for phrase match
                doc_text_lower = doc_text.lower()
                if term in doc_text_lower:
                    phrase_score = query_weight * 2.0  # Boost for exact phrase match
                    score += phrase_score
                    term_scores[term] = phrase_score
                continue

            if term not in self.idf:
                continue

            tf = term_freqs.get(term, 0)
            if tf == 0:
                continue

            idf = self.idf[term]

            # BM25 with query term weight
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
            term_score = idf * (numerator / denominator) * query_weight if denominator > 0 else 0

            score += term_score
            term_scores[term] = term_score

        return score, term_scores

    async def search(
        self,
        query: str,
        documents: List[str],
        doc_ids: List[str],
        top_k: int = 5
    ) -> List[Dict]:
        """Search with SPLADE-style query expansion"""
        # Expand query
        query_terms = await self._expand_query(query)
        logger.info(f"SPLADE expanded query: {[t[0] for t in query_terms[:5]]}...")

        results = []
        for idx, (doc, doc_id) in enumerate(zip(documents, doc_ids)):
            score, term_scores = self.score(query_terms, idx, doc)
            if score > 0:
                results.append({
                    "idx": idx,
                    "doc_id": doc_id,
                    "score": score,
                    "matched_terms": list(term_scores.keys())[:5]
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


# ============== Qdrant-based Dense Search ==============

class QdrantDenseSearch:
    """
    Dense vector search using Qdrant vector database.
    Supports three modes:
    1. Server mode: Connect to Qdrant server (Docker/standalone)
    2. Local mode: Use local file-based storage (no Docker required)
    3. In-memory mode: Fallback if Qdrant is not available
    """
    def __init__(self):
        self.qdrant_client = None
        self.collection_name = settings.QDRANT_COLLECTION or "biomedical_papers"
        self.embedding_dim = 1536
        self.use_qdrant = False
        self.qdrant_mode = "none"  # "server", "local", or "none"
        self._init_qdrant()

    def _init_qdrant(self):
        """Initialize Qdrant client - try server first, then local mode"""
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import Distance, VectorParams

        # Try 1: Connect to Qdrant server (Docker)
        try:
            self.qdrant_client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                timeout=5.0
            )
            # Test connection
            self.qdrant_client.get_collections()
            self._ensure_collection()
            self.use_qdrant = True
            self.qdrant_mode = "server"
            logger.info(f"Connected to Qdrant server at {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
            return
        except Exception as e:
            logger.info(f"Qdrant server not available: {e}")

        # Try 2: Use local file-based Qdrant
        try:
            import os
            qdrant_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "qdrant")
            os.makedirs(qdrant_path, exist_ok=True)

            self.qdrant_client = QdrantClient(path=qdrant_path)
            self._ensure_collection()
            self.use_qdrant = True
            self.qdrant_mode = "local"
            logger.info(f"Using Qdrant local storage at {qdrant_path}")
            return
        except Exception as e:
            logger.warning(f"Qdrant local mode failed: {e}")

        # Fallback: In-memory mode
        logger.warning("Qdrant not available, using in-memory search")
        self.use_qdrant = False
        self.qdrant_mode = "none"

    def _ensure_collection(self):
        """Ensure collection exists"""
        from qdrant_client.http.models import Distance, VectorParams

        collections = self.qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created Qdrant collection: {self.collection_name}")

    def get_collection_info(self) -> Dict:
        """Get Qdrant collection info"""
        if not self.use_qdrant:
            return {"status": "in_memory", "vectors_count": 0, "mode": "none"}

        try:
            info = self.qdrant_client.get_collection(self.collection_name)
            return {
                "status": f"qdrant_{self.qdrant_mode}",
                "vectors_count": info.vectors_count if hasattr(info, 'vectors_count') else 0,
                "points_count": info.points_count if hasattr(info, 'points_count') else 0,
                "mode": self.qdrant_mode
            }
        except Exception as e:
            logger.debug(f"get_collection_info error: {e}")
            # If we're in local mode and Qdrant is working, return success status
            if self.qdrant_mode == "local":
                return {"status": f"qdrant_local", "vectors_count": 0, "mode": self.qdrant_mode}
            return {"status": "error", "error": str(e), "mode": self.qdrant_mode}


# ============== Hybrid Vector Store ==============

class HybridVectorStore:
    """
    Hybrid vector store combining:
    - Dense Search: Qdrant vector similarity (OpenAI embeddings)
    - Sparse Search: SPLADE-inspired keyword expansion with BM25

    Provides hybrid scores showing both dense and sparse contributions.
    """
    def __init__(self):
        self.documents = []  # List of {id, text, embedding, metadata}
        self.embedding_dim = 1536  # OpenAI text-embedding-ada-002
        self.splade = SPLADESearch()  # SPLADE for sparse search
        self.qdrant_dense = QdrantDenseSearch()  # Qdrant for dense search
        self._splade_fitted = False
        # Auto-sync from Qdrant on startup
        self._sync_from_qdrant()

    def _sync_from_qdrant(self):
        """Sync documents from Qdrant on startup"""
        if not self.qdrant_dense.use_qdrant:
            logger.info("Qdrant not available, skipping sync")
            return

        try:
            # Get all points from Qdrant
            scroll_result = self.qdrant_dense.qdrant_client.scroll(
                collection_name=self.qdrant_dense.collection_name,
                limit=10000,  # Adjust based on expected max documents
                with_payload=True,
                with_vectors=True
            )

            points, _ = scroll_result
            if not points:
                logger.info("No documents in Qdrant to sync")
                return

            # Rebuild local documents list
            self.documents = []
            for point in points:
                self.documents.append({
                    "id": str(point.id),
                    "text": point.payload.get("text", ""),
                    "embedding": np.array(point.vector) if point.vector else None,
                    "metadata": {k: v for k, v in point.payload.items() if k != "text"}
                })

            # Rebuild SPLADE index
            self._rebuild_sparse_index()
            logger.info(f"Synced {len(self.documents)} documents from Qdrant")

        except Exception as e:
            logger.error(f"Failed to sync from Qdrant: {e}")

    def _rebuild_sparse_index(self):
        """Rebuild SPLADE sparse index from current documents"""
        if self.documents:
            texts = [doc["text"] for doc in self.documents]
            doc_ids = [doc["id"] for doc in self.documents]
            self.splade.fit(texts, doc_ids)
            self._splade_fitted = True
            logger.info(f"SPLADE index rebuilt with {len(texts)} documents")

    async def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding using OpenAI API"""
        if not settings.OPENAI_API_KEY:
            return None

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "text-embedding-ada-002",
                    "input": text[:8000]
                }

                async with session.post(
                    "https://api.openai.com/v1/embeddings",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return np.array(data["data"][0]["embedding"])
                    else:
                        logger.error(f"Embedding API error: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return None

    async def get_batch_embeddings(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """Get embeddings for multiple texts"""
        if not settings.OPENAI_API_KEY:
            return [None] * len(texts)

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                }
                truncated_texts = [t[:8000] for t in texts]
                payload = {
                    "model": "text-embedding-ada-002",
                    "input": truncated_texts
                }

                async with session.post(
                    "https://api.openai.com/v1/embeddings",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [np.array(d["embedding"]) for d in data["data"]]
                    else:
                        logger.error(f"Batch embedding API error: {response.status}")
                        return [None] * len(texts)
        except Exception as e:
            logger.error(f"Batch embedding error: {e}")
            return [None] * len(texts)

    async def add_documents(
        self,
        texts: List[str],
        metadatas: List[dict],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add documents with both Dense (Qdrant) and Sparse (SPLADE) indexing"""
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        # Get embeddings in batch (Dense vectors)
        embeddings = await self.get_batch_embeddings(texts)

        # Add to Qdrant if available
        if self.qdrant_dense.use_qdrant:
            try:
                from qdrant_client.http.models import PointStruct

                points = []
                for doc_id, text, embedding, metadata in zip(ids, texts, embeddings, metadatas):
                    if embedding is not None:
                        payload = {"text": text, **metadata}
                        points.append(PointStruct(
                            id=doc_id,
                            vector=embedding.tolist(),
                            payload=payload
                        ))

                if points:
                    self.qdrant_dense.qdrant_client.upsert(
                        collection_name=self.qdrant_dense.collection_name,
                        points=points
                    )
                    logger.info(f"Added {len(points)} vectors to Qdrant")
            except Exception as e:
                logger.error(f"Qdrant upsert error: {e}")

        # Store locally for fallback and SPLADE
        for doc_id, text, embedding, metadata in zip(ids, texts, embeddings, metadatas):
            self.documents.append({
                "id": doc_id,
                "text": text,
                "embedding": embedding,
                "metadata": metadata
            })

        # Rebuild SPLADE sparse index
        self._rebuild_sparse_index()

        logger.info(f"Added {len(texts)} documents with Qdrant dense + SPLADE sparse indexing")
        return ids

    def _normalize_scores(self, scores: List[float], target_min: float = 0.0, target_max: float = 1.0) -> List[float]:
        """Min-max normalization of scores to [target_min, target_max]"""
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [target_max] * len(scores)
        # Normalize to [0, 1] first, then scale to target range
        normalized = [(s - min_score) / (max_score - min_score) for s in scores]
        return [target_min + n * (target_max - target_min) for n in normalized]

    def _normalize_dense_score(self, score: float) -> float:
        """Normalize dense score to [0, 1] range"""
        # Cosine similarity is already in [-1, 1], we convert to [0, 1]
        return max(0.0, min(1.0, (score + 1) / 2 if score < 0 else score))

    def _normalize_sparse_score(self, score: float, max_possible: float = 30.0) -> float:
        """Normalize sparse score to [0, 30] range using logarithmic scaling"""
        if score <= 0:
            return 0.0
        # Use log scaling for better distribution, cap at max_possible
        import math
        # Scale factor to map typical BM25 scores (0-10) to 0-30 range
        scaled = min(score * 3.0, max_possible)
        return round(scaled, 2)

    async def search_dense(self, query: str, top_k: int = 5) -> List[dict]:
        """Dense search using Qdrant (falls back to in-memory if unavailable)"""
        if not self.documents:
            return []

        query_embedding = await self.get_embedding(query)
        if query_embedding is None:
            return []

        results = []

        # Try Qdrant first
        if self.qdrant_dense.use_qdrant:
            try:
                # Use query_points API (Qdrant 1.7+)
                qdrant_response = self.qdrant_dense.qdrant_client.query_points(
                    collection_name=self.qdrant_dense.collection_name,
                    query=query_embedding.tolist(),
                    limit=top_k,
                    with_payload=True
                )

                for r in qdrant_response.points:
                    # Normalize dense score to [0, 1]
                    normalized_dense = self._normalize_dense_score(r.score)
                    results.append({
                        "id": str(r.id),
                        "text": r.payload.get("text", ""),
                        "dense_score": round(normalized_dense, 4),
                        "sparse_score": None,
                        "score": round(normalized_dense, 4),
                        "metadata": {k: v for k, v in r.payload.items() if k != "text"},
                        "search_engine": "qdrant"
                    })

                logger.info(f"Qdrant dense search returned {len(results)} results")
                return results
            except Exception as e:
                logger.warning(f"Qdrant search failed, falling back to in-memory: {e}")

        # Fallback to in-memory search
        for doc in self.documents:
            if doc["embedding"] is not None:
                score = float(np.dot(query_embedding, doc["embedding"]) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc["embedding"])
                ))
                # Normalize dense score to [0, 1]
                normalized_dense = self._normalize_dense_score(score)
                results.append({
                    "id": doc["id"],
                    "text": doc["text"],
                    "dense_score": round(normalized_dense, 4),
                    "sparse_score": None,
                    "score": round(normalized_dense, 4),
                    "metadata": doc["metadata"],
                    "search_engine": "in_memory"
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    async def search_sparse(self, query: str, top_k: int = 5) -> List[dict]:
        """Sparse search using SPLADE-inspired query expansion"""
        if not self.documents or not self._splade_fitted:
            return []

        texts = [doc["text"] for doc in self.documents]
        doc_ids = [doc["id"] for doc in self.documents]

        # Use SPLADE search with query expansion
        splade_results = await self.splade.search(query, texts, doc_ids, top_k=len(texts))

        results = []
        for r in splade_results:
            doc = self.documents[r["idx"]]
            # Normalize sparse score to [0, 30]
            normalized_sparse = self._normalize_sparse_score(r["score"])
            results.append({
                "id": doc["id"],
                "text": doc["text"],
                "dense_score": None,
                "sparse_score": normalized_sparse,
                "score": normalized_sparse,
                "metadata": doc["metadata"],
                "matched_terms": r.get("matched_terms", []),
                "search_engine": "splade"
            })

        return results[:top_k]

    async def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        dense_weight: float = 0.7
    ) -> List[dict]:
        """
        Hybrid search combining Qdrant dense + SPLADE sparse using weighted average.

        Score formula: hybrid_score = (dense_score * 0.7) + (sparse_score_normalized * 0.3)

        Features:
        - Dense weight: 0.7 (semantic similarity)
        - Sparse weight: 0.3 (keyword matching)
        - Synergy boost: +0.1 when document appears in both search results
        - Query expansion: Original words weight 2.0, expanded words <= 1.0

        Score ranges:
        - Dense score: [0, 1] (cosine similarity)
        - Sparse score: [0, 1] (normalized from BM25)
        - Hybrid score: [0, 1] (weighted combination with synergy boost)
        """
        if not self.documents:
            return []

        sparse_weight = 1.0 - dense_weight  # 0.3

        # Get dense results
        dense_results = await self.search_dense(query, top_k=len(self.documents))
        dense_scores_map = {r["id"]: r["dense_score"] for r in dense_results}

        # Get sparse results
        sparse_results = await self.search_sparse(query, top_k=len(self.documents))
        sparse_scores_map = {r["id"]: r["sparse_score"] for r in sparse_results}
        matched_terms_map = {r["id"]: r.get("matched_terms", []) for r in sparse_results}

        # Normalize sparse scores to [0, 1] for weighted combination
        max_sparse = max(sparse_scores_map.values()) if sparse_scores_map else 1.0
        sparse_scores_normalized = {}
        for doc_id, score in sparse_scores_map.items():
            sparse_scores_normalized[doc_id] = score / max_sparse if max_sparse > 0 else 0.0

        # Combine scores for all documents
        all_doc_ids = set(dense_scores_map.keys()) | set(sparse_scores_map.keys())
        combined_results = []

        for doc_id in all_doc_ids:
            # Get document
            doc = next((d for d in self.documents if d["id"] == doc_id), None)
            if doc is None:
                dense_r = next((r for r in dense_results if r["id"] == doc_id), None)
                if dense_r:
                    doc = {"id": doc_id, "text": dense_r["text"], "metadata": dense_r["metadata"]}
                else:
                    continue

            # Get scores
            d_score = dense_scores_map.get(doc_id, 0.0)  # [0, 1]
            s_score_raw = sparse_scores_map.get(doc_id, 0.0)  # Original sparse score
            s_score_norm = sparse_scores_normalized.get(doc_id, 0.0)  # [0, 1]

            # Weighted combination: dense * 0.7 + sparse * 0.3
            hybrid_score = (d_score * dense_weight) + (s_score_norm * sparse_weight)

            # Synergy boost: +0.1 when document appears in BOTH search results
            has_dense = doc_id in dense_scores_map and d_score > 0
            has_sparse = doc_id in sparse_scores_map and s_score_raw > 0
            if has_dense and has_sparse:
                # Boost proportional to the quality of both scores
                synergy_boost = 0.1 * min(d_score, s_score_norm)
                hybrid_score = min(hybrid_score + synergy_boost, 1.0)

            combined_results.append({
                "id": doc_id,
                "text": doc["text"],
                "dense_score": round(d_score, 4),  # [0, 1]
                "sparse_score": round(s_score_raw, 2),  # Original scale for display
                "score": round(hybrid_score, 4),  # [0, 1] hybrid score
                "metadata": doc["metadata"],
                "matched_terms": matched_terms_map.get(doc_id, []),
                "search_engine": "hybrid"
            })

        # Sort by hybrid score
        combined_results.sort(key=lambda x: x["score"], reverse=True)

        logger.info(
            f"Hybrid search: {len(dense_results)} dense + {len(sparse_results)} sparse "
            f"-> top score: {combined_results[0]['score'] if combined_results else 0}"
        )

        return combined_results[:top_k]

    async def search(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        dense_weight: float = 0.7
    ) -> List[dict]:
        """
        Search documents using specified mode.

        Args:
            query: Search query
            top_k: Number of results to return
            mode: "dense" (Qdrant), "sparse" (SPLADE), or "hybrid" (both)
            dense_weight: Weight for dense search in hybrid mode (0-1)
        """
        if mode == "dense":
            return await self.search_dense(query, top_k)
        elif mode == "sparse":
            return await self.search_sparse(query, top_k)
        else:  # hybrid
            return await self.search_hybrid(query, top_k, dense_weight)

    def get_stats(self) -> dict:
        """Get store statistics"""
        has_embeddings = sum(1 for d in self.documents if d["embedding"] is not None)
        qdrant_info = self.qdrant_dense.get_collection_info()
        qdrant_mode = self.qdrant_dense.qdrant_mode

        # Determine actual search mode based on availability
        if self.qdrant_dense.use_qdrant:
            dense_engine = f"qdrant_{qdrant_mode}"
        else:
            dense_engine = "in_memory"
        sparse_engine = "splade" if self._splade_fitted else "none"

        if self.qdrant_dense.use_qdrant:
            search_mode = f"hybrid (Qdrant {qdrant_mode} + SPLADE sparse)"
        else:
            search_mode = f"hybrid (In-memory dense + SPLADE sparse)"

        return {
            "collection_name": "biomedical_papers",
            "vectors_count": len(self.documents),
            "with_embeddings": has_embeddings,
            "splade_indexed": self._splade_fitted,
            "splade_vocab_size": len(self.splade.idf) if self._splade_fitted else 0,
            "qdrant_status": qdrant_info.get("status", "unknown"),
            "qdrant_vectors": qdrant_info.get("vectors_count", 0),
            "qdrant_mode": qdrant_mode,
            "dense_engine": dense_engine,
            "sparse_engine": sparse_engine,
            "search_mode": search_mode,
            "status": "ready"
        }

    def get_papers(self) -> List[dict]:
        """Get all papers stored in VectorDB"""
        papers = {}
        for doc in self.documents:
            metadata = doc.get("metadata", {})
            pmid = metadata.get("pmid", "")
            if pmid and pmid not in papers:
                # Handle authors - can be list or comma-separated string
                authors = metadata.get("authors", [])
                if isinstance(authors, str):
                    authors = [a.strip() for a in authors.split(",") if a.strip()]

                # Handle keywords - can be list or comma-separated string
                keywords = metadata.get("keywords", [])
                if isinstance(keywords, str):
                    keywords = [k.strip() for k in keywords.split(",") if k.strip()]

                papers[pmid] = {
                    "id": doc["id"],
                    "pmid": pmid,
                    "title": metadata.get("title", "Untitled"),
                    "abstract": doc.get("text", "")[:500],
                    "journal": metadata.get("journal", ""),
                    "authors": authors,
                    "keywords": keywords,
                    "indexed_at": metadata.get("indexed_at", ""),
                    "section": metadata.get("section", "abstract")
                }
        return list(papers.values())

    def clear(self):
        """Clear all documents"""
        self.documents = []
        self._splade_fitted = False
        self.splade = SPLADESearch()

        # Clear Qdrant collection if available
        if self.qdrant_dense.use_qdrant:
            try:
                from qdrant_client.http.models import Distance, VectorParams

                self.qdrant_dense.qdrant_client.delete_collection(
                    self.qdrant_dense.collection_name
                )
                self.qdrant_dense.qdrant_client.create_collection(
                    collection_name=self.qdrant_dense.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                logger.info("Cleared Qdrant collection")
            except Exception as e:
                logger.error(f"Failed to clear Qdrant: {e}")


# Singleton instance
_vector_store: Optional[HybridVectorStore] = None


def get_vector_store() -> HybridVectorStore:
    """Get or create vector store instance"""
    global _vector_store
    if _vector_store is None:
        _vector_store = HybridVectorStore()
    return _vector_store


# ============== Text Chunking ==============

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks"""
    if not text:
        return []

    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
        if start >= len(words) - overlap:
            break

    return chunks


# ============== Endpoints ==============

@router.post("/papers/save", response_model=SavePapersResponse)
async def save_papers_to_vectordb(request: SavePapersRequest):
    """
    Save papers to vector database

    - Chunks paper text (title + abstract)
    - Generates embeddings using OpenAI
    - Stores in vector database
    - Returns count of saved papers and chunks
    """
    start_time = time.time()

    vector_store = get_vector_store()

    all_texts = []
    all_metadatas = []
    paper_ids = []

    for paper in request.papers:
        # Combine title and abstract
        full_text = f"{paper.title}. {paper.abstract}" if paper.abstract else paper.title

        # Chunk the text
        chunks = chunk_text(full_text)

        for i, chunk in enumerate(chunks):
            all_texts.append(chunk)
            all_metadatas.append({
                "pmid": paper.pmid,
                "title": paper.title,
                "journal": paper.journal,
                "publication_date": paper.publication_date or "",
                "section": "abstract",
                "chunk_index": i,
                "authors": ", ".join(paper.authors[:3]) if paper.authors else "",
                "keywords": ", ".join(paper.keywords[:5]) if paper.keywords else ""
            })

        paper_ids.append(paper.pmid)

    # Save to vector store
    if all_texts:
        await vector_store.add_documents(
            texts=all_texts,
            metadatas=all_metadatas
        )

    # Save full metadata to metadata store (with complete abstract and all authors)
    metadata_papers = [
        {
            "pmid": paper.pmid,
            "title": paper.title,
            "abstract": paper.abstract,
            "authors": paper.authors,
            "journal": paper.journal,
            "publication_date": paper.publication_date,
            "keywords": paper.keywords
        }
        for paper in request.papers
    ]
    vectordb_metadata_store.save_papers_batch(metadata_papers)

    took_ms = int((time.time() - start_time) * 1000)

    logger.info(f"Saved {len(request.papers)} papers ({len(all_texts)} chunks) to VectorDB + metadata in {took_ms}ms")

    return SavePapersResponse(
        saved_count=len(request.papers),
        total_chunks=len(all_texts),
        processing_time_ms=took_ms,
        paper_ids=paper_ids
    )


@router.get("/stats", response_model=VectorDBStatsResponse)
async def get_vectordb_stats():
    """
    Get vector database statistics

    - Returns collection info
    - Shows number of stored vectors
    - Indicates search mode (hybrid/dense/sparse)
    - Shows SPLADE indexing status
    """
    vector_store = get_vector_store()
    stats = vector_store.get_stats()

    return VectorDBStatsResponse(
        collection_name=stats["collection_name"],
        vectors_count=stats["vectors_count"],
        status=stats["status"],
        search_mode=stats["search_mode"],
        dense_engine=stats.get("dense_engine"),
        sparse_engine=stats.get("sparse_engine"),
        splade_indexed=stats.get("splade_indexed"),
        splade_vocab_size=stats.get("splade_vocab_size"),
        with_embeddings=stats.get("with_embeddings"),
        qdrant_status=stats.get("qdrant_status")
    )


class VectorDBPaper(BaseModel):
    """Paper stored in VectorDB"""
    id: str
    pmid: str
    title: str
    abstract: str
    journal: Optional[str] = None
    authors: List[str] = []
    keywords: List[str] = []
    indexed_at: Optional[str] = None


class VectorDBPapersResponse(BaseModel):
    """Response for listing VectorDB papers"""
    papers: List[VectorDBPaper]
    total: int


@router.get("/papers", response_model=VectorDBPapersResponse)
async def get_vectordb_papers():
    """
    Get all papers stored in VectorDB

    - Returns list of indexed papers with FULL metadata
    - Uses metadata store for complete abstract and all authors
    - Falls back to vector store if metadata not available
    - Used for library page to show indexed papers
    """
    # Get full metadata from metadata store
    metadata_papers = vectordb_metadata_store.get_all_papers()

    # Also get papers from vector store (for any not in metadata)
    vector_store = get_vector_store()
    vector_papers = vector_store.get_papers()

    # Create a merged dictionary (metadata store takes priority)
    papers_dict = {}

    # First add papers from vector store (as fallback)
    for p in vector_papers:
        pmid = p.get("pmid", "")
        if pmid:
            papers_dict[pmid] = {
                "id": p.get("id", pmid),
                "pmid": pmid,
                "title": p.get("title", "Untitled"),
                "abstract": p.get("abstract", ""),
                "journal": p.get("journal", ""),
                "authors": p.get("authors", []),
                "keywords": p.get("keywords", []),
                "indexed_at": p.get("indexed_at", "")
            }

    # Override with full metadata from metadata store
    for p in metadata_papers:
        pmid = p.get("pmid", "")
        if pmid:
            papers_dict[pmid] = {
                "id": pmid,  # Use pmid as id
                "pmid": pmid,
                "title": p.get("title", "Untitled"),
                "abstract": p.get("abstract", ""),  # Full abstract
                "journal": p.get("journal", ""),
                "authors": p.get("authors", []),  # All authors
                "keywords": p.get("keywords", []),
                "indexed_at": p.get("indexed_at", "")
            }

    papers = list(papers_dict.values())

    return VectorDBPapersResponse(
        papers=[VectorDBPaper(**p) for p in papers],
        total=len(papers)
    )


@router.get("/metadata", response_model=VectorDBPapersResponse)
async def get_vectordb_metadata():
    """
    Get all papers from metadata store (full metadata)

    - Returns papers with COMPLETE abstract and ALL authors
    - Data is auto-saved when papers are indexed to VectorDB
    - Used for "저장된 Meta 데이터" tab
    """
    metadata_papers = vectordb_metadata_store.get_all_papers()

    papers = []
    for p in metadata_papers:
        pmid = p.get("pmid", "")
        if pmid:
            papers.append({
                "id": pmid,
                "pmid": pmid,
                "title": p.get("title", "Untitled"),
                "abstract": p.get("abstract", ""),
                "journal": p.get("journal", ""),
                "authors": p.get("authors", []),
                "keywords": p.get("keywords", []),
                "indexed_at": p.get("indexed_at", "")
            })

    return VectorDBPapersResponse(
        papers=[VectorDBPaper(**p) for p in papers],
        total=len(papers)
    )


@router.post("/search", response_model=SearchVectorDBResponse)
async def search_vectordb(request: SearchVectorDBRequest):
    """
    Hybrid search in vector database

    - Supports three search modes:
      - "hybrid": Combined dense (semantic) + sparse (BM25) search (default)
      - "dense": Embedding-based semantic search only
      - "sparse": BM25 keyword-based search only
    - dense_weight: Weight for dense search in hybrid mode (0-1, default 0.7)
    - Returns top-k most similar papers with both dense and sparse scores
    """
    start_time = time.time()

    vector_store = get_vector_store()
    results = await vector_store.search(
        query=request.query,
        top_k=request.top_k,
        mode=request.search_mode,
        dense_weight=request.dense_weight
    )

    took_ms = int((time.time() - start_time) * 1000)

    formatted_results = [
        VectorSearchResult(
            pmid=r["metadata"].get("pmid", ""),
            title=r["metadata"].get("title", ""),
            text=r["text"][:300] + "..." if len(r["text"]) > 300 else r["text"],
            score=r["score"],
            dense_score=r.get("dense_score"),
            sparse_score=r.get("sparse_score"),
            section=r["metadata"].get("section", "")
        )
        for r in results
    ]

    return SearchVectorDBResponse(
        results=formatted_results,
        took_ms=took_ms,
        search_mode=request.search_mode
    )


@router.delete("/clear")
async def clear_vectordb():
    """
    Clear all documents from vector database

    WARNING: This deletes all stored papers!
    """
    vector_store = get_vector_store()
    vector_store.clear()

    return {"message": "Vector database cleared", "status": "success"}
