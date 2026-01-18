"""Qdrant Vector Store for semantic search"""

import logging
from typing import List, Optional, Dict, Any
import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from src.core.config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Qdrant vector database wrapper for storing and searching embeddings.

    Provides:
    - Document storage with metadata
    - Semantic similarity search
    - Filtered search
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None
    ):
        """
        Initialize Qdrant client.

        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Collection name for documents
        """
        self.host = host or settings.QDRANT_HOST
        self.port = port or settings.QDRANT_PORT
        self.collection_name = collection_name or settings.QDRANT_COLLECTION
        self.vector_size = settings.EMBEDDING_DIMENSION

        # Initialize client
        self.client = QdrantClient(host=self.host, port=self.port)
        logger.info(f"Connected to Qdrant at {self.host}:{self.port}")

        # Ensure collection exists
        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection: {self.collection_name}")
        else:
            logger.info(f"Using existing collection: {self.collection_name}")

    def add_documents(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add documents with embeddings to the vector store.

        Args:
            texts: List of document texts
            embeddings: Embedding matrix (n_docs, embedding_dim)
            metadatas: Optional list of metadata dicts
            ids: Optional list of document IDs

        Returns:
            List of document IDs
        """
        if len(texts) != len(embeddings):
            raise ValueError("Number of texts must match number of embeddings")

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        # Prepare points
        points = []
        for i, (doc_id, text, embedding) in enumerate(zip(ids, texts, embeddings)):
            payload = {"text": text}
            if metadatas and i < len(metadatas):
                payload.update(metadatas[i])

            points.append(PointStruct(
                id=doc_id,
                vector=embedding.tolist(),
                payload=payload
            ))

        # Upsert points
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        logger.info(f"Added {len(points)} documents to {self.collection_name}")
        return ids

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional filter conditions
            score_threshold: Minimum similarity score

        Returns:
            List of results with text, score, and metadata
        """
        # Build filter if provided
        query_filter = None
        if filter_dict:
            must_conditions = []
            for key, value in filter_dict.items():
                if isinstance(value, list):
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchAny(any=value)
                        )
                    )
                else:
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                    )
            query_filter = models.Filter(must=must_conditions)

        # Execute search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            query_filter=query_filter,
            score_threshold=score_threshold
        )

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": str(result.id),
                "text": result.payload.get("text", ""),
                "score": result.score,
                "metadata": {
                    k: v for k, v in result.payload.items()
                    if k != "text"
                }
            })

        return formatted_results

    def delete(self, ids: List[str]) -> None:
        """
        Delete documents by IDs.

        Args:
            ids: List of document IDs to delete
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=ids)
        )
        logger.info(f"Deleted {len(ids)} documents from {self.collection_name}")

    def delete_by_filter(self, filter_dict: Dict[str, Any]) -> None:
        """
        Delete documents matching filter conditions.

        Args:
            filter_dict: Filter conditions
        """
        must_conditions = [
            models.FieldCondition(
                key=key,
                match=models.MatchValue(value=value)
            )
            for key, value in filter_dict.items()
        ]

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=must_conditions)
            )
        )
        logger.info(f"Deleted documents matching filter: {filter_dict}")

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status
        }

    def clear_collection(self) -> None:
        """Delete and recreate the collection."""
        self.client.delete_collection(self.collection_name)
        self._ensure_collection()
        logger.info(f"Cleared collection: {self.collection_name}")


# Singleton instance
_vector_store_instance: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get or create singleton vector store."""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance
