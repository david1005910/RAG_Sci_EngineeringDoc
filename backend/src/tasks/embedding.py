"""Embedding generation tasks"""

import asyncio
import logging
from typing import List
import uuid

from .celery import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="src.tasks.embedding.process_paper_embeddings")
def process_paper_embeddings(paper_id: str) -> dict:
    """
    Generate embeddings for a paper and store in vector DB.

    Args:
        paper_id: Paper ID (UUID)

    Returns:
        Processing result
    """
    result = asyncio.run(_async_process_paper(paper_id))
    return result


async def _async_process_paper(paper_id: str) -> dict:
    """Async implementation of paper embedding processing."""
    from src.services.embedding.generator import EmbeddingGenerator
    from src.services.embedding.chunker import TextChunker
    from src.services.storage.vector_store import VectorStore

    try:
        # TODO: Fetch paper from database
        # For now, use mock data
        paper_data = {
            "id": paper_id,
            "pmid": "12345678",
            "title": "Sample Paper Title",
            "abstract": "This is a sample abstract for testing purposes."
        }

        # Initialize services
        chunker = TextChunker()
        embedding_generator = EmbeddingGenerator()
        vector_store = VectorStore()

        # Chunk the paper
        chunks = chunker.chunk_paper(
            title=paper_data["title"],
            abstract=paper_data["abstract"]
        )

        if not chunks:
            return {
                "paper_id": paper_id,
                "status": "skipped",
                "reason": "No chunks generated"
            }

        # Generate embeddings
        texts = [c.text for c in chunks]
        embeddings = embedding_generator.batch_encode(texts)

        # Prepare metadata
        metadatas = [
            {
                "pmid": paper_data["pmid"],
                "title": paper_data["title"],
                "section": c.section,
                "chunk_index": c.index
            }
            for c in chunks
        ]

        # Store in vector DB
        ids = [str(uuid.uuid4()) for _ in chunks]
        vector_store.add_documents(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        logger.info(f"Processed {len(chunks)} chunks for paper {paper_id}")

        return {
            "paper_id": paper_id,
            "chunks_created": len(chunks),
            "status": "completed"
        }

    except Exception as e:
        logger.error(f"Error processing paper {paper_id}: {e}")
        return {
            "paper_id": paper_id,
            "status": "error",
            "error": str(e)
        }


@celery_app.task(name="src.tasks.embedding.batch_process_embeddings")
def batch_process_embeddings(paper_ids: List[str]) -> dict:
    """
    Process embeddings for multiple papers.

    Args:
        paper_ids: List of paper IDs

    Returns:
        Batch processing results
    """
    results = []

    for paper_id in paper_ids:
        result = process_paper_embeddings.delay(paper_id)
        results.append({
            "paper_id": paper_id,
            "task_id": result.id
        })

    return {
        "papers_queued": len(paper_ids),
        "tasks": results
    }


@celery_app.task(name="src.tasks.embedding.reindex_all")
def reindex_all() -> dict:
    """
    Reindex all papers in the vector store.

    WARNING: This clears the existing index!
    """
    # TODO: Implement full reindex
    return {
        "status": "not_implemented",
        "message": "Full reindex not yet implemented"
    }
