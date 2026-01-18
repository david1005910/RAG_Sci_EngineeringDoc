"""Graph API Endpoints - Neo4j Knowledge Graph Operations"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import logging

from src.services.graph import get_neo4j_service, Neo4jService
from src.services.graph.neo4j_service import PaperNode

logger = logging.getLogger(__name__)
router = APIRouter()


# ============== Schemas ==============

class PaperInput(BaseModel):
    """Input for creating a paper in the graph"""
    pmid: str
    title: str
    abstract: str
    authors: List[str] = []
    keywords: List[str] = []
    mesh_terms: List[str] = []
    journal: Optional[str] = None
    publication_date: Optional[str] = None
    doi: Optional[str] = None


class PaperBatchInput(BaseModel):
    """Input for batch paper creation"""
    papers: List[PaperInput]


class CitationInput(BaseModel):
    """Input for creating citation relationship"""
    citing_pmid: str
    cited_pmid: str


class SimilarityInput(BaseModel):
    """Input for creating similarity relationship"""
    pmid1: str
    pmid2: str
    similarity_score: float
    similarity_type: str = "semantic"


class RelatedPaperResponse(BaseModel):
    """Response for related papers"""
    pmid: str
    title: str
    abstract: Optional[str] = None
    relationship_type: str
    score: float


class CoauthorResponse(BaseModel):
    """Response for coauthor"""
    name: str
    collaborations: int
    sample_papers: List[str]


class KeywordNetworkResponse(BaseModel):
    """Response for keyword network"""
    keyword: str
    related_keywords: List[dict]


class GraphStatsResponse(BaseModel):
    """Response for graph statistics"""
    papers: int
    authors: int
    keywords: int
    journals: int
    relationships: int
    status: str


# ============== Helper ==============

def get_service() -> Neo4jService:
    """Get Neo4j service with connection check"""
    try:
        service = get_neo4j_service()
        if not service.is_connected():
            raise HTTPException(
                status_code=503,
                detail="Neo4j database is not available. Please check your connection settings."
            )
        return service
    except Exception as e:
        logger.error(f"Neo4j service error: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Neo4j connection error: {str(e)}"
        )


# ============== Endpoints ==============

@router.get("/stats", response_model=GraphStatsResponse)
async def get_graph_stats():
    """
    Get graph database statistics

    Returns counts of nodes and relationships in the knowledge graph.
    """
    try:
        service = get_service()
        stats = service.get_stats()
        return GraphStatsResponse(**stats)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting graph stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/setup")
async def setup_schema():
    """
    Setup Neo4j schema with indexes and constraints

    Creates necessary indexes for optimal query performance.
    """
    try:
        service = get_service()
        service.setup_schema()
        return {"message": "Schema setup completed", "status": "success"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting up schema: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/papers")
async def create_paper(paper: PaperInput):
    """
    Create a paper node with all relationships

    Creates Paper node and connects it to Authors, Keywords, MeSH terms, and Journal.
    """
    try:
        service = get_service()

        paper_node = PaperNode(
            pmid=paper.pmid,
            title=paper.title,
            abstract=paper.abstract,
            journal=paper.journal,
            publication_date=paper.publication_date,
            doi=paper.doi
        )

        success = service.create_paper_with_relationships(
            paper=paper_node,
            authors=paper.authors,
            keywords=paper.keywords,
            mesh_terms=paper.mesh_terms
        )

        if success:
            return {
                "message": f"Paper {paper.pmid} created successfully",
                "pmid": paper.pmid,
                "authors_count": len(paper.authors),
                "keywords_count": len(paper.keywords)
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to create paper")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating paper: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/papers/batch")
async def create_papers_batch(batch: PaperBatchInput):
    """
    Create multiple papers in batch

    Creates multiple Paper nodes with their relationships.
    """
    try:
        service = get_service()
        created = 0
        errors = []

        for paper in batch.papers:
            try:
                paper_node = PaperNode(
                    pmid=paper.pmid,
                    title=paper.title,
                    abstract=paper.abstract,
                    journal=paper.journal,
                    publication_date=paper.publication_date,
                    doi=paper.doi
                )

                success = service.create_paper_with_relationships(
                    paper=paper_node,
                    authors=paper.authors,
                    keywords=paper.keywords,
                    mesh_terms=paper.mesh_terms
                )

                if success:
                    created += 1
                else:
                    errors.append(paper.pmid)
            except Exception as e:
                errors.append(f"{paper.pmid}: {str(e)}")

        return {
            "message": f"Batch processing completed",
            "created": created,
            "total": len(batch.papers),
            "errors": errors if errors else None
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch creation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/papers/{pmid}")
async def get_paper(pmid: str):
    """
    Get a paper with all its relationships

    Returns paper details including authors, keywords, and journal.
    """
    try:
        service = get_service()
        paper = service.get_paper(pmid)

        if not paper:
            raise HTTPException(status_code=404, detail=f"Paper {pmid} not found")

        return paper

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting paper: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/papers/{pmid}/related", response_model=List[RelatedPaperResponse])
async def get_related_papers(
    pmid: str,
    limit: int = Query(default=10, ge=1, le=50)
):
    """
    Find papers related to a given paper

    Finds related papers through shared authors, keywords, citations, or semantic similarity.
    """
    try:
        service = get_service()
        related = service.find_related_papers(pmid, limit=limit)

        return [RelatedPaperResponse(**p) for p in related]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding related papers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/citations")
async def create_citation(citation: CitationInput):
    """
    Create a citation relationship between papers

    Creates a CITES relationship from citing paper to cited paper.
    """
    try:
        service = get_service()
        success = service.create_citation(
            citing_pmid=citation.citing_pmid,
            cited_pmid=citation.cited_pmid
        )

        if success:
            return {
                "message": "Citation created",
                "citing": citation.citing_pmid,
                "cited": citation.cited_pmid
            }
        else:
            raise HTTPException(status_code=404, detail="One or both papers not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating citation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/similarity")
async def create_similarity(similarity: SimilarityInput):
    """
    Create a similarity relationship between papers

    Creates a SIMILAR_TO relationship with a similarity score.
    """
    try:
        service = get_service()
        success = service.create_similarity_relationship(
            pmid1=similarity.pmid1,
            pmid2=similarity.pmid2,
            similarity_score=similarity.similarity_score,
            similarity_type=similarity.similarity_type
        )

        if success:
            return {
                "message": "Similarity relationship created",
                "pmid1": similarity.pmid1,
                "pmid2": similarity.pmid2,
                "score": similarity.similarity_score
            }
        else:
            raise HTTPException(status_code=404, detail="One or both papers not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating similarity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/authors/{author_name}/coauthors", response_model=List[CoauthorResponse])
async def get_coauthors(
    author_name: str,
    limit: int = Query(default=20, ge=1, le=100)
):
    """
    Find co-authors of a given author

    Returns list of co-authors with collaboration count.
    """
    try:
        service = get_service()
        coauthors = service.find_coauthors(author_name, limit=limit)

        return [CoauthorResponse(**c) for c in coauthors]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding coauthors: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/authors/{author_name}/network")
async def get_author_network(author_name: str):
    """
    Get author collaboration network

    Returns the collaboration network for an author.
    """
    try:
        service = get_service()
        network = service.get_author_network(author_name)

        return network

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting author network: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/keywords/trending")
async def get_trending_keywords(
    limit: int = Query(default=20, ge=1, le=100)
):
    """
    Get trending keywords

    Returns the most connected keywords in the graph.
    """
    try:
        service = get_service()
        keywords = service.find_trending_keywords(limit=limit)

        return {"keywords": keywords}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting trending keywords: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/keywords/{keyword}/network", response_model=KeywordNetworkResponse)
async def get_keyword_network(
    keyword: str,
    limit: int = Query(default=20, ge=1, le=100)
):
    """
    Get keyword co-occurrence network

    Returns keywords that frequently co-occur with the given keyword.
    """
    try:
        service = get_service()
        network = service.get_keyword_network(keyword, limit=limit)

        return KeywordNetworkResponse(**network)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting keyword network: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/path")
async def find_path_between_papers(
    pmid1: str = Query(..., description="First paper PMID"),
    pmid2: str = Query(..., description="Second paper PMID"),
    max_depth: int = Query(default=4, ge=1, le=6)
):
    """
    Find shortest path between two papers

    Returns the shortest path connecting two papers through the knowledge graph.
    """
    try:
        service = get_service()
        path = service.find_path_between_papers(pmid1, pmid2, max_depth)

        if not path:
            return {"message": "No path found", "path": None}

        return path

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding path: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear")
async def clear_graph():
    """
    Clear the entire graph database

    WARNING: This will delete all nodes and relationships!
    """
    try:
        service = get_service()
        service.clear_database()

        return {"message": "Graph database cleared", "status": "success"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))
