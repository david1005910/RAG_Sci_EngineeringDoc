"""Library API Endpoints - User's saved papers"""

import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from src.core.security import get_current_user_id
from src.data.library_store import library_store
from src.services.pubmed import get_pubmed_service
from src.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


# ============== Schemas ==============

class SavePaperRequest(BaseModel):
    """Save paper request"""
    pmid: str
    tags: List[str] = []
    notes: Optional[str] = None


class SavedPaper(BaseModel):
    """Saved paper"""
    id: str
    pmid: str
    title: str
    abstract: str
    authors: List[str] = []
    journal: str = ""
    tags: List[str] = []
    notes: Optional[str] = None
    saved_at: str


class SavedPaperListResponse(BaseModel):
    """Saved paper list response"""
    total: int
    papers: List[SavedPaper]


class TagListResponse(BaseModel):
    """Tag list response"""
    tags: List[str]


class UpdatePaperRequest(BaseModel):
    """Update saved paper request"""
    tags: Optional[List[str]] = None
    notes: Optional[str] = None


class CheckSavedResponse(BaseModel):
    """Check if paper is saved response"""
    is_saved: bool
    paper_id: Optional[str] = None


# ============== Endpoints ==============

@router.post("/papers", response_model=SavedPaper)
async def save_paper(
    request: SavePaperRequest,
    user_id: str = Depends(get_current_user_id)
):
    """
    Save a paper to user's library

    - Requires authentication
    - Fetches paper details from PubMed
    - Can add tags and notes
    """
    # Fetch paper details from PubMed
    title = "Unknown Title"
    abstract = "No abstract available"
    authors = []
    journal = ""

    try:
        pubmed = get_pubmed_service(api_key=settings.PUBMED_API_KEY)
        papers = await pubmed.fetch_papers([request.pmid])

        if papers:
            paper = papers[0]
            title = paper.title
            abstract = paper.abstract or "No abstract available"
            authors = paper.authors
            journal = paper.journal

    except Exception as e:
        logger.error(f"Failed to fetch paper {request.pmid}: {e}")

    # Save to library
    saved = library_store.save_paper(
        user_id=user_id,
        pmid=request.pmid,
        title=title,
        abstract=abstract,
        authors=authors,
        journal=journal,
        tags=request.tags,
        notes=request.notes
    )

    return SavedPaper(
        id=saved.id,
        pmid=saved.pmid,
        title=saved.title,
        abstract=saved.abstract,
        authors=saved.authors,
        journal=saved.journal,
        tags=saved.tags,
        notes=saved.notes,
        saved_at=saved.saved_at if isinstance(saved.saved_at, str) else saved.saved_at.isoformat()
    )


@router.get("/papers", response_model=SavedPaperListResponse)
async def get_saved_papers(
    user_id: str = Depends(get_current_user_id),
    tag: Optional[str] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0)
):
    """
    Get user's saved papers

    - Requires authentication
    - Supports filtering by tag
    - Supports pagination
    """
    total, papers = library_store.get_papers(
        user_id=user_id,
        tag=tag,
        limit=limit,
        offset=offset
    )

    return SavedPaperListResponse(
        total=total,
        papers=[
            SavedPaper(
                id=p.id,
                pmid=p.pmid,
                title=p.title,
                abstract=p.abstract,
                authors=p.authors,
                journal=p.journal,
                tags=p.tags,
                notes=p.notes,
                saved_at=p.saved_at if isinstance(p.saved_at, str) else p.saved_at.isoformat()
            )
            for p in papers
        ]
    )


@router.get("/papers/check/{pmid}", response_model=CheckSavedResponse)
async def check_paper_saved(
    pmid: str,
    user_id: str = Depends(get_current_user_id)
):
    """
    Check if a paper is already saved

    - Requires authentication
    """
    paper = library_store.get_paper_by_pmid(user_id, pmid)

    return CheckSavedResponse(
        is_saved=paper is not None,
        paper_id=paper.id if paper else None
    )


@router.get("/papers/{paper_id}", response_model=SavedPaper)
async def get_saved_paper(
    paper_id: str,
    user_id: str = Depends(get_current_user_id)
):
    """
    Get a specific saved paper

    - Requires authentication
    """
    paper = library_store.get_paper(user_id, paper_id)

    if not paper:
        raise HTTPException(
            status_code=404,
            detail=f"Saved paper {paper_id} not found"
        )

    return SavedPaper(
        id=paper.id,
        pmid=paper.pmid,
        title=paper.title,
        abstract=paper.abstract,
        authors=paper.authors,
        journal=paper.journal,
        tags=paper.tags,
        notes=paper.notes,
        saved_at=paper.saved_at if isinstance(paper.saved_at, str) else paper.saved_at.isoformat()
    )


@router.put("/papers/{paper_id}", response_model=SavedPaper)
async def update_saved_paper(
    paper_id: str,
    request: UpdatePaperRequest,
    user_id: str = Depends(get_current_user_id)
):
    """
    Update a saved paper (tags, notes)

    - Requires authentication
    """
    paper = library_store.update_paper(
        user_id=user_id,
        paper_id=paper_id,
        tags=request.tags,
        notes=request.notes
    )

    if not paper:
        raise HTTPException(
            status_code=404,
            detail=f"Saved paper {paper_id} not found"
        )

    return SavedPaper(
        id=paper.id,
        pmid=paper.pmid,
        title=paper.title,
        abstract=paper.abstract,
        authors=paper.authors,
        journal=paper.journal,
        tags=paper.tags,
        notes=paper.notes,
        saved_at=paper.saved_at if isinstance(paper.saved_at, str) else paper.saved_at.isoformat()
    )


@router.delete("/papers/{paper_id}")
async def delete_saved_paper(
    paper_id: str,
    user_id: str = Depends(get_current_user_id)
):
    """
    Remove a paper from library

    - Requires authentication
    """
    success = library_store.delete_paper(user_id, paper_id)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Saved paper {paper_id} not found"
        )

    return {"message": f"Paper {paper_id} removed from library"}


@router.get("/tags", response_model=TagListResponse)
async def get_tags(
    user_id: str = Depends(get_current_user_id)
):
    """
    Get all tags used by user

    - Requires authentication
    """
    tags = library_store.get_tags(user_id)

    return TagListResponse(tags=tags)
