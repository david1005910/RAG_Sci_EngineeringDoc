"""Search API Endpoints"""

import time
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from fastapi.responses import Response

from src.data import sample_papers
from src.services.pubmed import get_pubmed_service, PubMedPaper
from src.services.pmc import get_pmc_service, PMCPaperInfo
from src.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


# ============== Schemas ==============

class SearchFilters(BaseModel):
    """Search filters"""
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    journals: Optional[List[str]] = None
    authors: Optional[List[str]] = None


class SearchRequest(BaseModel):
    """Search request"""
    query: str
    filters: Optional[SearchFilters] = None
    limit: int = 10
    offset: int = 0
    source: str = "pubmed"  # "pubmed" for real API, "mock" for sample data


class PaperResult(BaseModel):
    """Paper search result"""
    pmid: str
    title: str
    abstract: str
    relevance_score: float
    authors: List[str] = []
    journal: str = ""
    publication_date: Optional[str] = None
    keywords: List[str] = []


class SearchResponse(BaseModel):
    """Search response"""
    total: int
    took_ms: int
    results: List[PaperResult]


class PaperDetailResponse(BaseModel):
    """Paper detail response"""
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    journal: str
    publication_date: Optional[str] = None
    doi: Optional[str] = None
    keywords: List[str] = []
    mesh_terms: List[str] = []


class SimilarPaperResponse(BaseModel):
    """Similar paper response"""
    pmid: str
    title: str
    similarity_score: float
    common_keywords: List[str] = []


class PDFInfoResponse(BaseModel):
    """PDF availability info response"""
    pmid: str
    pmcid: Optional[str] = None
    has_pdf: bool
    pdf_url: Optional[str] = None
    is_open_access: bool


# ============== Endpoints ==============

@router.post("/search", response_model=SearchResponse)
async def search_papers_endpoint(request: SearchRequest):
    """
    Semantic search for papers

    - source="pubmed": Real PubMed API search
    - source="mock": Sample data for testing
    - Supports filters for year, journal, authors
    - Returns top-K most relevant papers
    """
    start_time = time.time()

    if request.source == "pubmed":
        # Use real PubMed API
        try:
            pubmed = get_pubmed_service(
                api_key=settings.PUBMED_API_KEY,
                email="bio-rag@example.com"
            )

            # Apply filters if provided
            filter_params = {}
            if request.filters:
                if request.filters.year_from:
                    filter_params["year_from"] = request.filters.year_from
                if request.filters.year_to:
                    filter_params["year_to"] = request.filters.year_to
                if request.filters.journals:
                    filter_params["journals"] = request.filters.journals
                if request.filters.authors:
                    filter_params["authors"] = request.filters.authors

            total, papers = await pubmed.search_and_fetch(
                query=request.query,
                max_results=request.limit,
                sort="relevance",
                **filter_params
            )

            took_ms = int((time.time() - start_time) * 1000)

            # Convert PubMedPaper to PaperResult
            results = []
            for i, paper in enumerate(papers):
                # Calculate relevance score (decreasing by position)
                relevance = 1.0 - (i * 0.05) if i < 20 else 0.05
                results.append(PaperResult(
                    pmid=paper.pmid,
                    title=paper.title,
                    abstract=paper.abstract or "No abstract available",
                    relevance_score=relevance,
                    authors=paper.authors,
                    journal=paper.journal,
                    publication_date=paper.publication_date,
                    keywords=paper.keywords + paper.mesh_terms[:5]
                ))

            logger.info(f"PubMed search '{request.query}': {total} total, {len(results)} returned in {took_ms}ms")

            return SearchResponse(
                total=total,
                took_ms=took_ms,
                results=results
            )

        except Exception as e:
            logger.error(f"PubMed search error: {e}")
            # Fallback to mock data on error
            pass

    # Use mock/sample data
    filters = None
    if request.filters:
        filters = request.filters.model_dump()

    total, results = sample_papers.search_papers(
        query=request.query,
        limit=request.limit,
        offset=request.offset,
        filters=filters
    )

    took_ms = int((time.time() - start_time) * 1000)

    return SearchResponse(
        total=total,
        took_ms=took_ms,
        results=[PaperResult(**r) for r in results]
    )


@router.get("/papers/{pmid}", response_model=PaperDetailResponse)
async def get_paper(pmid: str):
    """
    Get paper details by PMID

    - First checks sample data
    - Falls back to PubMed API for real papers
    - Returns full paper metadata
    """
    # Try sample data first
    paper = sample_papers.get_paper_by_pmid(pmid)

    if paper:
        return PaperDetailResponse(**paper)

    # Fetch from PubMed
    try:
        pubmed = get_pubmed_service(api_key=settings.PUBMED_API_KEY)
        papers = await pubmed.fetch_papers([pmid])

        if papers:
            p = papers[0]
            return PaperDetailResponse(
                pmid=p.pmid,
                title=p.title,
                abstract=p.abstract or "No abstract available",
                authors=p.authors,
                journal=p.journal,
                publication_date=p.publication_date,
                doi=p.doi,
                keywords=p.keywords,
                mesh_terms=p.mesh_terms
            )
    except Exception as e:
        logger.error(f"PubMed fetch error for PMID {pmid}: {e}")

    raise HTTPException(
        status_code=404,
        detail=f"Paper with PMID {pmid} not found"
    )


@router.get("/papers/{pmid}/similar", response_model=List[SimilarPaperResponse])
async def get_similar_papers_endpoint(
    pmid: str,
    limit: int = Query(default=5, ge=1, le=20)
):
    """
    Get similar papers

    - Uses cosine similarity on embeddings
    - Returns top-K most similar papers
    """
    similar = sample_papers.get_similar_papers(pmid, limit)
    return [SimilarPaperResponse(**s) for s in similar]


@router.get("/papers/{pmid}/ask")
async def ask_about_paper(
    pmid: str,
    question: str = Query(..., min_length=5)
):
    """
    Ask a question about a specific paper

    - Uses RAG with the paper as context
    - Returns AI-generated answer with citations
    """
    # TODO: Implement with RAGService

    return {
        "answer": "This feature is coming soon.",
        "sources": [pmid]
    }


class SummarizeRequest(BaseModel):
    """Summarize request"""
    text: str
    language: str = "ko"  # Target language for summary


class SummarizeResponse(BaseModel):
    """Summarize response"""
    summary: str
    original_length: int
    summary_length: int


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_text(request: SummarizeRequest):
    """
    Summarize text in Korean or other languages

    - Uses OpenAI GPT to generate concise summaries
    - Supports Korean (ko), English (en) output
    - Ideal for summarizing paper abstracts
    """
    import aiohttp

    if not settings.OPENAI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key not configured"
        )

    # Build prompt based on target language
    if request.language == "ko":
        system_prompt = """당신은 의생명과학 논문 요약 전문가입니다.
주어진 영어 초록을 한국어로 간결하게 요약해주세요.

요약 규칙:
1. 3-4문장으로 핵심 내용만 요약
2. 전문 용어는 적절히 한국어로 번역
3. 연구 목적, 방법, 주요 발견을 포함
4. 자연스러운 한국어 문장으로 작성"""
        user_prompt = f"다음 논문 초록을 한국어로 요약해주세요:\n\n{request.text}"
    else:
        system_prompt = """You are an expert at summarizing biomedical research papers.
Summarize the given abstract concisely.

Rules:
1. Summarize in 3-4 sentences
2. Include research objective, methods, and key findings
3. Use clear, accessible language"""
        user_prompt = f"Summarize this abstract:\n\n{request.text}"

    try:
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 500
            }

            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"OpenAI API error: {response.status} - {error_text}")
                    raise HTTPException(
                        status_code=502,
                        detail="Failed to generate summary"
                    )

                data = await response.json()
                summary = data["choices"][0]["message"]["content"]

                return SummarizeResponse(
                    summary=summary,
                    original_length=len(request.text),
                    summary_length=len(summary)
                )

    except aiohttp.ClientError as e:
        logger.error(f"HTTP error during summarization: {e}")
        raise HTTPException(
            status_code=502,
            detail="Failed to connect to AI service"
        )


class TranslateRequest(BaseModel):
    """Translation request"""
    text: str
    source_lang: str = "ko"  # Source language
    target_lang: str = "en"  # Target language


class TranslateResponse(BaseModel):
    """Translation response"""
    original: str
    translated: str
    source_lang: str
    target_lang: str


@router.post("/translate", response_model=TranslateResponse)
async def translate_text(request: TranslateRequest):
    """
    Translate biomedical search query from Korean to English

    - Uses OpenAI GPT for accurate translation
    - Optimized for biomedical/scientific terminology
    - Preserves search query intent
    """
    import aiohttp

    if not settings.OPENAI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key not configured"
        )

    # If text is already in English or target language, return as is
    import re
    if not re.search(r'[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]', request.text):
        return TranslateResponse(
            original=request.text,
            translated=request.text,
            source_lang=request.source_lang,
            target_lang=request.target_lang
        )

    system_prompt = """You are a biomedical search query translator.
Translate the Korean search query to English for PubMed search.

IMPORTANT RULES:
1. Translate ONLY the Korean text to English
2. Keep the translation concise and suitable for database search
3. Use standard biomedical/scientific terminology
4. Do NOT add any explanations or extra text
5. If there are English words mixed in, keep them as is
6. Return ONLY the translated search terms, nothing else

Examples:
- "암 면역치료" → "cancer immunotherapy"
- "유전자 편집 부작용" → "gene editing side effects"
- "CRISPR 치료 최신 연구" → "CRISPR therapy latest research"
- "줄기세포 치료법" → "stem cell therapy"
- "폐암 표적치료제" → "lung cancer targeted therapy"
- "알츠하이머 신약 개발" → "Alzheimer drug development"
- "코로나 백신 효과" → "COVID vaccine efficacy"
- "유방암 조기 진단" → "breast cancer early diagnosis"
- "당뇨병 합병증 예방" → "diabetes complications prevention"
- "뇌졸중 재활 치료" → "stroke rehabilitation therapy"
"""

    user_prompt = f"Translate this Korean search query to English:\n{request.text}"

    try:
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.1,  # Low temperature for consistent translations
                "max_tokens": 100
            }

            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"OpenAI API error: {response.status} - {error_text}")
                    raise HTTPException(
                        status_code=502,
                        detail="Failed to translate"
                    )

                data = await response.json()
                translated = data["choices"][0]["message"]["content"].strip()

                # Clean up the translation (remove quotes if present)
                translated = translated.strip('"\'')

                logger.info(f"Translation: '{request.text}' -> '{translated}'")

                return TranslateResponse(
                    original=request.text,
                    translated=translated,
                    source_lang=request.source_lang,
                    target_lang=request.target_lang
                )

    except aiohttp.ClientError as e:
        logger.error(f"HTTP error during translation: {e}")
        raise HTTPException(
            status_code=502,
            detail="Failed to connect to AI service"
        )


# ============== PMC PDF Endpoints ==============

@router.get("/papers/{pmid}/pdf-info", response_model=PDFInfoResponse)
async def get_pdf_info(pmid: str):
    """
    Get PDF availability info for a paper

    - Checks if the paper is in PMC (PubMed Central)
    - Returns PDF URL if available as open access
    """
    pmc_service = get_pmc_service()

    try:
        info = await pmc_service.get_single_pdf_info(pmid)

        return PDFInfoResponse(
            pmid=info.pmid,
            pmcid=info.pmcid,
            has_pdf=info.has_pdf,
            pdf_url=info.pdf_url,
            is_open_access=info.is_open_access
        )
    except Exception as e:
        logger.error(f"Error getting PDF info for PMID {pmid}: {e}")
        return PDFInfoResponse(
            pmid=pmid,
            pmcid=None,
            has_pdf=False,
            pdf_url=None,
            is_open_access=False
        )


@router.get("/papers/{pmid}/pdf")
async def download_pdf(pmid: str):
    """
    Download PDF for a paper if available

    - Returns the PDF file directly
    - Only works for open access papers in PMC
    """
    pmc_service = get_pmc_service()

    try:
        pdf_bytes, result = await pmc_service.download_pdf(pmid)

        if pdf_bytes is None:
            raise HTTPException(
                status_code=404,
                detail=result  # Error message
            )

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={result}"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading PDF for PMID {pmid}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download PDF: {str(e)}"
        )


class BatchPDFInfoRequest(BaseModel):
    """Request for batch PDF info check"""
    pmids: List[str]


class BatchPDFInfoResponse(BaseModel):
    """Response for batch PDF info check"""
    papers: List[PDFInfoResponse]


@router.post("/papers/pdf-info-batch", response_model=BatchPDFInfoResponse)
async def get_pdf_info_batch(request: BatchPDFInfoRequest):
    """
    Get PDF availability info for multiple papers

    - Checks if papers are in PMC (PubMed Central)
    - Returns PDF URLs for open access papers
    - More efficient than individual requests
    """
    pmc_service = get_pmc_service()

    try:
        results = await pmc_service.get_pdf_info(request.pmids)

        papers = []
        for pmid in request.pmids:
            info = results.get(pmid)
            if info:
                papers.append(PDFInfoResponse(
                    pmid=info.pmid,
                    pmcid=info.pmcid,
                    has_pdf=info.has_pdf,
                    pdf_url=info.pdf_url,
                    is_open_access=info.is_open_access
                ))
            else:
                papers.append(PDFInfoResponse(
                    pmid=pmid,
                    pmcid=None,
                    has_pdf=False,
                    pdf_url=None,
                    is_open_access=False
                ))

        return BatchPDFInfoResponse(papers=papers)
    except Exception as e:
        logger.error(f"Error getting batch PDF info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get PDF info: {str(e)}"
        )
