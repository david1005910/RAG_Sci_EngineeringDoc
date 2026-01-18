"""Chat API Endpoints - RAG-based Q&A with VectorDB Hybrid Search"""

import logging
import re
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.core.security import get_current_user_id, get_current_user_id_optional
from src.data import sample_papers
from src.services.pubmed import get_pubmed_service
from src.services.ai_chat import get_ai_service, ChatSource
from src.core.config import settings
from src.api.v1.vectordb import get_vector_store

logger = logging.getLogger(__name__)


# ============== Korean Translation ==============

def contains_korean(text: str) -> bool:
    """Check if text contains Korean characters"""
    return bool(re.search(r'[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]', text))


# Korean to English biomedical term mapping
KOREAN_TO_ENGLISH_MAP = {
    '암': 'cancer',
    '유방암': 'breast cancer',
    '폐암': 'lung cancer',
    '대장암': 'colorectal cancer',
    '위암': 'gastric cancer',
    '간암': 'liver cancer',
    '면역': 'immune',
    '면역치료': 'immunotherapy',
    '면역요법': 'immunotherapy',
    '항암': 'anticancer',
    '항암제': 'anticancer drug',
    '유전자': 'gene',
    '유전자 치료': 'gene therapy',
    '유전자 편집': 'gene editing',
    '세포': 'cell',
    '줄기세포': 'stem cell',
    '단백질': 'protein',
    '항체': 'antibody',
    '백신': 'vaccine',
    '바이러스': 'virus',
    '박테리아': 'bacteria',
    '감염': 'infection',
    '염증': 'inflammation',
    '당뇨': 'diabetes',
    '당뇨병': 'diabetes',
    '고혈압': 'hypertension',
    '심장': 'heart',
    '심장병': 'heart disease',
    '뇌': 'brain',
    '신경': 'nerve',
    '신경과학': 'neuroscience',
    '알츠하이머': 'alzheimer',
    '파킨슨': 'parkinson',
    '치료': 'treatment',
    '진단': 'diagnosis',
    '예방': 'prevention',
    '부작용': 'side effects',
    '효과': 'efficacy',
    '임상시험': 'clinical trial',
    '최신': 'latest',
    '연구': 'research',
    '동향': 'trends',
    '주요': 'major',
    '발견': 'findings',
    '논문': 'paper',
    '알려줘': '',
    '설명해줘': '',
    '무엇인가요': '',
    '어떤가요': '',
}


def translate_korean_to_english(korean_text: str) -> str:
    """Translate Korean biomedical terms to English"""
    translated = korean_text

    # Sort by length (longer terms first) to avoid partial replacements
    sorted_terms = sorted(
        KOREAN_TO_ENGLISH_MAP.items(),
        key=lambda x: len(x[0]),
        reverse=True
    )

    for korean, english in sorted_terms:
        translated = translated.replace(korean, english)

    # Remove any remaining Korean characters
    translated = re.sub(r'[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]', ' ', translated)
    translated = re.sub(r'\s+', ' ', translated).strip()

    return translated or korean_text

router = APIRouter()


# ============== Schemas ==============

class ChatQueryRequest(BaseModel):
    """Chat query request"""
    question: str
    session_id: Optional[str] = None
    context_pmids: List[str] = []
    max_sources: int = 5
    use_ai: bool = True  # Enable AI-powered responses
    source: str = "pubmed"  # "pubmed", "vectordb", or "mock"
    use_vectordb: bool = True  # Use VectorDB hybrid search first
    search_mode: str = "hybrid"  # "hybrid", "dense", or "sparse" for VectorDB
    dense_weight: float = 0.7  # Weight for dense search in hybrid mode


class SourceInfo(BaseModel):
    """Source information"""
    pmid: str
    title: str
    relevance: float
    excerpt: str
    source_type: str = "pubmed"  # "vectordb" or "pubmed"
    dense_score: Optional[float] = None  # For VectorDB hybrid search
    sparse_score: Optional[float] = None  # For VectorDB hybrid search


class ChatQueryResponse(BaseModel):
    """Chat query response"""
    answer: str
    sources: List[SourceInfo]
    confidence: float
    processing_time_ms: int
    session_id: str
    vectordb_used: bool = False
    search_mode: Optional[str] = None  # "hybrid", "dense", "sparse", or None


class ChatSession(BaseModel):
    """Chat session"""
    id: str
    title: str
    created_at: str
    message_count: int


class ChatMessage(BaseModel):
    """Chat message"""
    id: str
    role: str  # "user" or "assistant"
    content: str
    sources: List[SourceInfo] = []
    created_at: str


class SessionListResponse(BaseModel):
    """Session list response"""
    sessions: List[ChatSession]


class MessageListResponse(BaseModel):
    """Message list response"""
    messages: List[ChatMessage]


# ============== Endpoints ==============

@router.post("/query", response_model=ChatQueryResponse)
async def chat_query(
    request: ChatQueryRequest,
    user_id: Optional[str] = Depends(get_current_user_id_optional)
):
    """
    AI-powered Q&A about biomedical research with VectorDB Hybrid Search

    - Uses RAG (Retrieval-Augmented Generation)
    - **VectorDB Hybrid Search**: Searches indexed papers using Qdrant (dense) + SPLADE (sparse)
    - **PubMed Fallback**: Falls back to PubMed API if VectorDB has no results
    - Uses OpenAI GPT to generate contextual answers
    - Includes source citations with PMID links

    Search modes for VectorDB:
    - "hybrid": Combined dense + sparse search (default)
    - "dense": Semantic similarity using embeddings
    - "sparse": SPLADE-based keyword expansion

    Response time target: < 10 seconds
    """
    import uuid
    import time

    start_time = time.time()
    sources = []
    papers_for_context = []
    original_question = request.question
    search_query = request.question
    vectordb_used = False
    actual_search_mode = None

    # Translate Korean questions to English for better search
    is_korean_query = contains_korean(request.question)
    if is_korean_query:
        search_query = translate_korean_to_english(request.question)
        logger.info(f"Korean query translated: '{request.question}' -> '{search_query}'")

    # Step 1: Try VectorDB Hybrid Search first
    if request.use_vectordb:
        try:
            vector_store = get_vector_store()
            stats = vector_store.get_stats()

            if stats["vectors_count"] > 0:
                logger.info(f"Searching VectorDB ({stats['vectors_count']} vectors, mode: {request.search_mode})")

                vectordb_results = await vector_store.search(
                    query=search_query,
                    top_k=request.max_sources,
                    mode=request.search_mode,
                    dense_weight=request.dense_weight
                )

                if vectordb_results:
                    vectordb_used = True
                    actual_search_mode = request.search_mode

                    for result in vectordb_results:
                        metadata = result.get("metadata", {})
                        pmid = metadata.get("pmid", "")
                        title = metadata.get("title", "Untitled")
                        text = result.get("text", "")

                        sources.append(SourceInfo(
                            pmid=pmid,
                            title=title,
                            relevance=result.get("score", 0.0),
                            excerpt=(text[:300] + "...") if len(text) > 300 else text,
                            source_type="vectordb",
                            dense_score=result.get("dense_score"),
                            sparse_score=result.get("sparse_score")
                        ))
                        papers_for_context.append(ChatSource(
                            pmid=pmid,
                            title=title,
                            abstract=text,
                            relevance=result.get("score", 0.0)
                        ))

                    logger.info(f"VectorDB hybrid search: {len(vectordb_results)} results (mode: {request.search_mode})")

        except Exception as e:
            logger.warning(f"VectorDB search failed: {e}")
            vectordb_used = False

    # Step 2: Fall back to PubMed if VectorDB has no results
    if not sources and request.source == "pubmed":
        try:
            pubmed = get_pubmed_service(api_key=settings.PUBMED_API_KEY)
            total, papers = await pubmed.search_and_fetch(
                query=search_query,
                max_results=request.max_sources
            )

            for i, paper in enumerate(papers):
                relevance = 1.0 - (i * 0.1) if i < 10 else 0.1
                sources.append(SourceInfo(
                    pmid=paper.pmid,
                    title=paper.title,
                    relevance=relevance,
                    excerpt=(paper.abstract[:300] + "...") if paper.abstract else "No abstract available",
                    source_type="pubmed"
                ))
                papers_for_context.append(ChatSource(
                    pmid=paper.pmid,
                    title=paper.title,
                    abstract=paper.abstract or "No abstract available",
                    relevance=relevance
                ))

            logger.info(f"PubMed fallback search: {total} total, {len(papers)} retrieved")

        except Exception as e:
            logger.error(f"PubMed search error: {e}")
            request.source = "mock"

    # Step 3: Fall back to mock data
    if not sources and request.source == "mock":
        total, results = sample_papers.search_papers(
            query=search_query,
            limit=request.max_sources
        )

        for paper in results:
            sources.append(SourceInfo(
                pmid=paper["pmid"],
                title=paper["title"],
                relevance=paper["relevance_score"],
                excerpt=paper["abstract"][:300] + "...",
                source_type="mock"
            ))
            papers_for_context.append(ChatSource(
                pmid=paper["pmid"],
                title=paper["title"],
                abstract=paper["abstract"],
                relevance=paper["relevance_score"]
            ))

    # Step 4: Generate AI response
    if request.use_ai and papers_for_context:
        try:
            ai_service = get_ai_service()
            ai_response = await ai_service.chat_with_context(
                question=request.question,
                sources=papers_for_context
            )

            answer = ai_response.answer
            confidence = ai_response.confidence

            # Boost confidence if VectorDB hybrid search was used
            if vectordb_used and actual_search_mode == "hybrid":
                confidence = min(confidence + 0.1, 1.0)

            logger.info(f"AI response generated, confidence: {confidence}")

        except Exception as e:
            logger.error(f"AI service error: {e}")
            answer = _generate_fallback_answer(request.question, papers_for_context)
            confidence = 0.3
    else:
        answer = _generate_fallback_answer(request.question, papers_for_context)
        confidence = 0.3 if papers_for_context else 0.1

    processing_time = int((time.time() - start_time) * 1000)

    return ChatQueryResponse(
        answer=answer,
        sources=sources,
        confidence=confidence,
        processing_time_ms=processing_time,
        session_id=request.session_id or str(uuid.uuid4()),
        vectordb_used=vectordb_used,
        search_mode=actual_search_mode
    )


def _generate_fallback_answer(question: str, sources: List[ChatSource]) -> str:
    """Generate fallback response when AI is unavailable"""
    if not sources:
        return (
            "I couldn't find relevant papers for your question. "
            "Please try rephrasing or use different keywords related to biomedical research."
        )

    answer_parts = [
        f"Based on my search, I found {len(sources)} relevant papers:\n\n"
    ]

    for i, source in enumerate(sources[:3], 1):
        answer_parts.append(
            f"**{i}. {source.title}** (PMID: {source.pmid})\n"
            f"{source.abstract[:250]}...\n\n"
        )

    answer_parts.append(
        "*For AI-powered analysis, please configure an OpenAI API key.*"
    )

    return "".join(answer_parts)


@router.post("/sessions", response_model=ChatSession)
async def create_session(
    title: Optional[str] = None,
    user_id: str = Depends(get_current_user_id)
):
    """
    Create a new chat session

    - Requires authentication
    - Stores conversation history
    """
    import uuid
    from datetime import datetime

    session_id = str(uuid.uuid4())

    return ChatSession(
        id=session_id,
        title=title or "New Conversation",
        created_at=datetime.utcnow().isoformat(),
        message_count=0
    )


@router.get("/sessions", response_model=SessionListResponse)
async def get_sessions(
    user_id: str = Depends(get_current_user_id)
):
    """
    Get all chat sessions for current user

    - Requires authentication
    - Returns list of sessions with metadata
    """
    # TODO: Fetch from database

    return SessionListResponse(sessions=[])


@router.get("/sessions/{session_id}", response_model=ChatSession)
async def get_session(
    session_id: str,
    user_id: str = Depends(get_current_user_id)
):
    """
    Get a specific chat session

    - Requires authentication
    - Returns session metadata
    """
    raise HTTPException(
        status_code=404,
        detail=f"Session {session_id} not found"
    )


@router.get("/sessions/{session_id}/messages", response_model=MessageListResponse)
async def get_session_messages(
    session_id: str,
    user_id: str = Depends(get_current_user_id)
):
    """
    Get all messages in a chat session

    - Requires authentication
    - Returns conversation history
    """
    # TODO: Fetch from database

    return MessageListResponse(messages=[])


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    user_id: str = Depends(get_current_user_id)
):
    """
    Delete a chat session

    - Requires authentication
    - Removes all messages in the session
    """
    # TODO: Delete from database

    return {"message": f"Session {session_id} deleted"}
