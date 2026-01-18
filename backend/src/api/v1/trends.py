"""Trends API Endpoints - Research trend analysis"""

from typing import List, Optional
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from datetime import date, datetime
import random
import logging
import aiohttp

from src.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


# ============== Sample Trend Data ==============

SAMPLE_HOT_TOPICS = [
    {"keyword": "CRISPR-Cas9", "count": 1847, "growth_rate": 0.23},
    {"keyword": "CAR-T therapy", "count": 1523, "growth_rate": 0.31},
    {"keyword": "mRNA vaccines", "count": 1456, "growth_rate": 0.45},
    {"keyword": "single-cell RNA-seq", "count": 1234, "growth_rate": 0.28},
    {"keyword": "immunotherapy", "count": 1189, "growth_rate": 0.15},
    {"keyword": "AlphaFold", "count": 987, "growth_rate": 0.52},
    {"keyword": "base editing", "count": 876, "growth_rate": 0.38},
    {"keyword": "tumor microenvironment", "count": 823, "growth_rate": 0.19},
    {"keyword": "gut microbiome", "count": 756, "growth_rate": 0.21},
    {"keyword": "spatial transcriptomics", "count": 698, "growth_rate": 0.67},
]

SAMPLE_EMERGING_TOPICS = [
    {"keyword": "spatial multi-omics", "growth_rate": 3.2, "recent_papers": 234},
    {"keyword": "prime editing", "growth_rate": 2.8, "recent_papers": 189},
    {"keyword": "AI drug discovery", "growth_rate": 2.6, "recent_papers": 312},
    {"keyword": "lipid nanoparticles", "growth_rate": 2.4, "recent_papers": 287},
    {"keyword": "bispecific antibodies", "growth_rate": 2.3, "recent_papers": 198},
    {"keyword": "TCR-T therapy", "growth_rate": 2.2, "recent_papers": 156},
    {"keyword": "organoids", "growth_rate": 2.1, "recent_papers": 423},
    {"keyword": "epigenetic editing", "growth_rate": 2.0, "recent_papers": 134},
]

SAMPLE_WORDCLOUD = [
    {"word": "cancer", "count": 4523},
    {"word": "immunotherapy", "count": 2341},
    {"word": "CRISPR", "count": 1987},
    {"word": "gene editing", "count": 1654},
    {"word": "CAR-T", "count": 1432},
    {"word": "mRNA", "count": 1298},
    {"word": "vaccine", "count": 1187},
    {"word": "single-cell", "count": 1098},
    {"word": "biomarker", "count": 987},
    {"word": "tumor", "count": 923},
    {"word": "T cell", "count": 876},
    {"word": "antibody", "count": 834},
    {"word": "nanoparticle", "count": 765},
    {"word": "microbiome", "count": 723},
    {"word": "sequencing", "count": 687},
    {"word": "protein", "count": 654},
    {"word": "drug discovery", "count": 612},
    {"word": "clinical trial", "count": 598},
    {"word": "AI", "count": 567},
    {"word": "deep learning", "count": 534},
]


# ============== Schemas ==============

class KeywordTrend(BaseModel):
    """Keyword trend data point"""
    date: str
    keyword: str
    count: int


class KeywordTrendResponse(BaseModel):
    """Keyword trend response"""
    keywords: List[str]
    data: List[KeywordTrend]


class HotTopic(BaseModel):
    """Hot topic"""
    keyword: str
    count: int
    growth_rate: float  # Compared to previous period


class HotTopicsResponse(BaseModel):
    """Hot topics response"""
    period: str
    topics: List[HotTopic]


class TopicHeatmapData(BaseModel):
    """Topic heatmap data"""
    topic: str
    year: int
    month: int
    count: int


class TopicHeatmapResponse(BaseModel):
    """Topic heatmap response"""
    data: List[TopicHeatmapData]


class EmergingTopic(BaseModel):
    """Emerging topic"""
    keyword: str
    growth_rate: float
    recent_papers: int


class EmergingTopicsResponse(BaseModel):
    """Emerging topics response"""
    topics: List[EmergingTopic]


# ============== Endpoints ==============

@router.get("/keywords", response_model=KeywordTrendResponse)
async def get_keyword_trends(
    keywords: List[str] = Query(default=["CRISPR", "CAR-T", "immunotherapy"]),
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    aggregation: str = Query(default="monthly", pattern="^(daily|weekly|monthly)$")
):
    """
    Get keyword trends over time

    - Shows publication count for each keyword
    - Supports different time aggregations
    """
    # Generate sample trend data
    data = []
    current_year = datetime.now().year

    for keyword in keywords:
        base_count = random.randint(50, 200)
        for month in range(1, 13):
            # Add some variation and growth trend
            count = base_count + random.randint(-20, 40) + (month * 5)
            data.append(KeywordTrend(
                date=f"{current_year}-{month:02d}",
                keyword=keyword,
                count=count
            ))

    return KeywordTrendResponse(
        keywords=keywords,
        data=data
    )


@router.get("/hot", response_model=HotTopicsResponse)
async def get_hot_topics(
    period: str = Query(default="month", pattern="^(week|month|quarter)$"),
    limit: int = Query(default=10, ge=1, le=50)
):
    """
    Get current hot topics

    - Based on publication frequency
    - Includes growth rate compared to previous period
    """
    topics = [HotTopic(**t) for t in SAMPLE_HOT_TOPICS[:limit]]

    return HotTopicsResponse(
        period=period,
        topics=topics
    )


@router.get("/emerging", response_model=EmergingTopicsResponse)
async def get_emerging_topics(
    window_months: int = Query(default=6, ge=1, le=12),
    growth_threshold: float = Query(default=2.0, ge=1.0),
    limit: int = Query(default=10, ge=1, le=50)
):
    """
    Get emerging research topics

    - Detects rapidly growing research areas
    - Based on growth rate over specified window
    """
    # Filter by growth threshold
    filtered = [t for t in SAMPLE_EMERGING_TOPICS if t["growth_rate"] >= growth_threshold]
    topics = [EmergingTopic(**t) for t in filtered[:limit]]

    return EmergingTopicsResponse(topics=topics)


@router.get("/heatmap", response_model=TopicHeatmapResponse)
async def get_topic_heatmap(
    topics: List[str] = Query(default=["cancer", "immunology", "genetics"]),
    years: int = Query(default=2, ge=1, le=5)
):
    """
    Get topic publication heatmap

    - Shows publication density by topic and month
    - Useful for visualizing research activity patterns
    """
    data = []
    current_year = datetime.now().year

    for topic in topics:
        base_count = random.randint(100, 300)
        for year_offset in range(years):
            year = current_year - year_offset
            for month in range(1, 13):
                # Generate varying counts with seasonal patterns
                seasonal_factor = 1.0 + 0.2 * abs(6 - month) / 6
                count = int(base_count * seasonal_factor + random.randint(-30, 50))
                data.append(TopicHeatmapData(
                    topic=topic,
                    year=year,
                    month=month,
                    count=max(10, count)
                ))

    return TopicHeatmapResponse(data=data)


@router.get("/wordcloud")
async def get_wordcloud_data(
    period: str = Query(default="month", pattern="^(week|month|quarter)$"),
    limit: int = Query(default=100, ge=10, le=200)
):
    """
    Get data for word cloud visualization

    - Returns keyword frequencies
    - Based on recent publications
    """
    # Adjust counts based on period
    multiplier = {"week": 0.25, "month": 1.0, "quarter": 3.0}.get(period, 1.0)

    words = [
        {"word": w["word"], "count": int(w["count"] * multiplier)}
        for w in SAMPLE_WORDCLOUD[:limit]
    ]

    return {
        "period": period,
        "words": words
    }


# ============== AI Trend Analysis ==============

class TrendAnalysisRequest(BaseModel):
    """Request for AI trend analysis"""
    query: str
    language: str = "ko"  # ko or en


class TrendAnalysisResponse(BaseModel):
    """AI trend analysis response"""
    query: str
    analysis: str
    key_trends: List[str]
    related_topics: List[str]
    research_direction: str
    summary: str


@router.post("/analyze", response_model=TrendAnalysisResponse)
async def analyze_research_trend(request: TrendAnalysisRequest):
    """
    AI-powered research trend analysis using OpenAI

    - Analyzes research trends for a given query
    - Provides key trends, related topics, and research direction
    - Returns analysis in the requested language
    """
    if not settings.OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    try:
        # Build comprehensive prompt for detailed trend analysis
        lang_instruction = "Korean (한국어)" if request.language == "ko" else "English"

        system_prompt = f"""You are a world-class biomedical research trend analyst with deep expertise in analyzing scientific literature and research trends. Provide a comprehensive, detailed analysis of research trends for the given query.

Your response MUST be in JSON format with the following structure:
{{
    "analysis": "VERY DETAILED analysis (minimum 800-1000 words) covering multiple aspects",
    "key_trends": ["Trend 1 with brief explanation", "Trend 2 with brief explanation", "Trend 3", "Trend 4", "Trend 5", "Trend 6", "Trend 7"],
    "related_topics": ["Related topic 1", "Related topic 2", "Related topic 3", "Related topic 4", "Related topic 5"],
    "research_direction": "Comprehensive future research direction (200-300 words)",
    "summary": "Executive summary (150-200 words)"
}}

The "analysis" field MUST include ALL of the following sections in detail:

1. **연구 개요 및 중요성 (Research Overview & Significance)**
   - 해당 연구 분야의 정의와 범위
   - 왜 이 연구가 중요한지
   - 사회적, 의학적 영향

2. **역사적 발전 과정 (Historical Development)**
   - 주요 마일스톤과 돌파구
   - 기술/방법론의 진화
   - 핵심 발견들

3. **현재 연구 동향 (Current Research Trends)**
   - 가장 활발한 연구 주제들
   - 주요 연구 접근법과 방법론
   - 최신 기술 및 도구

4. **주요 연구 기관 및 연구자 (Key Institutions & Researchers)**
   - 선도적인 연구 기관들
   - 영향력 있는 연구자들
   - 주요 협력 네트워크

5. **임상 적용 및 산업 동향 (Clinical Applications & Industry Trends)**
   - 현재 임상 시험 현황
   - 승인된 치료법/제품
   - 주요 기업들의 활동

6. **도전과제 및 한계점 (Challenges & Limitations)**
   - 기술적 한계
   - 윤리적 고려사항
   - 규제 관련 이슈

7. **미래 전망 (Future Outlook)**
   - 예상되는 발전 방향
   - 새로운 기회 영역
   - 해결해야 할 과제들

Guidelines:
- Be extremely detailed and comprehensive
- Include specific examples, statistics, and data when possible
- Mention recent publications (2023-2024) and breakthroughs
- Reference specific clinical trials, companies, or products
- Provide actionable insights for researchers
- Use proper scientific terminology

CRITICAL: Respond entirely in {lang_instruction}. Make the analysis as detailed and informative as possible."""

        user_prompt = f"""Please provide a comprehensive and detailed research trend analysis for: {request.query}

Include specific data, recent developments, key players, clinical applications, challenges, and future directions. Make the analysis thorough and informative, suitable for researchers and professionals in the field."""

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
                "temperature": 0.7,
                "max_tokens": 4000,
                "response_format": {"type": "json_object"}
            }

            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"OpenAI API error: {response.status} - {error_text}")
                    raise HTTPException(status_code=500, detail="AI analysis failed")

                data = await response.json()
                content = data["choices"][0]["message"]["content"]

                # Parse JSON response
                import json
                result = json.loads(content)

                return TrendAnalysisResponse(
                    query=request.query,
                    analysis=result.get("analysis", ""),
                    key_trends=result.get("key_trends", []),
                    related_topics=result.get("related_topics", []),
                    research_direction=result.get("research_direction", ""),
                    summary=result.get("summary", "")
                )

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse AI response: {e}")
        raise HTTPException(status_code=500, detail="Failed to parse AI response")
    except Exception as e:
        logger.error(f"Trend analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
