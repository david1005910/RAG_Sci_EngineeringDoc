"""Paper crawling tasks"""

import asyncio
import logging
from typing import List

from .celery import celery_app

logger = logging.getLogger(__name__)

# Default keywords to crawl
DEFAULT_KEYWORDS = [
    "CAR-T cell therapy",
    "CRISPR gene editing",
    "cancer immunotherapy",
    "mRNA vaccine",
    "gene therapy",
    "checkpoint inhibitor",
    "PD-1 PD-L1",
]


@celery_app.task(name="src.tasks.crawler.daily_paper_crawl")
def daily_paper_crawl(keywords: List[str] = None) -> dict:
    """
    Daily task to crawl new papers from PubMed.

    Args:
        keywords: List of search keywords (uses defaults if None)

    Returns:
        Summary of crawl results
    """
    keywords = keywords or DEFAULT_KEYWORDS

    # Run async crawl
    result = asyncio.run(_async_crawl(keywords))

    return result


async def _async_crawl(keywords: List[str]) -> dict:
    """Async implementation of paper crawling."""
    from src.services.collector.pubmed_collector import PubMedCollector

    collector = PubMedCollector()

    total_papers = 0
    results = {}

    for keyword in keywords:
        try:
            logger.info(f"Crawling papers for: {keyword}")

            # Search for recent papers (last 7 days)
            papers = await collector.search_and_fetch(
                query=f"{keyword}[Title/Abstract]",
                max_results=50
            )

            results[keyword] = len(papers)
            total_papers += len(papers)

            # TODO: Save papers to database
            # TODO: Trigger embedding generation

            logger.info(f"Found {len(papers)} papers for '{keyword}'")

        except Exception as e:
            logger.error(f"Error crawling '{keyword}': {e}")
            results[keyword] = 0

    return {
        "total_papers": total_papers,
        "by_keyword": results,
        "status": "completed"
    }


@celery_app.task(name="src.tasks.crawler.crawl_keyword")
def crawl_keyword(keyword: str, max_results: int = 100) -> dict:
    """
    Crawl papers for a specific keyword.

    Args:
        keyword: Search keyword
        max_results: Maximum papers to fetch

    Returns:
        Crawl results
    """
    result = asyncio.run(_async_crawl_keyword(keyword, max_results))
    return result


async def _async_crawl_keyword(keyword: str, max_results: int) -> dict:
    """Async implementation for single keyword crawl."""
    from src.services.collector.pubmed_collector import PubMedCollector

    collector = PubMedCollector()

    try:
        papers = await collector.search_and_fetch(
            query=f"{keyword}[Title/Abstract]",
            max_results=max_results
        )

        return {
            "keyword": keyword,
            "papers_found": len(papers),
            "status": "completed"
        }

    except Exception as e:
        logger.error(f"Error crawling '{keyword}': {e}")
        return {
            "keyword": keyword,
            "papers_found": 0,
            "status": "error",
            "error": str(e)
        }
