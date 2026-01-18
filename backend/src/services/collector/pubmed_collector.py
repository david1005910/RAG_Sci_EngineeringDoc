"""PubMed API Client for paper collection"""

import asyncio
import json
import logging
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple
from datetime import datetime

import httpx
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.config import settings

logger = logging.getLogger(__name__)


class PaperMetadata(BaseModel):
    """Paper metadata from PubMed"""
    pmid: str
    title: str
    abstract: str
    authors: List[str] = []
    journal: str = ""
    publication_date: Optional[datetime] = None
    doi: Optional[str] = None
    keywords: List[str] = []
    mesh_terms: List[str] = []


class PubMedAPIError(Exception):
    """Custom exception for PubMed API errors"""
    pass


class PubMedCollector:
    """
    PubMed E-utilities API client for paper collection.

    Supports:
    - Paper search with keywords
    - Batch metadata fetching
    - Rate limiting (10 req/sec with API key)
    """

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.PUBMED_API_KEY
        self._semaphore = asyncio.Semaphore(settings.PUBMED_RATE_LIMIT)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _api_call(self, endpoint: str, params: dict) -> str:
        """
        Make API call with rate limiting and retry logic.

        Args:
            endpoint: API endpoint (e.g., 'esearch.fcgi')
            params: Query parameters

        Returns:
            Response text
        """
        async with self._semaphore:
            if self.api_key:
                params["api_key"] = self.api_key

            async with httpx.AsyncClient() as client:
                try:
                    response = await client.get(
                        f"{self.BASE_URL}{endpoint}",
                        params=params,
                        timeout=30.0
                    )
                    response.raise_for_status()
                    return response.text
                except httpx.HTTPError as e:
                    logger.error(f"PubMed API error: {e}")
                    raise PubMedAPIError(f"API call failed: {e}")

    async def search_papers(
        self,
        query: str,
        max_results: int = 100,
        date_range: Optional[Tuple[str, str]] = None
    ) -> List[str]:
        """
        Search for papers and return PMIDs.

        Args:
            query: Search query (e.g., "cancer immunotherapy[Title/Abstract]")
            max_results: Maximum number of results
            date_range: Optional (start_date, end_date) in YYYY/MM/DD format

        Returns:
            List of PMIDs
        """
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance"
        }

        if date_range:
            params["mindate"] = date_range[0]
            params["maxdate"] = date_range[1]
            params["datetype"] = "pdat"  # Publication date

        response = await self._api_call("esearch.fcgi", params)
        result = json.loads(response)

        pmids = result.get("esearchresult", {}).get("idlist", [])
        logger.info(f"Found {len(pmids)} papers for query: {query}")

        return pmids

    async def fetch_paper(self, pmid: str) -> Optional[PaperMetadata]:
        """
        Fetch single paper metadata.

        Args:
            pmid: PubMed ID

        Returns:
            PaperMetadata or None if not found
        """
        papers = await self.batch_fetch([pmid])
        return papers[0] if papers else None

    async def batch_fetch(self, pmids: List[str]) -> List[PaperMetadata]:
        """
        Fetch metadata for multiple papers.

        Args:
            pmids: List of PubMed IDs

        Returns:
            List of PaperMetadata objects
        """
        if not pmids:
            return []

        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
            "retmode": "xml"
        }

        xml_response = await self._api_call("efetch.fcgi", params)
        return self._parse_xml(xml_response)

    def _parse_xml(self, xml_text: str) -> List[PaperMetadata]:
        """Parse PubMed XML response into PaperMetadata objects."""
        papers = []

        try:
            root = ET.fromstring(xml_text)

            for article in root.findall(".//PubmedArticle"):
                try:
                    paper = self._parse_article(article)
                    if paper:
                        papers.append(paper)
                except Exception as e:
                    logger.warning(f"Error parsing article: {e}")

        except ET.ParseError as e:
            logger.error(f"XML parse error: {e}")

        return papers

    def _parse_article(self, article: ET.Element) -> Optional[PaperMetadata]:
        """Parse a single PubmedArticle element."""
        # PMID
        pmid_elem = article.find(".//PMID")
        if pmid_elem is None:
            return None
        pmid = pmid_elem.text

        # Title
        title_elem = article.find(".//ArticleTitle")
        title = title_elem.text if title_elem is not None else ""

        # Abstract
        abstract_parts = []
        for ab in article.findall(".//AbstractText"):
            if ab.text:
                label = ab.get("Label", "")
                if label:
                    abstract_parts.append(f"{label}: {ab.text}")
                else:
                    abstract_parts.append(ab.text)
        abstract = " ".join(abstract_parts)

        # Authors
        authors = []
        for author in article.findall(".//Author"):
            last_name = author.find("LastName")
            fore_name = author.find("ForeName")
            if last_name is not None:
                name = last_name.text
                if fore_name is not None:
                    name = f"{fore_name.text} {name}"
                authors.append(name)

        # Journal
        journal_elem = article.find(".//Journal/Title")
        journal = journal_elem.text if journal_elem is not None else ""

        # Publication Date
        pub_date = None
        date_elem = article.find(".//PubDate")
        if date_elem is not None:
            year = date_elem.find("Year")
            month = date_elem.find("Month")
            day = date_elem.find("Day")
            if year is not None:
                try:
                    pub_date = datetime(
                        int(year.text),
                        int(month.text) if month is not None and month.text.isdigit() else 1,
                        int(day.text) if day is not None else 1
                    )
                except (ValueError, TypeError):
                    pass

        # DOI
        doi = None
        for article_id in article.findall(".//ArticleId"):
            if article_id.get("IdType") == "doi":
                doi = article_id.text
                break

        # Keywords
        keywords = []
        for kw in article.findall(".//Keyword"):
            if kw.text:
                keywords.append(kw.text)

        # MeSH Terms
        mesh_terms = []
        for mesh in article.findall(".//MeshHeading/DescriptorName"):
            if mesh.text:
                mesh_terms.append(mesh.text)

        return PaperMetadata(
            pmid=pmid,
            title=title or "",
            abstract=abstract,
            authors=authors,
            journal=journal,
            publication_date=pub_date,
            doi=doi,
            keywords=keywords,
            mesh_terms=mesh_terms
        )

    async def search_and_fetch(
        self,
        query: str,
        max_results: int = 100
    ) -> List[PaperMetadata]:
        """
        Convenience method: search and fetch in one call.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of PaperMetadata objects
        """
        pmids = await self.search_papers(query, max_results)
        if not pmids:
            return []

        return await self.batch_fetch(pmids)
