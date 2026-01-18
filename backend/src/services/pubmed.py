"""PubMed API Service - E-utilities Integration"""

import asyncio
import aiohttp
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# PubMed E-utilities base URLs
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


@dataclass
class PubMedPaper:
    """PubMed paper data structure"""
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    journal: str
    publication_date: Optional[str]
    doi: Optional[str]
    keywords: List[str]
    mesh_terms: List[str]


class PubMedService:
    """Service for interacting with PubMed E-utilities API"""

    def __init__(self, api_key: str = "", email: str = ""):
        """
        Initialize PubMed service

        Args:
            api_key: NCBI API key (optional, increases rate limit)
            email: Email for NCBI (recommended)
        """
        self.api_key = api_key
        self.email = email
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Close the aiohttp session"""
        if self._session and not self._session.closed:
            await self._session.close()

    def _build_params(self, base_params: Dict) -> Dict:
        """Add API key and email to params if available"""
        params = base_params.copy()
        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["email"] = self.email
        return params

    async def search(
        self,
        query: str,
        max_results: int = 10,
        sort: str = "relevance",
        min_date: Optional[str] = None,
        max_date: Optional[str] = None
    ) -> Tuple[int, List[str]]:
        """
        Search PubMed for papers matching the query

        Args:
            query: Search query (supports PubMed query syntax)
            max_results: Maximum number of results to return
            sort: Sort order ('relevance' or 'date')
            min_date: Minimum publication date (YYYY/MM/DD)
            max_date: Maximum publication date (YYYY/MM/DD)

        Returns:
            Tuple of (total count, list of PMIDs)
        """
        session = await self._get_session()

        params = self._build_params({
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": sort,
            "usehistory": "n"
        })

        # Apply date filters - datetype must be set for date filtering to work
        if min_date or max_date:
            params["datetype"] = "pdat"  # Publication date
            if min_date:
                params["mindate"] = min_date
            if max_date:
                params["maxdate"] = max_date

        try:
            async with session.get(ESEARCH_URL, params=params) as response:
                if response.status != 200:
                    logger.error(f"PubMed search failed: {response.status}")
                    return 0, []

                data = await response.json()
                result = data.get("esearchresult", {})

                total = int(result.get("count", 0))
                pmids = result.get("idlist", [])

                logger.info(f"PubMed search '{query}': {total} total, {len(pmids)} returned")
                return total, pmids

        except Exception as e:
            logger.error(f"PubMed search error: {e}")
            return 0, []

    async def fetch_papers(self, pmids: List[str]) -> List[PubMedPaper]:
        """
        Fetch paper details for given PMIDs

        Args:
            pmids: List of PubMed IDs

        Returns:
            List of PubMedPaper objects
        """
        if not pmids:
            return []

        session = await self._get_session()

        params = self._build_params({
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract"
        })

        try:
            async with session.get(EFETCH_URL, params=params) as response:
                if response.status != 200:
                    logger.error(f"PubMed fetch failed: {response.status}")
                    return []

                xml_content = await response.text()
                return self._parse_pubmed_xml(xml_content)

        except Exception as e:
            logger.error(f"PubMed fetch error: {e}")
            return []

    def _parse_pubmed_xml(self, xml_content: str) -> List[PubMedPaper]:
        """Parse PubMed XML response"""
        papers = []

        try:
            root = ET.fromstring(xml_content)

            for article in root.findall(".//PubmedArticle"):
                paper = self._parse_article(article)
                if paper:
                    papers.append(paper)

        except ET.ParseError as e:
            logger.error(f"XML parse error: {e}")

        return papers

    def _parse_article(self, article: ET.Element) -> Optional[PubMedPaper]:
        """Parse a single PubMed article element"""
        try:
            medline = article.find(".//MedlineCitation")
            if medline is None:
                return None

            # PMID
            pmid_elem = medline.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else ""

            # Article info
            article_elem = medline.find(".//Article")
            if article_elem is None:
                return None

            # Title
            title_elem = article_elem.find(".//ArticleTitle")
            title = self._get_text_content(title_elem) if title_elem is not None else ""

            # Abstract
            abstract_elem = article_elem.find(".//Abstract")
            abstract = ""
            if abstract_elem is not None:
                abstract_texts = []
                for text_elem in abstract_elem.findall(".//AbstractText"):
                    label = text_elem.get("Label", "")
                    text = self._get_text_content(text_elem)
                    if label:
                        abstract_texts.append(f"{label}: {text}")
                    else:
                        abstract_texts.append(text)
                abstract = " ".join(abstract_texts)

            # Authors
            authors = []
            author_list = article_elem.find(".//AuthorList")
            if author_list is not None:
                for author in author_list.findall(".//Author"):
                    lastname = author.find("LastName")
                    forename = author.find("ForeName")
                    if lastname is not None:
                        name = lastname.text or ""
                        if forename is not None and forename.text:
                            name = f"{name}, {forename.text}"
                        authors.append(name)

            # Journal
            journal_elem = article_elem.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None else ""

            # Publication date
            pub_date = None
            date_elem = article_elem.find(".//ArticleDate")
            if date_elem is None:
                date_elem = article_elem.find(".//Journal/JournalIssue/PubDate")

            if date_elem is not None:
                year = date_elem.find("Year")
                month = date_elem.find("Month")
                day = date_elem.find("Day")

                if year is not None:
                    pub_date = year.text
                    if month is not None:
                        pub_date += f"-{month.text.zfill(2) if month.text.isdigit() else month.text}"
                        if day is not None:
                            pub_date += f"-{day.text.zfill(2)}"

            # DOI
            doi = None
            for id_elem in article_elem.findall(".//ELocationID"):
                if id_elem.get("EIdType") == "doi":
                    doi = id_elem.text
                    break

            # Keywords
            keywords = []
            keyword_list = medline.find(".//KeywordList")
            if keyword_list is not None:
                for kw in keyword_list.findall(".//Keyword"):
                    if kw.text:
                        keywords.append(kw.text)

            # MeSH terms
            mesh_terms = []
            mesh_list = medline.find(".//MeshHeadingList")
            if mesh_list is not None:
                for mesh in mesh_list.findall(".//MeshHeading/DescriptorName"):
                    if mesh.text:
                        mesh_terms.append(mesh.text)

            return PubMedPaper(
                pmid=pmid,
                title=title,
                abstract=abstract,
                authors=authors,
                journal=journal,
                publication_date=pub_date,
                doi=doi,
                keywords=keywords,
                mesh_terms=mesh_terms
            )

        except Exception as e:
            logger.error(f"Error parsing article: {e}")
            return None

    def _get_text_content(self, elem: ET.Element) -> str:
        """Get all text content from an element, including nested elements"""
        return "".join(elem.itertext())

    async def search_and_fetch(
        self,
        query: str,
        max_results: int = 10,
        sort: str = "relevance",
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        journals: Optional[List[str]] = None,
        authors: Optional[List[str]] = None
    ) -> Tuple[int, List[PubMedPaper]]:
        """
        Search PubMed and fetch paper details in one call

        Args:
            query: Search query
            max_results: Maximum number of results
            sort: Sort order
            year_from: Minimum publication year
            year_to: Maximum publication year
            journals: Filter by journal names
            authors: Filter by author names

        Returns:
            Tuple of (total count, list of papers)
        """
        # Build enhanced query with filters
        enhanced_query = query

        # Add date filter to query (more reliable than mindate/maxdate params with relevance sort)
        if year_from and year_to:
            enhanced_query = f"({enhanced_query}) AND ({year_from}:{year_to}[pdat])"
        elif year_from:
            current_year = datetime.now().year
            enhanced_query = f"({enhanced_query}) AND ({year_from}:{current_year}[pdat])"
        elif year_to:
            enhanced_query = f"({enhanced_query}) AND (1900:{year_to}[pdat])"

        # Add journal filter to query
        if journals:
            journal_terms = " OR ".join([f'"{j}"[Journal]' for j in journals])
            enhanced_query = f"({enhanced_query}) AND ({journal_terms})"

        # Add author filter to query
        if authors:
            author_terms = " OR ".join([f'{a}[Author]' for a in authors])
            enhanced_query = f"({enhanced_query}) AND ({author_terms})"

        total, pmids = await self.search(enhanced_query, max_results, sort)

        if not pmids:
            return total, []

        papers = await self.fetch_papers(pmids)
        return total, papers


# Global service instance
_pubmed_service: Optional[PubMedService] = None


def get_pubmed_service(api_key: str = "", email: str = "") -> PubMedService:
    """Get or create PubMed service instance"""
    global _pubmed_service
    if _pubmed_service is None:
        _pubmed_service = PubMedService(api_key=api_key, email=email)
    return _pubmed_service
