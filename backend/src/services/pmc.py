"""PMC (PubMed Central) Service - PDF availability and download"""

import aiohttp
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# PMC API endpoints
PMC_ID_CONVERTER_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
PMC_OA_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"


@dataclass
class PMCPaperInfo:
    """PMC paper info with PDF availability"""
    pmid: str
    pmcid: Optional[str]
    has_pdf: bool
    pdf_url: Optional[str]
    is_open_access: bool


class PMCService:
    """Service for interacting with PMC (PubMed Central) API"""

    def __init__(self):
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

    async def get_pmcid(self, pmid: str) -> Optional[str]:
        """Convert PMID to PMCID"""
        session = await self._get_session()

        params = {
            "ids": pmid,
            "format": "json",
            "idtype": "pmid"
        }

        try:
            async with session.get(PMC_ID_CONVERTER_URL, params=params) as response:
                if response.status != 200:
                    return None

                data = await response.json()
                records = data.get("records", [])

                if records and "pmcid" in records[0]:
                    return records[0]["pmcid"]

                return None
        except Exception as e:
            logger.error(f"Error converting PMID {pmid} to PMCID: {e}")
            return None

    async def get_single_pdf_info(self, pmid: str) -> PMCPaperInfo:
        """Get PDF info for a single paper"""
        results = await self.get_pdf_info([pmid])
        return results.get(pmid, PMCPaperInfo(
            pmid=pmid,
            pmcid=None,
            has_pdf=False,
            pdf_url=None,
            is_open_access=False
        ))

    async def get_pdf_info(self, pmids: List[str]) -> Dict[str, PMCPaperInfo]:
        """
        Get PDF availability info for multiple papers

        Args:
            pmids: List of PubMed IDs

        Returns:
            Dict mapping PMID to PMCPaperInfo
        """
        results = {}

        for pmid in pmids:
            pmcid = await self.get_pmcid(pmid)

            if pmcid:
                # Check if open access PDF is available
                pdf_url, is_oa = await self._check_oa_availability(pmcid)

                results[pmid] = PMCPaperInfo(
                    pmid=pmid,
                    pmcid=pmcid,
                    has_pdf=pdf_url is not None,
                    pdf_url=pdf_url,
                    is_open_access=is_oa
                )
            else:
                results[pmid] = PMCPaperInfo(
                    pmid=pmid,
                    pmcid=None,
                    has_pdf=False,
                    pdf_url=None,
                    is_open_access=False
                )

        return results

    async def _check_oa_availability(self, pmcid: str) -> Tuple[Optional[str], bool]:
        """Check if open access PDF is available for a PMCID"""
        session = await self._get_session()

        params = {"id": pmcid}

        try:
            async with session.get(PMC_OA_URL, params=params) as response:
                if response.status != 200:
                    return None, False

                text = await response.text()
                root = ET.fromstring(text)

                # Check for PDF link
                for link in root.findall(".//link"):
                    if link.get("format") == "pdf":
                        pdf_url = link.get("href")
                        return pdf_url, True

                return None, False
        except Exception as e:
            logger.error(f"Error checking OA availability for {pmcid}: {e}")
            return None, False

    async def download_pdf(self, pmid: str) -> Tuple[Optional[bytes], str]:
        """
        Download PDF for a paper if available

        Args:
            pmid: PubMed ID

        Returns:
            Tuple of (PDF bytes or None, filename or error message)
        """
        info = await self.get_single_pdf_info(pmid)

        if not info.has_pdf or not info.pdf_url:
            return None, "PDF not available for this paper"

        session = await self._get_session()

        try:
            async with session.get(info.pdf_url) as response:
                if response.status != 200:
                    return None, f"Failed to download PDF: HTTP {response.status}"

                pdf_bytes = await response.read()
                filename = f"{info.pmcid or pmid}.pdf"

                return pdf_bytes, filename
        except Exception as e:
            logger.error(f"Error downloading PDF for PMID {pmid}: {e}")
            return None, f"Error downloading PDF: {str(e)}"


# Singleton instance
_pmc_service: Optional[PMCService] = None


def get_pmc_service() -> PMCService:
    """Get or create PMC service instance"""
    global _pmc_service
    if _pmc_service is None:
        _pmc_service = PMCService()
    return _pmc_service
