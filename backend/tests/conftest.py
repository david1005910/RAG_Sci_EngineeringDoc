"""Pytest configuration and fixtures"""

import asyncio
import pytest
from typing import Generator, AsyncGenerator
from httpx import AsyncClient, ASGITransport

from src.main import app


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Create a test HTTP client without database dependency."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sample_paper_data():
    """Sample paper data for testing."""
    return {
        "pmid": "12345678",
        "title": "Test Paper Title",
        "abstract": "This is a test abstract for unit testing.",
        "authors": ["Author One", "Author Two"],
        "journal": "Test Journal",
        "keywords": ["test", "sample"],
    }
