"""Tests for API endpoints"""

import pytest
from httpx import AsyncClient


class TestHealthEndpoint:
    """Test health check endpoint"""

    @pytest.mark.asyncio
    async def test_health_check(self, client: AsyncClient):
        """Test that health endpoint returns healthy status"""
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestRootEndpoint:
    """Test root endpoint"""

    @pytest.mark.asyncio
    async def test_root(self, client: AsyncClient):
        """Test that root endpoint returns app info"""
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data


class TestSearchEndpoint:
    """Test search endpoint"""

    @pytest.mark.asyncio
    async def test_search_empty_query(self, client: AsyncClient):
        """Test search with empty results"""
        response = await client.post(
            "/api/v1/search",
            json={"query": "test query", "limit": 10}
        )
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "results" in data


class TestChatEndpoint:
    """Test chat endpoint"""

    @pytest.mark.asyncio
    async def test_chat_query(self, client: AsyncClient):
        """Test chat query endpoint"""
        response = await client.post(
            "/api/v1/chat/query",
            json={"question": "What is CRISPR?"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
