from __future__ import annotations

import pytest

from core.bridges.browser_mcp_client import BrowserMCPClient, SearchResult


def test_search_result_dataclass() -> None:
    result = SearchResult("Title", "https://example.com", "Snippet")
    assert result.title == "Title"
    assert result.url.startswith("https://")


@pytest.mark.asyncio
async def test_mock_mode_search_is_deterministic() -> None:
    client = BrowserMCPClient(mode="mock")
    first = await client.search("decentralized ai vcs", limit=5)
    second = await client.search("anything", limit=5)

    assert first == second
    assert len(first) == 5


@pytest.mark.asyncio
async def test_mock_mode_respects_limit() -> None:
    client = BrowserMCPClient(mode="mock")
    results = await client.search("query", limit=2)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_research_returns_structured_payload() -> None:
    client = BrowserMCPClient(mode="mock")
    payload = await client.research("top vc firms")

    assert payload["query"] == "top vc firms"
    assert payload["mode"] == "mock"
    assert len(payload["results"]) == 3
    assert "summary" in payload


@pytest.mark.asyncio
async def test_fetch_page_mock_mode() -> None:
    client = BrowserMCPClient(mode="mock")
    content = await client.fetch_page("https://example.com")
    assert "mock page" in content


def test_invalid_mode_raises() -> None:
    with pytest.raises(ValueError):
        BrowserMCPClient(mode="invalid")
