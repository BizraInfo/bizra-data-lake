"""Browser research adapter: MCP, direct HTTP, and mock."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


_MOCK_RESULTS: tuple[SearchResult, ...] = (
    SearchResult(
        title="a16z Crypto",
        url="https://a16zcrypto.com",
        snippet="Active investor in decentralized AI and cryptographic infrastructure.",
    ),
    SearchResult(
        title="Delphi Ventures",
        url="https://delphidigital.io",
        snippet="Research-driven fund covering the agent economy and open networks.",
    ),
    SearchResult(
        title="Framework Ventures",
        url="https://framework.ventures",
        snippet="Web3 venture firm focused on network-native products and protocols.",
    ),
    SearchResult(
        title="Paradigm",
        url="https://paradigm.xyz",
        snippet=(
            "Engineering-heavy crypto investment firm"
            " with deep technical diligence."
        ),
    ),
    SearchResult(
        title="Polychain Capital",
        url="https://polychain.capital",
        snippet=(
            "Long-running crypto fund investing in"
            " decentralized compute and infra."
        ),
    ),
)


class BrowserMCPClient:
    """Lightweight browser client for mission research subtasks."""

    def __init__(self, mode: str = "mock", mcp_client: Any = None) -> None:
        if mode not in {"mock", "direct", "mcp"}:
            raise ValueError("mode must be one of: mock, direct, mcp")
        self.mode = mode
        self._mcp_client = mcp_client

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        if self.mode == "mock":
            return list(_MOCK_RESULTS[: max(1, limit)])

        if self.mode == "mcp":
            results = await self._search_mcp(query, limit)
            if results:
                return results
            logger.warning("MCP search unavailable; falling back to mock results")
            return list(_MOCK_RESULTS[: max(1, limit)])

        return await self._search_direct(query, limit)

    async def fetch_page(self, url: str) -> str:
        if self.mode == "mock":
            return f"[mock page] {url}"

        try:
            import httpx

            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.text
        except Exception as exc:
            logger.warning("Page fetch failed for %s (%s)", url, exc)
            return ""

    async def research(self, query: str) -> dict[str, Any]:
        results = await self.search(query, limit=5)
        pages = []
        for result in results[:3]:
            pages.append(
                {
                    "title": result.title,
                    "url": result.url,
                    "snippet": result.snippet,
                }
            )

        summary = " ; ".join(item["title"] for item in pages)
        return {
            "query": query,
            "results": pages,
            "summary": f"Top matches: {summary}" if summary else "No matches",
            "mode": self.mode,
        }

    async def _search_mcp(self, query: str, limit: int) -> list[SearchResult]:
        client = self._mcp_client
        if client is None:
            return []

        try:
            raw = await client.search(query=query, limit=limit)
        except Exception:
            return []

        normalized: list[SearchResult] = []
        for item in raw or []:
            title = str(item.get("title", "")).strip()
            url = str(item.get("url", "")).strip()
            snippet = str(item.get("snippet", "")).strip()
            if title and url:
                normalized.append(SearchResult(title=title, url=url, snippet=snippet))

        return normalized[: max(1, limit)]

    async def _search_direct(self, query: str, limit: int) -> list[SearchResult]:
        try:
            import httpx

            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                ddg_url = "https://lite.duckduckgo.com/lite/"
                response = await client.get(
                    ddg_url, params={"q": query}
                )
                response.raise_for_status()
                parsed = self._parse_ddg_lite(response.text, limit)
                if parsed:
                    return parsed
        except Exception as exc:
            logger.warning("Direct browser search failed (%s)", exc)

        return list(_MOCK_RESULTS[: max(1, limit)])

    @staticmethod
    def _parse_ddg_lite(html: str, limit: int) -> list[SearchResult]:
        pattern = re.compile(r'<a[^>]+href="(https?://[^"]+)"[^>]*>([^<]+)</a>')
        matches = pattern.findall(html)

        results: list[SearchResult] = []
        for url, raw_title in matches:
            if "duckduckgo.com" in url:
                continue
            title = re.sub(r"\s+", " ", raw_title).strip()
            if not title:
                continue
            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet=f"Search match for '{title}'",
                )
            )
            if len(results) >= max(1, limit):
                break
        return results


__all__ = ["BrowserMCPClient", "SearchResult"]
