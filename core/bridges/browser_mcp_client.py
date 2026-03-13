"""Browser research adapter: Brave Search, MCP, direct HTTP, and mock."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import re
import socket
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ── SSRF Protection ──────────────────────────────────────────────────
# Standing on Giants: OWASP SSRF Prevention Cheat Sheet
# Validates URLs before any outbound HTTP fetch to prevent Server-Side
# Request Forgery attacks against internal/private network resources.

_ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})


class SSRFValidationError(ValueError):
    """Raised when a URL targets a private/reserved network range."""


def _is_private_ip(addr: str) -> bool:
    """Return True if *addr* resolves to a private or reserved IP range."""
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
    )


def _validate_url(url: str) -> str:
    """Validate a URL against SSRF attack surfaces.

    Checks:
      1. Only http/https schemes are allowed.
      2. Hostname must resolve to a public (non-private) IP address.
      3. Explicit IP literals in the URL are also checked.

    Returns the validated URL (unchanged) on success.
    Raises :class:`SSRFValidationError` on failure.
    """
    parsed = urlparse(url)

    # 1. Scheme allowlist
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise SSRFValidationError(
            f"URL scheme '{parsed.scheme}' not allowed; must be http or https"
        )

    hostname = parsed.hostname
    if not hostname:
        raise SSRFValidationError("URL has no hostname")

    # 2. Check IP literal
    try:
        ip = ipaddress.ip_address(hostname)
        if _is_private_ip(str(ip)):
            raise SSRFValidationError(f"URL targets private/reserved IP: {ip}")
        return url
    except ValueError:
        pass  # hostname is not an IP literal — resolve via DNS

    # 3. DNS resolution check
    try:
        resolved = socket.getaddrinfo(
            hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM
        )
    except socket.gaierror:
        raise SSRFValidationError(f"Cannot resolve hostname: {hostname}")

    for _family, _type, _proto, _canonname, sockaddr in resolved:
        addr = sockaddr[0]
        if _is_private_ip(addr):
            raise SSRFValidationError(
                f"Hostname '{hostname}' resolves to private/reserved IP: {addr}"
            )

    return url


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
            "Engineering-heavy crypto investment firm" " with deep technical diligence."
        ),
    ),
    SearchResult(
        title="Polychain Capital",
        url="https://polychain.capital",
        snippet=(
            "Long-running crypto fund investing in" " decentralized compute and infra."
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

        # direct mode: Brave Search → DDG → mock
        return await self._search_direct(query, limit)

    async def fetch_page(self, url: str) -> str:
        if self.mode == "mock":
            return f"[mock page] {url}"

        try:
            _validate_url(url)
        except SSRFValidationError as exc:
            logger.warning("URL blocked by SSRF guard: %s (%s)", url, exc)
            return ""

        try:
            import httpx

            # Disable automatic redirects to prevent redirect-to-private SSRF.
            # Each redirect target is validated before following.
            async with httpx.AsyncClient(
                timeout=10.0, follow_redirects=False
            ) as client:
                response = await client.get(url)

                # Manually follow up to 5 redirects with SSRF validation
                redirects_left = 5
                while response.is_redirect and redirects_left > 0:
                    redirect_url = (
                        str(response.next_request.url)
                        if response.next_request
                        else None
                    )
                    if redirect_url is None:
                        break
                    try:
                        _validate_url(redirect_url)
                    except SSRFValidationError as exc:
                        logger.warning(
                            "Redirect blocked by SSRF guard: %s (%s)", redirect_url, exc
                        )
                        return ""
                    response = await client.get(redirect_url)
                    redirects_left -= 1

                response.raise_for_status()
                return response.text
        except (
            asyncio.CancelledError,
            RuntimeError,
            OSError,
        ) as exc:  # SEC-003 — async boundary
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
        except (
            asyncio.CancelledError,
            RuntimeError,
            OSError,
        ):  # SEC-003 — async boundary
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
        # 1. Try Brave Search API (best quality, requires BRAVE_API_KEY)
        brave_results = await self._search_brave(query, limit)
        if brave_results:
            return brave_results

        # 2. Try DuckDuckGo Lite (no API key needed)
        ddg_results = await self._search_ddg(query, limit)
        if ddg_results:
            return ddg_results

        # 3. Fall back to mock data
        return list(_MOCK_RESULTS[: max(1, limit)])

    async def _search_brave(self, query: str, limit: int) -> list[SearchResult]:
        """Search via Brave Search API (requires BRAVE_API_KEY env var)."""
        api_key = os.environ.get("BRAVE_API_KEY", "")
        if not api_key:
            return []

        try:
            import httpx

            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": min(limit, 20)},
                    headers={
                        "X-Subscription-Token": api_key,
                        "Accept": "application/json",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            results: list[SearchResult] = []
            for item in data.get("web", {}).get("results", []):
                title = str(item.get("title", "")).strip()
                url = str(item.get("url", "")).strip()
                snippet = str(item.get("description", "")).strip()
                if title and url:
                    results.append(SearchResult(title=title, url=url, snippet=snippet))
                if len(results) >= limit:
                    break

            if results:
                logger.info("Brave Search returned %d results", len(results))
            return results
        except Exception as exc:  # noqa: BLE001 — boundary boundary
            logger.warning("Brave Search failed (%s)", exc)
            return []

    async def _search_ddg(self, query: str, limit: int) -> list[SearchResult]:
        """Search via DuckDuckGo Lite (no API key needed)."""
        try:
            import httpx

            ddg_url = "https://lite.duckduckgo.com/lite/"
            _validate_url(ddg_url)

            async with httpx.AsyncClient(
                timeout=15.0, follow_redirects=False
            ) as client:
                response = await client.post(
                    ddg_url,
                    data={"q": query},
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                response.raise_for_status()
                parsed = self._parse_ddg_lite(response.text, limit)
                if parsed:
                    return parsed
        except SSRFValidationError as exc:
            logger.warning("DDG URL blocked by SSRF guard (%s)", exc)
        except (OSError, ValueError) as exc:  # SEC-003 — network boundary
            logger.warning("DDG search failed (%s)", exc)

        return []

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


__all__ = ["BrowserMCPClient", "SSRFValidationError", "SearchResult"]
