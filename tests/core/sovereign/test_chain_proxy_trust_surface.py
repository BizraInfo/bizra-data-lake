"""Trust-surface chain proxy guard — /v1/chain → cognition-gateway.

Sprint: Node0 Closure — row 6 (trust_surface) binding (2026-04-21).

The Rust cognition-gateway exposes the authoritative receipt chain head
at ``GET /chain``. The web face (Dema, ``frontend/src``) needs to reveal
this truth to the operator. The FastAPI ``core/sovereign/api.py`` is the
web face's existing backend (auth middleware, session), so the thinnest
path is a minimal proxy endpoint there: the frontend hits its existing
``VITE_API_URL`` base, the proxy forwards verbatim to the Rust gateway.

This guard enforces the "no shadow state" canon:

1. On success: the proxy returns the gateway's JSON verbatim (no
   reshaping, no simulation).
2. On gateway unreachable (ConnectError / timeout): returns 503 with a
   structured ``gateway_unreachable`` payload so the UI can reveal the
   truth of an offline backend instead of fabricating a healthy chain
   head.
3. On gateway non-200 upstream: returns the upstream status code and
   body in a structured envelope so the UI can distinguish "gateway is
   up but returned an error" from "gateway is down."
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("starlette")

import httpx
from starlette.testclient import TestClient

from core.sovereign.api import create_fastapi_app
from core.sovereign.runtime_types import RuntimeMetrics


# Canonical wire shape emitted by Rust cognition-gateway at GET /chain.
# See bizra-omega/bizra-cognition-gateway/src/main.rs::ReceiptChainHeadDto.
CANONICAL_CHAIN_HEAD = {
    "head": "369319cd83e2419114dc5c3f36467f5665ab7fddac299e01b8d8374302ff676a",
    "length": 9,
    "latestTimestamp": 1745180214,
    "sovereignEnvelopes": 1,
    "sovereignEntries": 5,
}


class _FakeAsyncClient:
    """Context-manager fake for httpx.AsyncClient that patches per-test."""

    def __init__(self, *, response=None, exc=None):
        self._response = response
        self._exc = exc

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def get(self, url, **_):  # noqa: D401 — pytest fake
        if self._exc is not None:
            raise self._exc
        return self._response


def _runtime_for_chain(tmp: Path):
    runtime = MagicMock()
    runtime.config = SimpleNamespace(state_dir=tmp)
    runtime.metrics = RuntimeMetrics(
        queries_processed=0,
        queries_succeeded=0,
        current_snr_score=0.91,
        current_ihsan_score=0.96,
        avg_query_time_ms=0.0,
    )
    runtime.status.return_value = {
        "health": {"status": "healthy"},
        "identity": {"version": "test"},
        "state": {"running": True},
    }
    return runtime


@pytest.fixture
def chain_client(tmp_path: Path) -> TestClient:
    runtime = _runtime_for_chain(tmp_path)
    app = create_fastapi_app(runtime)
    return TestClient(app)


@pytest.fixture(autouse=True)
def _auth_allow_anonymous(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let tests reach /v1/chain without full auth setup.

    The /v1/chain endpoint uses the same _authenticate_http_request
    pattern as /v1/health etc.; setting BIZRA_MCP_ALLOW_ANONYMOUS (the
    flag respected by the auth middleware) is the simplest way to keep
    these tests focused on the proxy contract, not the auth layer
    (which has its own dedicated test suites).
    """
    monkeypatch.setenv("BIZRA_MCP_ALLOW_ANONYMOUS", "1")


class TestChainProxySuccessPath:
    """/v1/chain forwards the gateway's JSON verbatim on 200 upstream."""

    def test_proxy_returns_canonical_chain_head_on_200(
        self, chain_client: TestClient
    ) -> None:
        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = CANONICAL_CHAIN_HEAD
        fake_response.text = ""

        def _factory(**kwargs: Any) -> _FakeAsyncClient:
            return _FakeAsyncClient(response=fake_response)

        with patch("httpx.AsyncClient", _factory):
            resp = chain_client.get("/v1/chain")

        assert resp.status_code == 200
        body = resp.json()
        # Verbatim pass-through — keys NOT reshaped.
        assert body == CANONICAL_CHAIN_HEAD, (
            f"Proxy must forward upstream JSON verbatim. "
            f"No reshaping, no simulation. Got: {body!r}"
        )
        # The canonical head shape is stable.
        assert isinstance(body["head"], str) and len(body["head"]) == 64
        assert isinstance(body["length"], int)


class TestChainProxyGatewayUnreachable:
    """/v1/chain returns honest 503 when the Rust gateway is down."""

    def test_503_on_connect_error(self, chain_client: TestClient) -> None:
        def _factory(**kwargs: Any) -> _FakeAsyncClient:
            return _FakeAsyncClient(
                exc=httpx.ConnectError("connection refused")
            )

        with patch("httpx.AsyncClient", _factory):
            resp = chain_client.get("/v1/chain")

        assert resp.status_code == 503, (
            "When cognition-gateway is unreachable, /v1/chain MUST return "
            "503 with a structured gateway_unreachable payload. It MUST NOT "
            "fabricate a 200 success — no shadow state canon."
        )
        body = resp.json()
        assert body["status"] == "gateway_unreachable"
        assert "gateway_url" in body
        assert body["error"] == "ConnectError"

    def test_503_on_timeout(self, chain_client: TestClient) -> None:
        def _factory(**kwargs: Any) -> _FakeAsyncClient:
            return _FakeAsyncClient(
                exc=httpx.ConnectTimeout("timed out")
            )

        with patch("httpx.AsyncClient", _factory):
            resp = chain_client.get("/v1/chain")

        assert resp.status_code == 503
        assert resp.json()["error"] == "ConnectTimeout"


class TestChainProxyGatewayNon200:
    """/v1/chain reveals upstream errors honestly."""

    def test_upstream_500_is_surfaced(self, chain_client: TestClient) -> None:
        fake_response = MagicMock()
        fake_response.status_code = 500
        fake_response.text = "internal gateway error"

        def _factory(**kwargs: Any) -> _FakeAsyncClient:
            return _FakeAsyncClient(response=fake_response)

        with patch("httpx.AsyncClient", _factory):
            resp = chain_client.get("/v1/chain")

        assert resp.status_code == 500, (
            "Upstream 500 must surface as 500 — not masked as 200."
        )
        body = resp.json()
        assert body["status"] == "gateway_non_200"
        assert body["upstream_status"] == 500
        assert "internal gateway error" in body["upstream_body"]


class TestChainProxyGatewayUrlOverride:
    """/v1/chain respects BIZRA_COGNITION_GATEWAY_URL env var."""

    def test_gateway_url_env_override(
        self, chain_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured_urls: list[str] = []

        async def _spy_get(self, url: str, **_):
            captured_urls.append(url)
            # Raise to short-circuit — we only care about the URL.
            raise httpx.ConnectError("unused")

        monkeypatch.setenv(
            "BIZRA_COGNITION_GATEWAY_URL", "http://override-gateway.test:1234"
        )

        def _factory(**kwargs: Any):
            client = _FakeAsyncClient(exc=httpx.ConnectError("unused"))
            # Capture the URL the proxy attempts
            original_get = client.get

            async def spy(url, **kw):
                captured_urls.append(url)
                return await original_get(url, **kw)

            client.get = spy  # type: ignore[method-assign]
            return client

        with patch("httpx.AsyncClient", _factory):
            chain_client.get("/v1/chain")

        assert captured_urls, "proxy did not attempt an upstream request"
        assert captured_urls[0] == "http://override-gateway.test:1234/chain", (
            f"Gateway URL override not respected. Attempted: {captured_urls[0]}"
        )
