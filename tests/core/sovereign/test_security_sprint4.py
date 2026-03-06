"""
Sprint 4 Security Tests — Phase 69
════════════════════════════════════

TDD anchors for:
- Hardcoded credential removal (golden_gems/algebraic_effects.py)
- Auth-gating previously unauthenticated endpoints
- API error sanitization (no internal details in HTTP responses)

Standing on Giants:
- OWASP (2021): API Security Top 10
- Beck (2002): TDD by Example
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.responses import JSONResponse

from core.sovereign.api import create_fastapi_app


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


def _runtime(tmp_path) -> MagicMock:
    """Minimal mock runtime for API testing."""
    runtime = MagicMock()
    runtime.config = SimpleNamespace(state_dir=tmp_path / "state")
    runtime.metrics = MagicMock(to_prometheus=lambda include_help=False: "")
    runtime.status.return_value = {
        "health": {
            "status": "healthy",
            "strict_gate": {"enabled": False, "passed": True},
        },
        "identity": {"version": "test"},
        "state": {"running": True},
        "autonomous": {"running": False},
        "pat_sat": {
            "negotiation_receipt_chain": {
                "verified_end_to_end": False,
                "chain_valid": None,
                "total_negotiation_receipts": 0,
                "latest_sequence": None,
                "latest_entry_hash": None,
                "latest_receipt_id": None,
            }
        },
    }
    runtime.query = AsyncMock(
        return_value=SimpleNamespace(
            query_id="q-test",
            success=True,
            response="ok",
            snr_score=0.9,
            ihsan_score=0.9,
            processing_time_ms=12.0,
            graph_hash=None,
        )
    )
    runtime._experience_ledger = None
    runtime._spearpoint_orchestrator = None
    runtime._judgment_telemetry = None
    runtime._living_memory = None
    return runtime


class _Request:
    """Fake HTTP request for testing endpoint auth."""

    def __init__(self, headers: dict[str, str] | None = None):
        self.headers = headers or {}


def _get_endpoint(app, path: str, method: str = "GET"):
    """Extract an endpoint function from the FastAPI app."""
    for route in app.routes:
        if getattr(route, "path", "") == path:
            methods = getattr(route, "methods", set())
            if method in methods:
                return route.endpoint
    raise ValueError(f"No {method} endpoint found for {path}")


# ═══════════════════════════════════════════════════════════════════
# Test: Hardcoded Credentials Removed
# ═══════════════════════════════════════════════════════════════════


class TestHardcodedCredentials:
    """Verify hardcoded default tokens are removed from AuthHandler."""

    def test_auth_handler_requires_tokens(self) -> None:
        """AuthHandler must raise ValueError when no tokens provided."""
        from golden_gems.algebraic_effects import AuthHandler

        with pytest.raises(ValueError, match="valid_tokens must be provided"):
            AuthHandler()

    def test_auth_handler_accepts_explicit_tokens(self) -> None:
        """AuthHandler works fine when tokens are explicitly provided."""
        from golden_gems.algebraic_effects import AuthHandler

        handler = AuthHandler(valid_tokens={"test_token": ["user"]})
        assert handler.valid_tokens == {"test_token": ["user"]}

    def test_no_default_credentials_in_source(self) -> None:
        """Verify the hardcoded strings are gone from source."""
        import inspect

        from golden_gems.algebraic_effects import AuthHandler

        source = inspect.getsource(AuthHandler.__init__)
        assert "bizra_secret_123" not in source
        assert "user_token_456" not in source


# ═══════════════════════════════════════════════════════════════════
# Test: SEL Episodes Requires Auth
# ═══════════════════════════════════════════════════════════════════


class TestSELEpisodesAuth:
    """Verify /v1/sel/episodes and /v1/sel/episodes/{hash} require auth."""

    @pytest.mark.asyncio
    async def test_sel_episodes_requires_auth(self, tmp_path, monkeypatch) -> None:
        """GET /v1/sel/episodes must reject unauthenticated requests."""
        monkeypatch.delenv("BIZRA_AUTH_ALLOW_ANONYMOUS", raising=False)
        runtime = _runtime(tmp_path)
        app = create_fastapi_app(runtime)
        endpoint = _get_endpoint(app, "/v1/sel/episodes", "GET")

        resp = await endpoint(request=_Request(), limit=50, offset=0)

        assert isinstance(resp, JSONResponse)
        assert resp.status_code in {401, 503}

    @pytest.mark.asyncio
    async def test_sel_episodes_hash_requires_auth(
        self, tmp_path, monkeypatch
    ) -> None:
        """GET /v1/sel/episodes/{hash} must reject unauthenticated requests."""
        monkeypatch.delenv("BIZRA_AUTH_ALLOW_ANONYMOUS", raising=False)
        runtime = _runtime(tmp_path)
        app = create_fastapi_app(runtime)
        endpoint = _get_endpoint(app, "/v1/sel/episodes/{episode_hash}", "GET")

        resp = await endpoint(episode_hash="abc123", request=_Request())

        assert isinstance(resp, JSONResponse)
        assert resp.status_code in {401, 503}


# ═══════════════════════════════════════════════════════════════════
# Test: Telemetry Endpoints Require Auth
# ═══════════════════════════════════════════════════════════════════


class TestTelemetryAuth:
    """Verify stats/judgment/suggestions endpoints require auth."""

    @pytest.mark.asyncio
    async def test_spearpoint_stats_requires_auth(
        self, tmp_path, monkeypatch
    ) -> None:
        """GET /v1/spearpoint/stats must reject unauthenticated requests."""
        monkeypatch.delenv("BIZRA_AUTH_ALLOW_ANONYMOUS", raising=False)
        runtime = _runtime(tmp_path)
        app = create_fastapi_app(runtime)
        endpoint = _get_endpoint(app, "/v1/spearpoint/stats", "GET")

        resp = await endpoint(request=_Request())

        assert isinstance(resp, JSONResponse)
        assert resp.status_code in {401, 503}

    @pytest.mark.asyncio
    async def test_judgment_stats_requires_auth(
        self, tmp_path, monkeypatch
    ) -> None:
        """GET /v1/judgment/stats must reject unauthenticated requests."""
        monkeypatch.delenv("BIZRA_AUTH_ALLOW_ANONYMOUS", raising=False)
        runtime = _runtime(tmp_path)
        app = create_fastapi_app(runtime)
        endpoint = _get_endpoint(app, "/v1/judgment/stats", "GET")

        resp = await endpoint(request=_Request())

        assert isinstance(resp, JSONResponse)
        assert resp.status_code in {401, 503}


# ═══════════════════════════════════════════════════════════════════
# Test: API Error Sanitization
# ═══════════════════════════════════════════════════════════════════


class TestErrorSanitization:
    """Verify API errors don't leak internal details."""

    @pytest.mark.asyncio
    async def test_api_error_no_internal_details(
        self, tmp_path, monkeypatch
    ) -> None:
        """500 errors must not include exception message text."""
        monkeypatch.setenv("BIZRA_AUTH_ALLOW_ANONYMOUS", "true")
        runtime = _runtime(tmp_path)
        # Make query raise an internal error
        runtime.query = AsyncMock(
            side_effect=RuntimeError("secret database connection string")
        )
        app = create_fastapi_app(runtime)

        from core.sovereign.api import QueryRequestModel

        endpoint = _get_endpoint(app, "/v1/query", "POST")
        resp = await endpoint(QueryRequestModel(query="hello"), _Request())

        assert isinstance(resp, JSONResponse)
        assert resp.status_code == 500
        body = json.loads(resp.body.decode())
        # Must NOT contain the internal error message
        assert "secret database" not in body.get("error", "")
        assert "connection string" not in body.get("error", "")
