"""
Token API Endpoint Tests — Validates /v1/token/* routes.

Tests FastAPI endpoints using TestClient (no real HTTP server needed).
"""

from __future__ import annotations

import pytest


class TestTokenAPIRoutes:
    """Verify token endpoints return valid responses."""

    @pytest.fixture
    def app(self):
        """Create FastAPI app with token routes."""
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig, RuntimeMode

        config = RuntimeConfig(mode=RuntimeMode.MINIMAL, autonomous_enabled=False)
        runtime = SovereignRuntime(config)

        try:
            from core.sovereign.api import create_fastapi_app
            return create_fastapi_app(runtime)
        except ImportError:
            pytest.skip("FastAPI not available")

    @pytest.fixture
    def client(self, app, tmp_path):
        """Create test client with isolated token ledger."""
        from unittest.mock import patch

        from core.token.ledger import TokenLedger as _RealLedger

        def _factory(*args, **kwargs):
            return _RealLedger(
                db_path=tmp_path / "token_test.db",
                log_path=tmp_path / "token_test.jsonl",
            )

        try:
            from starlette.testclient import TestClient

            with patch("core.token.ledger.TokenLedger", _factory):
                yield TestClient(app)
        except ImportError:
            pytest.skip("starlette not available")

    def test_token_balance_endpoint(self, client):
        """GET /v1/token/balance returns valid structure."""
        resp = client.get("/v1/token/balance?account=BIZRA-00000000")
        assert resp.status_code == 200
        data = resp.json()
        assert "account" in data
        assert "balances" in data
        assert data["account"] == "BIZRA-00000000"

    def test_token_supply_endpoint(self, client):
        """GET /v1/token/supply returns supply info."""
        resp = client.get("/v1/token/supply")
        assert resp.status_code == 200
        data = resp.json()
        assert "year" in data
        assert "supply" in data
        assert "ledger_valid" in data

    def test_token_verify_endpoint(self, client):
        """GET /v1/token/verify returns chain integrity."""
        resp = client.get("/v1/token/verify")
        assert resp.status_code == 200
        data = resp.json()
        assert "valid" in data
        assert data["valid"] is True

    def test_token_balance_unknown_account(self, client):
        """Unknown account returns empty balances, not error."""
        resp = client.get("/v1/token/balance?account=UNKNOWN-ACCOUNT")
        assert resp.status_code == 200
        data = resp.json()
        assert data["account"] == "UNKNOWN-ACCOUNT"
        assert data["balances"] == {}
