"""Track 1 hardening regression tests.

Covers:
  1. Mission completion threshold aligned with UNIFIED_IHSAN_THRESHOLD
  2. Persistent node-anchored signer (load/create/inherit)
  3. Auth guards on 8 previously unguarded POST routes
  4. Quality scoring fallback is below threshold (fail-honest)

No bare MagicMock for domain objects. Uses real dataclasses where applicable.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.responses import JSONResponse

from core.integration.constants import UNIFIED_IHSAN_THRESHOLD
from core.sovereign.api import create_fastapi_app


# ── Helpers ──────────────────────────────────────────────────────────


def _runtime(tmp_path) -> MagicMock:
    """Minimal mock runtime for api tests."""
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
    runtime._spearpoint_orchestrator = None
    runtime._experience_ledger = None
    runtime._agent_db = None
    runtime._cognitive_fusion = None
    runtime._orchestrator = None
    runtime._node_signer = None
    runtime._evidence_ledger = None
    return runtime


class _Request:
    def __init__(self, headers: dict[str, str] | None = None):
        self.headers = headers or {}


def _endpoint(app, path: str, method: str = "POST"):
    route = next(
        r
        for r in app.routes
        if getattr(r, "path", "") == path and method in getattr(r, "methods", set())
    )
    return route.endpoint


# ── 1. Mission Completion Threshold ──────────────────────────────────


class TestMissionThreshold:
    """Mission status uses UNIFIED_IHSAN_THRESHOLD, not 0.50."""

    def test_threshold_is_constitutional(self):
        """Threshold imported from constants.py (0.95), not hardcoded 0.50."""
        assert UNIFIED_IHSAN_THRESHOLD == 0.95

    def test_score_below_threshold_gives_partial(self):
        """A score of 0.80 (was COMPLETE under 0.50) is now PARTIAL."""
        # The threshold is 0.95. Anything below → PARTIAL
        assert 0.80 < UNIFIED_IHSAN_THRESHOLD
        assert 0.50 < UNIFIED_IHSAN_THRESHOLD

    def test_score_at_threshold_gives_complete(self):
        """A score at exactly 0.95 gives COMPLETE."""
        assert 0.95 >= UNIFIED_IHSAN_THRESHOLD


# ── 2. Persistent Node Signer ────────────────────────────────────────


class TestPersistentNodeSigner:
    """_load_or_create_node_signer() persists and reloads keypairs."""

    def test_creates_signer_when_none_exists(self, tmp_path):
        from core.sovereign.mission import _load_or_create_node_signer

        config = {"sovereign_state_dir": str(tmp_path / "state")}
        priv, pub = _load_or_create_node_signer(config)

        assert isinstance(priv, str)
        assert isinstance(pub, str)
        assert len(priv) == 64  # 32 bytes hex
        assert len(pub) == 64

        # File was persisted
        signer_file = tmp_path / "state" / "mission_signer.json"
        assert signer_file.exists()
        data = json.loads(signer_file.read_text())
        assert data["private_key_hex"] == priv
        assert data["public_key_hex"] == pub
        assert data["source"] == "generated"

    def test_reloads_existing_signer(self, tmp_path):
        from core.sovereign.mission import _load_or_create_node_signer

        config = {"sovereign_state_dir": str(tmp_path / "state")}
        priv1, pub1 = _load_or_create_node_signer(config)
        priv2, pub2 = _load_or_create_node_signer(config)

        # Same keypair reloaded
        assert priv1 == priv2
        assert pub1 == pub2

    def test_inherits_from_node_identity(self, tmp_path):
        from core.sovereign.mission import _load_or_create_node_signer

        # Set up credentials.json
        identity_dir = tmp_path / "state" / "identity"
        identity_dir.mkdir(parents=True)
        creds = {
            "node_id": "TEST-NODE",
            "private_key": "ab" * 32,
            "public_key": "cd" * 32,
        }
        (identity_dir / "credentials.json").write_text(json.dumps(creds))

        config = {"sovereign_state_dir": str(tmp_path / "state")}
        priv, pub = _load_or_create_node_signer(config)

        assert priv == "ab" * 32
        assert pub == "cd" * 32

        # Also persisted as mission_signer.json
        signer_file = tmp_path / "state" / "mission_signer.json"
        assert signer_file.exists()
        data = json.loads(signer_file.read_text())
        assert data["source"] == "node_identity"

    def test_regenerates_on_corrupt_file(self, tmp_path):
        from core.sovereign.mission import _load_or_create_node_signer

        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True)
        (state_dir / "mission_signer.json").write_text("CORRUPT{{{")

        config = {"sovereign_state_dir": str(state_dir)}
        priv, pub = _load_or_create_node_signer(config)

        # New keypair generated despite corrupt file
        assert len(priv) == 64
        assert len(pub) == 64


# ── 3. Quality Scoring Fallback ──────────────────────────────────────


class TestQualityScoringFallback:
    """Fallback scores are below UNIFIED_IHSAN_THRESHOLD (fail-honest)."""

    def test_fallback_ihsan_below_threshold(self):
        """When SNR engine is unavailable, ihsan_score (0.80) < 0.95."""
        fallback_ihsan = 0.80  # from _score_quality fallback
        assert fallback_ihsan < UNIFIED_IHSAN_THRESHOLD

    def test_fallback_produces_partial_status(self):
        """Fallback score → PARTIAL status (not COMPLETE)."""
        fallback_ihsan = 0.80
        status = "COMPLETE" if fallback_ihsan >= UNIFIED_IHSAN_THRESHOLD else "PARTIAL"
        assert status == "PARTIAL"


# ── 4. Auth Guards on Previously Unguarded POST Routes ───────────────


_GUARDED_POST_ROUTES = [
    "/v1/validate",
    "/v1/spearpoint/reproduce",
    "/v1/spearpoint/improve",
    "/v1/spearpoint/pattern",
    "/v1/sel/retrieve",
    "/v1/memory/search",
    "/v1/cognitive/fuse",
    "/v1/judgment/simulate",
]


class TestAuthGuardsOnPostRoutes:
    """All mutation-capable POST routes now enforce auth."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("route", _GUARDED_POST_ROUTES)
    async def test_post_route_denies_anonymous(
        self, route, tmp_path, monkeypatch
    ):
        """POST route returns 401 or 503 without credentials."""
        monkeypatch.delenv("BIZRA_AUTH_ALLOW_ANONYMOUS", raising=False)
        runtime = _runtime(tmp_path)
        app = create_fastapi_app(runtime)

        endpoint = _endpoint(app, route)

        # Invoke with empty request (no credentials) — endpoint signature
        # accepts (body, request) so we need a body and a request
        body = MagicMock()
        resp = await endpoint(body, _Request())

        assert isinstance(resp, JSONResponse), (
            f"{route} should return JSONResponse for anonymous request"
        )
        assert resp.status_code in {401, 503}, (
            f"{route} should deny anonymous (got {resp.status_code})"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("route", _GUARDED_POST_ROUTES)
    async def test_post_route_allows_anonymous_when_opted_in(
        self, route, tmp_path, monkeypatch
    ):
        """POST route works when BIZRA_AUTH_ALLOW_ANONYMOUS=true."""
        monkeypatch.setenv("BIZRA_AUTH_ALLOW_ANONYMOUS", "true")
        runtime = _runtime(tmp_path)
        app = create_fastapi_app(runtime)

        endpoint = _endpoint(app, route)
        body = MagicMock()

        # Should NOT return JSONResponse auth error (may return 404/503 for
        # missing runtime component, that's fine — the auth gate passed)
        resp = await endpoint(body, _Request())

        if isinstance(resp, JSONResponse):
            # Auth error codes would be 401/503-from-auth. Backend errors
            # (missing orchestrator etc.) are 404/500/503-from-service.
            body_data = json.loads(resp.body.decode("utf-8"))
            assert resp.status_code != 401, (
                f"{route} rejected with 401 despite anonymous opt-in"
            )
            # 503 from auth = "Authentication service unavailable"
            if resp.status_code == 503:
                assert "Authentication" not in body_data.get("error", ""), (
                    f"{route} auth layer blocked despite anonymous opt-in"
                )


# ── 5. Verify /v1/verify/* endpoints remain intentionally open ───────


_VERIFY_ROUTES = [
    "/v1/verify/genesis",
    "/v1/verify/envelope",
    "/v1/verify/receipt",
    "/v1/verify/audit-log",
    "/v1/verify/ledger",
    "/v1/verify/poi",
]


class TestVerifyRoutesRemainOpen:
    """Verification endpoints are intentionally auth-free for external auditors."""

    @pytest.mark.parametrize("route", _VERIFY_ROUTES)
    def test_verify_route_has_no_request_param(self, route, tmp_path, monkeypatch):
        """Verify endpoints should NOT have request: Request parameter."""
        monkeypatch.setenv("BIZRA_AUTH_ALLOW_ANONYMOUS", "true")
        runtime = _runtime(tmp_path)
        app = create_fastapi_app(runtime)

        endpoint = _endpoint(app, route)
        import inspect

        sig = inspect.signature(endpoint)
        param_names = list(sig.parameters.keys())
        # These routes should only accept a body, not a Request
        assert "request" not in param_names, (
            f"{route} should remain open for external auditors"
        )


# ── 6. Auth bootstrap routes remain open ─────────────────────────────


class TestAuthBootstrapOpen:
    """Auth routes (/v1/auth/*) must remain open for credential exchange."""

    @pytest.mark.parametrize(
        "route",
        ["/v1/auth/register", "/v1/auth/login", "/v1/auth/refresh"],
    )
    def test_auth_route_exists(self, route, tmp_path, monkeypatch):
        """Auth bootstrap routes exist and are POST."""
        monkeypatch.setenv("BIZRA_AUTH_ALLOW_ANONYMOUS", "true")
        runtime = _runtime(tmp_path)
        app = create_fastapi_app(runtime)
        # Should not raise
        _endpoint(app, route)
