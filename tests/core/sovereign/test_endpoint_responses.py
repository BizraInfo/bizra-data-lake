"""Phase 77 — Functional endpoint response tests.

Validates that each /v1/* endpoint consumed by Terminal v1 returns
the correct response shape when called with valid context.

These tests close the gap identified in the SAPE audit:
- Contract integrity tests verify auth guard PRESENCE
- These tests verify handler BEHAVIOR (response shape, status code)

Standing on Giants: Shannon (SNR), Al-Khwarizmi (determinism)
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.testclient import TestClient

# ── Fixtures ─────────────────────────────────────────────────


def _mock_seed_potential():
    """Return a realistic SeedPotential dataclass instance."""
    from core.sovereign.seed_engine import SeedPotential

    return SeedPotential(
        sovereignty_score=0.42,
        tier="SPROUT",
        tier_progress=0.68,
        episodes_total=25,
        episodes_qualified=18,
        qualification_rate=0.72,
        reward_ema=0.85,
        streak=3,
        compiled=False,
        converged=False,
        chain_valid=True,
        potential_unlocked=0.42,
        potential_remaining=0.58,
        weakest_dimension="consistency",
        growth_velocity=0.03,
        last_receipt_hash="abc123",
    )


def _mock_runtime(state_dir: Path) -> MagicMock:
    """Runtime mock with enough state for all Terminal v1 endpoints."""
    runtime = MagicMock()
    runtime.config = SimpleNamespace(state_dir=state_dir)
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

    # ── Seed engine mock (required by /v1/seed/*, /v1/node/*, /v1/network/*) ──
    seed_engine = MagicMock()
    seed_engine.potential.return_value = _mock_seed_potential()
    seed_engine.recent_episodes.return_value = []
    seed_engine._dimension_scores = {}
    seed_engine._genesis_ts = None
    runtime._seed_engine = seed_engine
    # Ensure lazy-init path is taken for NodeValueEngine (not MagicMock)
    runtime._node_value_engine = None

    return runtime


@pytest.fixture()
def app(tmp_path: Path, monkeypatch):
    # Enable anonymous auth for functional testing
    monkeypatch.setenv("BIZRA_AUTH_ALLOW_ANONYMOUS", "1")
    from core.sovereign.api import create_fastapi_app

    return create_fastapi_app(_mock_runtime(tmp_path))


@pytest.fixture()
def client(app):
    return TestClient(app, raise_server_exceptions=False)


# ── PUBLIC endpoints (no auth needed) ────────────────────────


class TestPublicEndpoints:
    """Endpoints that should return 200 without auth."""

    def test_health(self, client):
        r = client.get("/v1/health")
        # Health may return 200 or 500 depending on subsystem init
        assert r.status_code in (200, 500)
        if r.status_code == 200:
            data = r.json()
            assert "status" in data

    def test_status(self, client):
        r = client.get("/v1/status")
        assert r.status_code == 200

    def test_cognitive_status(self, client):
        r = client.get("/v1/cognitive/status")
        assert r.status_code == 200
        data = r.json()
        # Backend returns subsystem availability, not the simplified shape
        # Frontend sovereign-client maps this
        assert isinstance(data, dict)

    def test_token_supply(self, client):
        r = client.get("/v1/token/supply")
        assert r.status_code == 200
        data = r.json()
        # Backend wraps supply under "supply" key with SEED/IMPT breakdown
        assert isinstance(data, dict)
        assert "supply" in data or "total_supply" in data

    def test_metrics(self, client):
        r = client.get("/v1/metrics")
        assert r.status_code == 200

    def test_sat_stats(self, client):
        r = client.get("/v1/sat/stats")
        assert r.status_code == 200


# ── AUTHENTICATED endpoints (anon auth enabled) ──────────────


class TestAuthenticatedEndpoints:
    """Endpoints that require auth — anon auth enabled for testing."""

    def test_seed_potential_responds(self, client):
        r = client.get("/v1/seed/potential")
        # Should return 200 with anon auth
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, dict)
        # Must have sovereignty_score (core contract field)
        assert "sovereignty_score" in data

    def test_seed_episodes_responds(self, client):
        r = client.get("/v1/seed/episodes")
        assert r.status_code == 200
        data = r.json()
        # Backend wraps under {count, episodes}
        assert "episodes" in data or "count" in data

    def test_node_value_responds(self, client):
        r = client.get("/v1/node/value")
        assert r.status_code == 200
        data = r.json()
        assert "composite" in data

    def test_node_lifecycle_responds(self, client):
        r = client.get("/v1/node/lifecycle")
        assert r.status_code == 200
        data = r.json()
        assert "current_stage" in data

    def test_network_effect_responds(self, client):
        r = client.get("/v1/network/effect?nodes=100")
        assert r.status_code == 200
        data = r.json()
        assert "nodes" in data

    def test_network_milestones_responds(self, client):
        r = client.get("/v1/network/milestones")
        assert r.status_code == 200
        data = r.json()
        assert "milestones" in data
        assert isinstance(data["milestones"], list)

    def test_token_balance_responds(self, client):
        r = client.get("/v1/token/balance")
        assert r.status_code == 200
        data = r.json()
        # Backend returns {account, balances: {SEED: {...}, IMPT: {...}}}
        assert "balances" in data or "balance" in data

    def test_constitutional_status_responds(self, client):
        r = client.get("/v1/constitutional/status")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, dict)

    def test_terminal_state_responds(self, client):
        r = client.get("/v1/terminal/state")
        # 200 if terminal module available, 503 otherwise
        assert r.status_code in (200, 503)

    def test_terminal_briefing_responds(self, client):
        r = client.get("/v1/terminal/briefing")
        assert r.status_code in (200, 503)

    def test_memory_stats_responds(self, client):
        r = client.get("/v1/memory/stats")
        assert r.status_code == 200
        data = r.json()
        # Memory stats shape depends on living_memory availability
        assert isinstance(data, dict)

    def test_onboarding_state_responds(self, client):
        r = client.get("/v1/onboarding/state")
        assert r.status_code == 200
        data = r.json()
        assert "step" in data


# ── MISSION endpoint ────────────────────────────────────────


class TestMissionEndpoint:
    """POST /v1/plan — the golden path."""

    def test_plan_returns_receipt(self, client):
        """POST /v1/plan must return a receipt with mission_id."""
        r = client.post(
            "/v1/plan",
            json={"description": "test mission", "source": "test"},
        )
        assert r.status_code == 200
        data = r.json()
        assert "mission_id" in data
        assert "status" in data


# ── VERIFY endpoints ────────────────────────────────────────


class TestVerifyEndpoints:
    """Public POST verify endpoints."""

    def test_verify_genesis_responds(self, client):
        """POST /v1/verify/genesis returns a decision."""
        r = client.post("/v1/verify/genesis")
        assert r.status_code == 200
        data = r.json()
        # Backend returns decision (ACCEPTED/REJECTED), not simple valid bool
        assert "decision" in data or "valid" in data

    def test_verify_envelope_accepts_json(self, client):
        """POST /v1/verify/envelope requires proper body."""
        r = client.post(
            "/v1/verify/envelope",
            json={"envelope_hash": "deadbeef", "signature": "test"},
        )
        # 200 or 422 (validation) are both acceptable responses
        assert r.status_code in (200, 422)

    def test_verify_receipt_accepts_json(self, client):
        """POST /v1/verify/receipt requires proper body."""
        r = client.post(
            "/v1/verify/receipt",
            json={"receipt_hash": "deadbeef"},
        )
        assert r.status_code in (200, 422)


# ── Fail-closed auth behavior ───────────────────────────────


class TestAuthFailClosed:
    """AUTHENTICATED routes must reject without valid auth when anon disabled."""

    PROTECTED_ROUTES = [
        ("GET", "/v1/seed/potential"),
        ("GET", "/v1/seed/episodes"),
        ("GET", "/v1/node/value"),
        ("GET", "/v1/node/lifecycle"),
        ("GET", "/v1/network/effect"),
        ("GET", "/v1/network/milestones"),
        ("GET", "/v1/token/balance"),
        ("GET", "/v1/constitutional/status"),
        ("GET", "/v1/terminal/state"),
        ("GET", "/v1/terminal/briefing"),
        ("GET", "/v1/memory/stats"),
        ("GET", "/v1/onboarding/state"),
    ]

    @pytest.fixture(autouse=True)
    def _disable_anon(self, monkeypatch):
        """Disable anon auth for fail-closed tests."""
        monkeypatch.delenv("BIZRA_AUTH_ALLOW_ANONYMOUS", raising=False)

    @pytest.fixture()
    def strict_app(self, tmp_path, monkeypatch):
        """App with anon auth disabled."""
        monkeypatch.delenv("BIZRA_AUTH_ALLOW_ANONYMOUS", raising=False)
        from core.sovereign.api import create_fastapi_app

        return create_fastapi_app(_mock_runtime(tmp_path))

    @pytest.fixture()
    def strict_client(self, strict_app):
        return TestClient(strict_app, raise_server_exceptions=False)

    @pytest.mark.parametrize("method,path", PROTECTED_ROUTES)
    def test_rejects_unauthenticated(self, strict_client, method: str, path: str):
        """Protected routes must not return 200 without auth."""
        r = strict_client.request(method, path)
        assert (
            r.status_code != 200
        ), f"{method} {path} returned 200 without auth — fail-open vulnerability"


# ── Response shape contracts (frontend ↔ backend binding) ──────


class TestResponseShapeContracts:
    """Validate exact field keys that the frontend sovereign-client expects.

    These tests prevent type drift between backend Python handlers and
    the TypeScript types in sovereign-client.ts.  If a backend change
    removes or renames a field the frontend depends on, this class FAILS.
    """

    def test_token_balance_shape(self, client):
        """Backend must return {account, balances} for token/balance."""
        r = client.get("/v1/token/balance")
        assert r.status_code == 200
        data = r.json()
        assert "account" in data, "Missing 'account' — frontend TokenBalance needs it"
        assert "balances" in data, "Missing 'balances' — frontend TokenBalance needs it"
        assert isinstance(data["balances"], dict)

    def test_cognitive_status_shape(self, client):
        """Backend must return subsystem booleans for cognitive/status."""
        r = client.get("/v1/cognitive/status")
        assert r.status_code == 200
        data = r.json()
        assert "cognitive_fusion_available" in data
        assert "subsystems" in data
        subs = data["subsystems"]
        for key in ("moe_router", "hrm_engine", "hypergraph_rag", "northstar_engine"):
            assert key in subs, f"Missing subsystem key '{key}'"

    def test_seed_potential_shape(self, client):
        """Backend must return all 16 SeedPotential fields."""
        r = client.get("/v1/seed/potential")
        assert r.status_code == 200
        data = r.json()
        required = {
            "sovereignty_score",
            "tier",
            "tier_progress",
            "episodes_total",
            "episodes_qualified",
            "qualification_rate",
            "reward_ema",
            "streak",
            "compiled",
            "converged",
            "chain_valid",
            "potential_unlocked",
            "potential_remaining",
            "weakest_dimension",
            "growth_velocity",
            "last_receipt_hash",
        }
        missing = required - set(data.keys())
        assert not missing, f"SeedPotential missing fields: {missing}"

    def test_seed_episodes_shape(self, client):
        """Backend must return {count, episodes} for seed/episodes."""
        r = client.get("/v1/seed/episodes")
        assert r.status_code == 200
        data = r.json()
        assert "count" in data, "Missing 'count' — frontend expects {count, episodes}"
        assert "episodes" in data
        assert isinstance(data["episodes"], list)

    def test_verify_genesis_shape(self, client):
        """Backend verify/genesis must return VerifierResponse shape."""
        r = client.post("/v1/verify/genesis")
        assert r.status_code == 200
        data = r.json()
        assert (
            "decision" in data
        ), "Missing 'decision' — frontend VerifierResponse needs it"
        assert data["decision"] in ("APPROVED", "REJECTED", "QUARANTINED")
        assert "reason_codes" in data
        assert "artifacts" in data

    def test_node_value_shape(self, client):
        """Backend must return 5-factor composite + tier for node/value."""
        r = client.get("/v1/node/value")
        assert r.status_code == 200
        data = r.json()
        for key in ("composite", "tier", "human_stage", "timestamp"):
            assert key in data, f"Missing '{key}' — frontend NodeValue needs it"

    def test_lifecycle_shape(self, client):
        """Backend must return lifecycle progression fields."""
        r = client.get("/v1/node/lifecycle")
        assert r.status_code == 200
        data = r.json()
        for key in ("current_stage", "rank", "sovereignty_score"):
            assert key in data, f"Missing '{key}' — frontend LifecycleStage needs it"
