"""
Integration test for POST /v1/plan — the sovereign mission golden path.

Sprint 1, Task 1.5: Full mission loop integration test.
Blueprint acceptance criterion: `curl /v1/plan` returns receipted result.

Standing on Giants:
- Boyd: OODA loop (observe → orient → decide → act)
- Lamport: Hash-chained evidence with ordering invariant
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")


@pytest.fixture
def plan_client(tmp_path, monkeypatch):
    """Create a test client for the FastAPI app with /v1/plan wired."""
    from httpx import ASGITransport, AsyncClient

    monkeypatch.setenv("BIZRA_AUTH_ALLOW_ANONYMOUS", "true")
    monkeypatch.setenv("SEMANTIC_MEMORY_PATH", str(tmp_path / "memory"))
    monkeypatch.setenv("EVENT_LOG_PATH", str(tmp_path / "events"))
    monkeypatch.setenv(
        "BIZRA_RECEIPT_PRIVATE_KEY_HEX",
        "1111111111111111111111111111111111111111111111111111111111111111",
    )

    from unittest.mock import MagicMock

    runtime = MagicMock()
    runtime.config = MagicMock()
    runtime.config.state_dir = tmp_path / "state"
    runtime.config.state_dir.mkdir(parents=True, exist_ok=True)
    runtime._constitutional_wallets = []
    runtime._constitutional_receipts = []
    runtime._constitutional_proposals = []
    runtime._constitutional_event_log = []
    runtime._constitutional_reflex_cache = {}
    runtime.inference_gateway = None
    # Disable Node0 so legacy tests exercise _submit_mission_to_tick fallback
    runtime._node0 = None
    runtime._organism = None
    runtime._canonical_mode = False

    from core.sovereign.api import create_fastapi_app

    app = create_fastapi_app(runtime)
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://testserver")


@pytest.mark.integration
async def test_plan_returns_receipted_result(plan_client):
    """Blueprint acceptance: POST /v1/plan returns receipted result with Ihsan/SNR."""
    async with plan_client as client:
        resp = await client.post(
            "/v1/plan",
            json={
                "description": "Summarize the current project status",
                "source": "test",
            },
        )
        assert (
            resp.status_code == 200
        ), f"Expected 200, got {resp.status_code}: {resp.text}"

        data = resp.json()
        assert "mission_id" in data
        assert "status" in data
        assert data["status"] in ("COMPLETE", "PARTIAL", "FAILED", "BLOCKED")
        assert "synthesis" in data
        assert "ihsan_score" in data
        assert "snr_score" in data
        assert "receipt_id" in data
        assert "duration_ms" in data
        assert isinstance(data["channels_executed"], list)
        # Contract §8.1: enriched receipt fields always present
        assert "execution_path" in data
        assert "wallet_delta" in data
        assert "reflex_delta" in data
        assert "memory_delta" in data


@pytest.mark.integration
async def test_plan_rejects_empty_description(plan_client):
    """POST /v1/plan with empty description returns 400."""
    async with plan_client as client:
        resp = await client.post("/v1/plan", json={"description": ""})
        assert resp.status_code == 400
        assert "description is required" in resp.json()["error"]


@pytest.mark.integration
async def test_plan_rejects_missing_body(plan_client):
    """POST /v1/plan with no body returns 400 or 422."""
    async with plan_client as client:
        resp = await client.post("/v1/plan", content=b"{}")
        # Empty JSON body → description="" → 400
        assert resp.status_code == 400


@pytest.mark.integration
async def test_plan_includes_channel_breakdown(plan_client):
    """POST /v1/plan result includes per-channel execution breakdown."""
    async with plan_client as client:
        resp = await client.post(
            "/v1/plan",
            json={"description": "Check system health and report status"},
        )
        assert resp.status_code == 200
        data = resp.json()
        for ch in data["channels_executed"]:
            assert "channel" in ch
            assert "success" in ch
            assert "duration_ms" in ch


@pytest.fixture
def plan_runtime(tmp_path, monkeypatch):
    """Return (client, runtime) so tests can inspect runtime state after calls."""
    monkeypatch.setenv("BIZRA_AUTH_ALLOW_ANONYMOUS", "true")
    monkeypatch.setenv("SEMANTIC_MEMORY_PATH", str(tmp_path / "memory"))
    monkeypatch.setenv("EVENT_LOG_PATH", str(tmp_path / "events"))
    monkeypatch.setenv(
        "BIZRA_RECEIPT_PRIVATE_KEY_HEX",
        "1111111111111111111111111111111111111111111111111111111111111111",
    )

    from unittest.mock import MagicMock

    from httpx import ASGITransport, AsyncClient

    runtime = MagicMock()
    runtime.config = MagicMock()
    runtime.config.state_dir = tmp_path / "state"
    runtime.config.state_dir.mkdir(parents=True, exist_ok=True)
    runtime._constitutional_wallets = []
    runtime._constitutional_receipts = []
    runtime._constitutional_proposals = []
    runtime._constitutional_event_log = []
    runtime._constitutional_reflex_cache = {}
    runtime.inference_gateway = None
    # Disable Node0 so this fixture tests the legacy _submit_mission_to_tick path
    runtime._node0 = None
    runtime._organism = None
    runtime._canonical_mode = False

    from core.sovereign.api import create_fastapi_app

    app = create_fastapi_app(runtime)
    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://testserver")
    return client, runtime


@pytest.mark.integration
async def test_plan_system1_reflex_cache_hit(plan_client, monkeypatch):
    """System-1 fast path: pre-loaded reflex returns O(1) cached response."""
    import core.sovereign.api as api_module
    from core.sovereign.reflex_compiler import ReflexCompiler

    compiler = ReflexCompiler()

    # Directly inject a reflex entry (bypasses precipitation)
    compiler.compile_from_candidate(
        pattern_id=compiler._hash_input("what is autopoiesis?"),
        input_template="what is autopoiesis?",
        output_template="Autopoiesis is the self-creation property of living systems.",
        ihsan_score=0.97,
        observation_count=5,
    )

    # Replace the module-level compiler
    original = getattr(api_module, "_reflex_compiler", None)
    api_module._reflex_compiler = compiler
    try:
        async with plan_client as client:
            resp = await client.post(
                "/v1/plan",
                json={"description": "what is autopoiesis?"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["execution_path"] == "SYSTEM_1_CACHE_HIT"
            assert "autopoiesis" in data["synthesis"].lower()
            assert data["ihsan_score"] == 0.97
            assert data["duration_ms"] < 1.0
            assert data["reflex_delta"]["compiled"] is True
    finally:
        api_module._reflex_compiler = original


@pytest.mark.integration
async def test_plan_system2_records_observation(plan_client, monkeypatch):
    """System-2 path records observation for future precipitation."""
    import core.sovereign.api as api_module
    from core.sovereign.reflex_compiler import ReflexCompiler

    compiler = ReflexCompiler()  # empty cache — no hit
    original = getattr(api_module, "_reflex_compiler", None)
    api_module._reflex_compiler = compiler
    try:
        async with plan_client as client:
            resp = await client.post(
                "/v1/plan",
                json={"description": "a unique test query for precipitation"},
            )
            assert resp.status_code == 200
            data = resp.json()
            # Should have gone through System-2
            assert data["execution_path"] == "SYSTEM_2_NOVEL"

            # Compiler should have recorded the observation
            pattern_hash = compiler._hash_input("a unique test query for precipitation")
            assert pattern_hash in compiler._candidates
            candidate = compiler._candidates[pattern_hash]
            assert len(candidate.observations) == 1
    finally:
        api_module._reflex_compiler = original


@pytest.mark.integration
async def test_plan_feeds_constitutional_tick_queue(plan_runtime):
    """Mission result must appear in runtime._constitutional_receipts."""
    client, runtime = plan_runtime
    assert len(runtime._constitutional_receipts) == 0

    async with client:
        resp = await client.post(
            "/v1/plan",
            json={"description": "Test reflex cache wiring"},
        )
        assert resp.status_code == 200

    # Mission result should have been converted to an ActionReceipt
    assert len(runtime._constitutional_receipts) == 1
    receipt = runtime._constitutional_receipts[0]
    assert receipt.action_type == "mission"
    assert receipt.intent_score > 0
    assert receipt.impact_score > 0


@pytest.mark.integration
async def test_plan_verified_proof_mode_returns_reasoning_proof(plan_runtime):
    """Verified missions should surface VRG proof metadata on the receipt."""
    client, runtime = plan_runtime

    async def _reason_verified(query, context=None):
        del query, context
        return SimpleNamespace(
            vrg_root="vrg-root-001",
            verified=True,
            branch_certificates=[
                {"included_in_root": True},
                {"included_in_root": True},
                {"included_in_root": False},
            ],
            receipt=SimpleNamespace(
                receipt_id="proof-receipt-001",
                status=SimpleNamespace(value="ACCEPTED"),
                payload_digest=bytes.fromhex("11" * 32),
                reason="",
            ),
        )

    runtime._got_bridge = SimpleNamespace(reason_verified=_reason_verified)

    async with client:
        resp = await client.post(
            "/v1/plan",
            json={
                "description": "Generate a verified reasoning proof for this mission",
                "source": "terminal",
                "proof_mode": "verified",
            },
        )
        assert resp.status_code == 200
        data = resp.json()

    assert data["reasoning_proof"]["vrg_root"] == "vrg-root-001"
    assert data["reasoning_proof"]["verified"] is True
    assert data["reasoning_proof"]["receipt_id"] == "proof-receipt-001"
    assert data["reasoning_proof"]["surviving_branches"] == 2
    assert data["reasoning_proof"]["branch_count"] == 3


# ─── Node0 Canonical Ingest Authority ────────────────────────────────


@pytest.fixture
def plan_client_with_node0(tmp_path, monkeypatch):
    """Test client with Node0Heartbeat booted and wired to runtime.

    ASGITransport does not trigger lifespan events, so we boot Node0
    in the fixture and wire it to runtime._node0 directly — matching
    what the lifespan does in production.
    """
    from httpx import ASGITransport, AsyncClient

    monkeypatch.setenv("BIZRA_AUTH_ALLOW_ANONYMOUS", "true")
    monkeypatch.setenv("SEMANTIC_MEMORY_PATH", str(tmp_path / "memory"))
    monkeypatch.setenv("EVENT_LOG_PATH", str(tmp_path / "events"))
    monkeypatch.setenv("NODE0_STATE_DIR", str(tmp_path / "api_node0"))
    monkeypatch.setenv(
        "BIZRA_RECEIPT_PRIVATE_KEY_HEX",
        "1111111111111111111111111111111111111111111111111111111111111111",
    )

    from unittest.mock import MagicMock

    from core.node0.heartbeat import Node0Heartbeat

    runtime = MagicMock()
    runtime.config = MagicMock()
    runtime.config.state_dir = tmp_path / "state"
    runtime.config.state_dir.mkdir(parents=True, exist_ok=True)
    runtime._constitutional_wallets = []
    runtime._constitutional_receipts = []
    runtime._constitutional_proposals = []
    runtime._constitutional_event_log = []
    runtime._constitutional_reflex_cache = {}
    runtime.inference_gateway = None
    runtime._organism = None
    runtime._canonical_mode = False

    # Boot Node0 — mirrors what _lifespan() does in production
    node0 = Node0Heartbeat(data_dir=tmp_path / "api_node0")
    node0.boot()
    runtime._node0 = node0

    from core.sovereign.api import create_fastapi_app

    app = create_fastapi_app(runtime)
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://testserver"), runtime


@pytest.mark.integration
async def test_plan_feeds_node0_canonical_authority(plan_client_with_node0):
    """POST /v1/plan feeds Node0Heartbeat — the canonical ingest authority.

    Proves: API mission → receipt → Node0.ingest_mission_receipt()
    instead of the legacy parallel _submit_mission_to_tick() bridge.

    Standing on Giants:
      Nakamoto (2008) — one chain, one authority
      Deming (1950)   — PDCA: every mission closes through one loop
    """
    client, runtime = plan_client_with_node0
    async with client:
        # After lifespan, runtime._node0 should be a real Node0Heartbeat
        node0 = runtime._node0
        assert node0 is not None, "Node0Heartbeat not booted in API lifespan"
        assert node0._booted is True

        # Record pending receipts before mission
        helix3 = node0._helix3
        before = len(helix3._pending_receipts) if helix3 else 0

        resp = await client.post(
            "/v1/plan",
            json={
                "description": "Test canonical Node0 ingest from API",
                "source": "test",
            },
        )
        assert resp.status_code == 200

        # Verify Node0 received exactly one receipt (not zero, not two)
        if helix3:
            after = len(helix3._pending_receipts)
            added = after - before
            assert added == 1, (
                f"Expected 1 pending receipt in Node0 Helix3, got {added}. "
                f"API is not feeding Node0 canonical authority."
            )

        # Verify legacy tick bridge was NOT called (receipts list unchanged)
        assert runtime._constitutional_receipts == [], (
            "Legacy _submit_mission_to_tick was called alongside Node0 — "
            "should be bypassed when Node0 is active."
        )


@pytest.fixture
def canonical_plan_client(tmp_path, monkeypatch):
    """Client backed by a runtime that exposes canonical organism authority."""
    from httpx import ASGITransport, AsyncClient
    from unittest.mock import AsyncMock, MagicMock

    monkeypatch.setenv("BIZRA_AUTH_ALLOW_ANONYMOUS", "true")

    runtime = MagicMock()
    runtime.config = MagicMock()
    runtime.config.state_dir = tmp_path / "state"
    runtime.config.state_dir.mkdir(parents=True, exist_ok=True)
    runtime._constitutional_wallets = []
    runtime._constitutional_receipts = []
    runtime._constitutional_proposals = []
    runtime._constitutional_event_log = []
    runtime._constitutional_reflex_cache = {}
    runtime._node0 = object()
    runtime._organism = object()
    runtime._canonical_mode = True
    runtime.status.return_value = {
        "canonical": {
            "enabled": True,
            "mission_authority": "organism",
            "authority_path": "runtime->organism->node0",
        }
    }
    runtime.mission = AsyncMock(
        return_value=SimpleNamespace(
            mission_id="org-mission-001",
            output_text="Canonical organism mission complete.",
            system="S2",
            ihsan_score=0.97,
            snr_score=0.93,
            duration_ms=42.5,
            evidence_hash="e" * 64,
            chain_hash="c" * 64,
            fate_verdict="approved",
            fate_reason_codes=[],
            fate_mode="enforced",
            identity_mode="genesis_ed25519",
            signer_public_key_prefix="abcd1234efgh5678",
        )
    )

    from core.sovereign.api import create_fastapi_app

    app = create_fastapi_app(runtime)
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://testserver"), runtime


@pytest.mark.integration
async def test_plan_routes_through_runtime_owned_organism(canonical_plan_client):
    """POST /v1/plan should use runtime.mission when canonical organism exists."""
    client, runtime = canonical_plan_client

    async with client:
        resp = await client.post(
            "/v1/plan",
            json={
                "description": "Run the canonical organism path",
                "source": "terminal",
            },
        )

    assert resp.status_code == 200
    runtime.mission.assert_awaited_once()
    data = resp.json()
    assert data["mission_id"] == "org-mission-001"
    assert data["execution_authority"] == "organism"
    assert data["authority_path"] == "runtime->organism->node0"
    assert data["fate_verdict"] == "approved"
    assert data["identity_mode"] == "genesis_ed25519"
    assert data["signer_public_key_prefix"] == "abcd1234efgh5678"
    assert data["hash_chain_ref"] == "c" * 64


@pytest.fixture
def canonical_plan_client_without_authority(tmp_path, monkeypatch):
    """Client in canonical mode without runtime organism authority."""
    from httpx import ASGITransport, AsyncClient
    from unittest.mock import MagicMock

    monkeypatch.setenv("BIZRA_AUTH_ALLOW_ANONYMOUS", "true")

    runtime = MagicMock()
    runtime.config = MagicMock()
    runtime.config.state_dir = tmp_path / "state"
    runtime.config.state_dir.mkdir(parents=True, exist_ok=True)
    runtime._constitutional_wallets = []
    runtime._constitutional_receipts = []
    runtime._constitutional_proposals = []
    runtime._constitutional_event_log = []
    runtime._constitutional_reflex_cache = {}
    runtime._node0 = None
    runtime._organism = None
    runtime._canonical_mode = True
    runtime.status.return_value = {
        "canonical": {
            "enabled": True,
            "mission_authority": "legacy",
            "authority_path": "",
        }
    }

    from core.sovereign.api import create_fastapi_app

    app = create_fastapi_app(runtime)
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://testserver"), runtime


@pytest.mark.integration
async def test_plan_fails_closed_in_canonical_mode_without_runtime_authority(
    canonical_plan_client_without_authority,
):
    """Canonical mode must not fall back to legacy orchestrator execution."""
    client, _runtime = canonical_plan_client_without_authority

    async with client:
        resp = await client.post(
            "/v1/plan",
            json={"description": "Do not bypass the canonical mission authority"},
        )

    assert resp.status_code == 503
    assert "Canonical mode requires runtime-owned organism mission authority" in resp.json()[
        "error"
    ]


@pytest.mark.integration
async def test_plan_canonical_mode_suppresses_direct_system1_cache_path(
    canonical_plan_client, monkeypatch
):
    """Canonical mode must not let the API return directly from the reflex cache."""
    import core.sovereign.api as api_module
    from core.sovereign.reflex_compiler import ReflexCompiler

    client, runtime = canonical_plan_client
    compiler = ReflexCompiler()
    compiler.compile_from_candidate(
        pattern_id=compiler._hash_input("what is autopoiesis?"),
        input_template="what is autopoiesis?",
        output_template="Autopoiesis is the self-creation property of living systems.",
        ihsan_score=0.97,
        observation_count=5,
    )
    original = getattr(api_module, "_reflex_compiler", None)
    api_module._reflex_compiler = compiler
    try:
        async with client:
            resp = await client.post(
                "/v1/plan",
                json={"description": "what is autopoiesis?", "source": "terminal"},
            )
    finally:
        api_module._reflex_compiler = original

    assert resp.status_code == 200
    runtime.mission.assert_awaited_once()
    data = resp.json()
    assert data["mission_id"] == "org-mission-001"
    assert data["execution_authority"] == "organism"
    assert data["authority_path"] == "runtime->organism->node0"
    assert data["execution_path"] == "SYSTEM_2_NOVEL"


# ═══════════════════════════════════════════════════════════════════
# E2E: CANONICAL AUTHORITY EXCLUSIVITY
# Standing on Giants:
#   Nakamoto (2008) — single authority, no parallel path
#   Lamport (1978) — total ordering, one truth surface
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.integration
async def test_plan_canonical_mode_never_calls_mission_orchestrator(
    canonical_plan_client, monkeypatch
):
    """In canonical mode, MissionOrchestrator.execute() must NEVER be reached.

    This proves exclusivity (one authority), not just preference.
    """
    from unittest.mock import AsyncMock

    client, runtime = canonical_plan_client

    # Patch MissionOrchestrator to trap any direct invocation
    orchestrator_sentinel = AsyncMock(side_effect=AssertionError(
        "MissionOrchestrator.execute() was called in canonical mode — truth fork!"
    ))
    monkeypatch.setattr(
        "core.sovereign.api.MissionOrchestrator",
        type("MockOrchestrator", (), {"execute": orchestrator_sentinel}),
        raising=False,
    )

    async with client:
        resp = await client.post(
            "/v1/plan",
            json={"description": "Prove single authority", "source": "terminal"},
        )

    assert resp.status_code == 200
    runtime.mission.assert_awaited_once()
    orchestrator_sentinel.assert_not_awaited()
    data = resp.json()
    assert data["execution_authority"] == "organism"


@pytest.mark.integration
async def test_plan_canonical_response_includes_fate_fields(canonical_plan_client):
    """Canonical response must carry full FATE lineage fields.

    These fields enable downstream evidence chain, memory, and reflex.
    """
    client, runtime = canonical_plan_client

    async with client:
        resp = await client.post(
            "/v1/plan",
            json={"description": "Check FATE lineage", "source": "terminal"},
        )

    assert resp.status_code == 200
    data = resp.json()

    # All consequence-carrying fields must be present
    assert "fate_verdict" in data
    assert "fate_mode" in data
    assert "identity_mode" in data
    assert "signer_public_key_prefix" in data
    assert "hash_chain_ref" in data
    assert "execution_authority" in data
    assert "authority_path" in data

    # Values must match canonical runtime mock
    assert data["fate_verdict"] == "approved"
    assert data["fate_mode"] == "enforced"
    assert data["identity_mode"] == "genesis_ed25519"
