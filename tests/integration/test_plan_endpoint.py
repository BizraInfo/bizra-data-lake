"""
Integration test for POST /v1/plan — the sovereign mission golden path.

Sprint 1, Task 1.5: Full mission loop integration test.
Blueprint acceptance criterion: `curl /v1/plan` returns receipted result.

Standing on Giants:
- Boyd: OODA loop (observe → orient → decide → act)
- Lamport: Hash-chained evidence with ordering invariant
"""

from __future__ import annotations

import os

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
        assert "evidence_receipt_id" in data or "receipt_id" in data
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
            assert data["execution_path"] == "system_1_reflex"
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
            assert data["execution_path"] in ("system_2", "SYSTEM_2")

            # Compiler should have recorded the observation
            pattern_hash = compiler._hash_input(
                "a unique test query for precipitation"
            )
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
