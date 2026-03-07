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
def plan_client(tmp_path):
    """Create a test client for the FastAPI app with /v1/plan wired."""
    from httpx import ASGITransport, AsyncClient

    os.environ["BIZRA_AUTH_ALLOW_ANONYMOUS"] = "true"
    os.environ["SEMANTIC_MEMORY_PATH"] = str(tmp_path / "memory")
    os.environ["EVENT_LOG_PATH"] = str(tmp_path / "events")
    os.environ["BIZRA_RECEIPT_PRIVATE_KEY_HEX"] = (
        "1111111111111111111111111111111111111111111111111111111111111111"
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
            json={"description": "Summarize the current project status", "source": "test"},
        )
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

        data = resp.json()
        assert "mission_id" in data
        assert "status" in data
        assert data["status"] in ("COMPLETE", "PARTIAL", "FAILED")
        assert "synthesis" in data
        assert "ihsan_score" in data
        assert "snr_score" in data
        assert "evidence_receipt_id" in data
        assert "duration_ms" in data
        assert isinstance(data["channels_executed"], list)


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
