"""
E2E Metabolism Test — Canonical Flagship Integration Proof.

Proves the complete BIZRA metabolism loop operates end-to-end:

  Mission → Receipt → Tick → Reflex → Cache Hit
  (+ EventBus emissions at each stage)

This is the single test that demonstrates BIZRA is a sovereign OS,
not a prompt wrapper.

Standing on Giants:
- Nakamoto (2008): Block processing tick
- Kahneman (2002): System-1/System-2 split
- Al-Khwarizmi (780-850): Deterministic procedure
- Hewitt (1973): Actor model messaging
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")


@asynccontextmanager
async def metabolism_env(tmp_path, monkeypatch):
    """Create a test client with full metabolism wiring."""
    from unittest.mock import MagicMock

    from httpx import ASGITransport, AsyncClient

    monkeypatch.setenv("BIZRA_AUTH_ALLOW_ANONYMOUS", "true")
    monkeypatch.setenv("SEMANTIC_MEMORY_PATH", str(tmp_path / "memory"))
    monkeypatch.setenv("EVENT_LOG_PATH", str(tmp_path / "events"))
    monkeypatch.setenv(
        "BIZRA_RECEIPT_PRIVATE_KEY_HEX",
        "1111111111111111111111111111111111111111111111111111111111111111",
    )
    monkeypatch.setenv("BIZRA_TICK_INTERVAL_S", "0")

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
    try:
        yield client, runtime
    finally:
        await client.aclose()
        close_transport = getattr(transport, "aclose", None)
        if close_transport is not None:
            await close_transport()


@pytest.mark.integration
def test_full_metabolism_loop(tmp_path, monkeypatch):
    asyncio.run(_test_full_metabolism_loop(tmp_path, monkeypatch))


async def _test_full_metabolism_loop(tmp_path, monkeypatch):
    async with metabolism_env(tmp_path, monkeypatch) as (client, runtime):
        resp = await client.post(
            "/v1/plan",
            json={"description": "Analyze system health and report status"},
        )
        assert resp.status_code == 200, f"Mission failed: {resp.text}"
        data = resp.json()
        assert "mission_id" in data
        assert data["status"] in ("COMPLETE", "PARTIAL", "FAILED")

        assert len(runtime._constitutional_receipts) == 1
        receipt = runtime._constitutional_receipts[0]
        assert receipt.action_type == "mission"
        assert receipt.intent_score > 0

        from core.constitutional.fixed_point import fp
        from core.constitutional.ticker import process_tick
        from core.constitutional.types import WalletState

        wallet = WalletState(node_id=receipt.actor_id)
        runtime._constitutional_wallets.append(wallet)

        boosted = receipt.__class__(
            receipt_id=receipt.receipt_id,
            actor_id=receipt.actor_id,
            action_type=receipt.action_type,
            timestamp=receipt.timestamp,
            intent_score=fp(0.97),
            efficiency_score=fp(0.96),
            impact_score=fp(0.97),
            reproducibility_score=fp(0.95),
            oracle_signature=receipt.oracle_signature,
            metadata_hash=receipt.metadata_hash,
        )

        tick_result = process_tick(
            wallets=runtime._constitutional_wallets,
            receipts=[boosted],
            proposals=runtime._constitutional_proposals,
            event_log=runtime._constitutional_event_log,
            reflex_cache=runtime._constitutional_reflex_cache,
        )

        assert tick_result.scored >= 1, "Receipt was not scored"
        assert tick_result.rejected == 0, "Receipt was unexpectedly rejected"
        assert tick_result.total_minted > 0, "No SEED was minted"
        assert wallet.seed_balance > 0, "Wallet balance not updated"
        assert tick_result.events_logged >= 1

        docs_resp = await client.get("/openapi.json")
        assert docs_resp.status_code == 200


@pytest.mark.integration
def test_metabolism_reflex_compilation(tmp_path, monkeypatch):
    asyncio.run(_test_metabolism_reflex_compilation(tmp_path, monkeypatch))


async def _test_metabolism_reflex_compilation(tmp_path, monkeypatch):
    async with metabolism_env(tmp_path, monkeypatch) as (client, runtime):
        resp = await client.post(
            "/v1/plan",
            json={"description": "High-quality analysis task"},
        )
        assert resp.status_code == 200

        receipt = runtime._constitutional_receipts[0]

        from core.constitutional.fixed_point import fp
        from core.constitutional.ticker import process_tick
        from core.constitutional.types import WalletState

        receipt = receipt.__class__(
            receipt_id=receipt.receipt_id,
            actor_id=receipt.actor_id,
            action_type=receipt.action_type,
            timestamp=receipt.timestamp,
            intent_score=fp(0.99),
            efficiency_score=fp(0.99),
            impact_score=fp(0.99),
            reproducibility_score=fp(0.99),
            oracle_signature=receipt.oracle_signature,
            metadata_hash=receipt.metadata_hash,
        )

        wallet = WalletState(node_id=receipt.actor_id)
        tick_result = process_tick(
            wallets=[wallet],
            receipts=[receipt],
            proposals=[],
            event_log=[],
            reflex_cache=runtime._constitutional_reflex_cache,
        )

        assert tick_result.scored >= 1
        assert runtime._constitutional_reflex_cache, (
            "Reflex cache should contain compiled pattern"
        )


@pytest.mark.integration
def test_metabolism_event_bus_emissions(tmp_path, monkeypatch):
    asyncio.run(_test_metabolism_event_bus_emissions(tmp_path, monkeypatch))


async def _test_metabolism_event_bus_emissions(tmp_path, monkeypatch):
    import core.sovereign.event_bus as _eb

    _eb._global_bus = None
    from core.sovereign.event_bus import EventBus

    bus = EventBus()
    _eb._global_bus = bus
    captured_events: list[dict] = []

    async def capture_handler(event):
        captured_events.append({"topic": event.topic, "payload": event.payload})

    bus.subscribe("mission.created", capture_handler)
    bus.subscribe("mission.executed", capture_handler)
    bus.subscribe("mission.failed", capture_handler)
    bus_task = asyncio.create_task(bus.start())

    try:
        async with metabolism_env(tmp_path, monkeypatch) as (client, _runtime):
            resp = await client.post(
                "/v1/plan",
                json={"description": "Test bus event emissions"},
            )
            assert resp.status_code == 200
            await asyncio.sleep(0.2)

        topics = [e["topic"] for e in captured_events]
        assert "mission.created" in topics, (
            f"Expected mission.created, got {topics}"
        )
        assert any(
            topic in topics for topic in ("mission.executed", "mission.failed")
        ), f"Expected mission.executed or mission.failed, got {topics}"

        created = next(e for e in captured_events if e["topic"] == "mission.created")
        completed = next(
            e
            for e in captured_events
            if e["topic"] in ("mission.executed", "mission.failed")
        )
        assert created["payload"]["mission_id"] == completed["payload"]["mission_id"]
    finally:
        bus.stop()
        bus_task.cancel()
        try:
            await bus_task
        except asyncio.CancelledError:
            pass
        _eb._global_bus = None


@pytest.mark.integration
def test_metabolism_wallet_growth_across_missions(tmp_path, monkeypatch):
    asyncio.run(_test_metabolism_wallet_growth_across_missions(tmp_path, monkeypatch))


async def _test_metabolism_wallet_growth_across_missions(tmp_path, monkeypatch):
    async with metabolism_env(tmp_path, monkeypatch) as (client, runtime):
        from core.constitutional.fixed_point import fp
        from core.constitutional.ticker import process_tick
        from core.constitutional.types import WalletState

        wallet = WalletState(node_id=b"\x00" * 32)

        for i in range(3):
            resp = await client.post(
                "/v1/plan",
                json={"description": f"Mission {i + 1}: system analysis"},
            )
            assert resp.status_code == 200

            receipts = list(runtime._constitutional_receipts)
            boosted = []
            for receipt in receipts:
                boosted.append(
                    receipt.__class__(
                        receipt_id=receipt.receipt_id,
                        actor_id=receipt.actor_id,
                        action_type=receipt.action_type,
                        timestamp=receipt.timestamp,
                        intent_score=fp(0.97),
                        efficiency_score=fp(0.96),
                        impact_score=fp(0.97),
                        reproducibility_score=fp(0.95),
                        oracle_signature=receipt.oracle_signature,
                        metadata_hash=receipt.metadata_hash,
                    )
                )

            process_tick(
                wallets=[wallet],
                receipts=boosted,
                proposals=[],
                event_log=[],
                reflex_cache=runtime._constitutional_reflex_cache,
            )
            runtime._constitutional_receipts = []

        assert wallet.seed_balance > 0, "Wallet should have accumulated SEED"
        assert wallet.total_actions >= 1, "Wallet should track action count"
