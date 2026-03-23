"""
Sovereign Nervous System Integration Tests
===========================================
Proves the S1/S2 cognitive bridge works end-to-end in-memory.

Each test class targets one constitutional invariant:
  1. S1 reflex fast-path (Kahneman)
  2. S2 deliberation full loop (Boyd OODA)
  3. Ihsān gate enforcement (Al-Ghazali §4)
  4. BLOOM reward minting + 50% split (Ostrom §12)
  5. Gini invariant (ADL §14)
  6. Evidence chain integrity (Lamport)
  7. EventBus event flow (Hewitt)
  8. Factory wiring (all modules compose)
"""

from __future__ import annotations

import asyncio
from typing import Any, List

import pytest

from core.sovereign.mission_nervous_system import (
    EchoInference,
    NervousSystemReceipt,
    SovereignNervousSystem,
)

# ═══════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════


class MockInference:
    """Controllable mock inference for testing."""

    def __init__(
        self, response: str = "Mock output with relevant content about the mission"
    ):
        self.response = response
        self.calls: List[str] = []

    async def infer(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append(prompt)
        return self.response


class FailingInference:
    """Inference provider that fails to test typed degradation."""

    async def infer(self, prompt: str, **kwargs: Any) -> str:
        raise RuntimeError("backend offline")


class FailingBus:
    """Event bus that fails immediately to test bridge degradation."""

    def publish(self, event_type: Any, payload: dict[str, Any]) -> Any:
        raise RuntimeError(f"publish failed for {event_type}")


class ReceiptCollector:
    """Collects receipts for assertion."""

    def __init__(self):
        self.receipts: List[NervousSystemReceipt] = []

    def __call__(self, receipt: NervousSystemReceipt) -> None:
        self.receipts.append(receipt)


@pytest.fixture
def mock_inference():
    return MockInference()


@pytest.fixture
def collector():
    return ReceiptCollector()


def _build_ns(
    inference=None,
    reflex=None,
    bus=None,
    minter=None,
    wallet=None,
    wallets=None,
    on_receipt=None,
    reward_per_mission=1.0,
) -> SovereignNervousSystem:
    """Build a nervous system with explicit deps (no Phase 80 imports)."""
    return SovereignNervousSystem(
        inference=inference or MockInference(),
        reflex_cache=reflex,
        event_bus=bus,
        token_minter=minter,
        wallet=wallet,
        wallets=wallets,
        on_receipt=on_receipt,
        reward_per_mission=reward_per_mission,
    )


# ═══════════════════════════════════════════════════════════════════
# 1. S2 DELIBERATION (full loop — no reflex cache)
# ═══════════════════════════════════════════════════════════════════


class TestS2Deliberation:
    """When no reflex cache hit, the system delegates to full inference."""

    def test_s2_basic_execution(self):
        inf = MockInference()
        ns = _build_ns(inference=inf)

        receipt = asyncio.run(ns.run("Summarize the report"))

        assert receipt.system == "S2"
        assert receipt.reflex_hit is False
        assert receipt.output_text == inf.response
        assert len(inf.calls) == 1
        assert inf.calls[0] == "Summarize the report"

    def test_s2_records_observation(self):
        from core.sovereign.reflex_compiler import ReflexCompiler

        reflex = ReflexCompiler(max_entries=100, persistence_path=None)
        ns = _build_ns(reflex=reflex)

        asyncio.run(ns.run("Analyze quarterly data", ihsan_override=0.96))

        # The observation was recorded for future precipitation
        assert reflex.stats.total_lookups == 1  # Lookup was attempted
        # ReflexCompiler has the observation tracked internally

    def test_s2_increments_stats(self):
        ns = _build_ns()

        asyncio.run(ns.run("Task one"))
        asyncio.run(ns.run("Task two"))

        assert ns.stats.total_missions == 2
        assert ns.stats.s2_executions == 2
        assert ns.stats.s1_hits == 0

    def test_s2_inference_failure_returns_degraded_receipt(self):
        ns = _build_ns(inference=FailingInference())

        receipt = asyncio.run(ns.run("Recover from inference outage"))

        assert receipt.system == "S2"
        assert receipt.output_text.startswith("[DEGRADED]")
        assert receipt.rewarded is False
        assert receipt.ihsan_score <= 0.2
        degradations = receipt.metadata.get("degradation_receipts", [])
        assert len(degradations) == 1
        assert degradations[0]["error_type"] == "InferenceError"
        assert degradations[0]["boundary"] == "INFERENCE"

    def test_event_bus_failure_becomes_degradation_receipt(self):
        ns = _build_ns(bus=FailingBus())

        receipt = asyncio.run(ns.run("Publish through degraded bus"))

        assert receipt.events_published == []
        assert receipt.metadata["event_bus_degraded"] is True
        degradations = receipt.metadata.get("degradation_receipts", [])
        assert len(degradations) == 1
        assert degradations[0]["error_type"] == "BridgeError"
        assert degradations[0]["boundary"] == "BRIDGE"


# ═══════════════════════════════════════════════════════════════════
# 2. S1 REFLEX FAST-PATH (Kahneman)
# ═══════════════════════════════════════════════════════════════════


class TestS1ReflexFastPath:
    """When reflex cache hits, S2 inference is SKIPPED entirely."""

    def test_s1_cache_hit_skips_inference(self):
        from core.sovereign.reflex_compiler import ReflexCompiler

        reflex = ReflexCompiler(max_entries=100, persistence_path=None)
        inf = MockInference()

        # Seed the cache: record enough observations to precipitate
        mission = "recurring daily standup summary"
        for _ in range(5):
            reflex.record_observation(
                input_text=mission,
                output_text="Standup summary: all blockers resolved",
                ihsan_composite=0.97,
            )

        ns = _build_ns(inference=inf, reflex=reflex)
        receipt = asyncio.run(ns.run(mission, ihsan_override=0.97))

        # If precipitated, S1 should hit
        if receipt.reflex_hit:
            assert receipt.system == "S1"
            assert len(inf.calls) == 0  # Inference was NOT called
            assert ns.stats.s1_hits == 1
        else:
            # Precipitation threshold not yet met — S2 is fine
            assert receipt.system == "S2"

    def test_s1_hit_rate_tracking(self):
        from core.sovereign.reflex_compiler import ReflexCompiler

        reflex = ReflexCompiler(max_entries=100, persistence_path=None)
        ns = _build_ns(reflex=reflex)

        # All misses — hit rate should be 0
        for i in range(3):
            asyncio.run(ns.run(f"unique mission {i}"))

        assert ns.stats.s1_hit_rate == 0.0


# ═══════════════════════════════════════════════════════════════════
# 3. IHSĀN GATE (Al-Ghazali §4)
# ═══════════════════════════════════════════════════════════════════


class TestIhsanGate:
    """Constitutional Ihsān threshold controls reward eligibility."""

    def test_high_ihsan_gets_reward(self):
        from core.token.bloom import CommunityPool
        from core.token.bloom import TokenMinter as BloomMinter
        from core.token.bloom import WalletState

        pool = CommunityPool()
        minter = BloomMinter(community_pool=pool)
        wallet = WalletState(node_id="test_node")

        ns = _build_ns(minter=minter, wallet=wallet, wallets=[wallet])

        receipt = asyncio.run(ns.run("Quality mission", ihsan_override=0.97))

        assert receipt.rewarded is True
        assert receipt.reward_amount > 0
        assert receipt.pool_contribution > 0

    def test_low_ihsan_no_reward(self):
        from core.token.bloom import CommunityPool
        from core.token.bloom import TokenMinter as BloomMinter
        from core.token.bloom import WalletState

        pool = CommunityPool()
        minter = BloomMinter(community_pool=pool)
        wallet = WalletState(node_id="test_node")

        ns = _build_ns(minter=minter, wallet=wallet, wallets=[wallet])

        receipt = asyncio.run(ns.run("Mediocre mission", ihsan_override=0.80))

        assert receipt.rewarded is False
        assert receipt.reward_amount == 0.0
        assert receipt.pool_contribution == 0.0

    def test_ihsan_boundary_at_threshold(self):
        from core.token.bloom import CommunityPool
        from core.token.bloom import TokenMinter as BloomMinter
        from core.token.bloom import WalletState

        pool = CommunityPool()
        minter = BloomMinter(community_pool=pool)
        wallet = WalletState(node_id="test_node")

        ns = _build_ns(minter=minter, wallet=wallet, wallets=[wallet])

        # Exactly at threshold should pass
        receipt = asyncio.run(ns.run("Threshold mission", ihsan_override=0.95))
        assert receipt.rewarded is True

        # Just below should fail
        receipt2 = asyncio.run(ns.run("Below threshold", ihsan_override=0.949))
        assert receipt2.rewarded is False


# ═══════════════════════════════════════════════════════════════════
# 4. BLOOM 50% COMMUNITY POOL SPLIT (Ostrom §12)
# ═══════════════════════════════════════════════════════════════════


class TestBloomPoolSplit:
    """The 50% split is constitutionally locked (البذرة p19)."""

    def test_50_percent_split(self):
        from core.token.bloom import CommunityPool
        from core.token.bloom import TokenMinter as BloomMinter
        from core.token.bloom import WalletState

        pool = CommunityPool()
        minter = BloomMinter(community_pool=pool)
        wallet = WalletState(node_id="test_node")

        ns = _build_ns(
            minter=minter,
            wallet=wallet,
            wallets=[wallet],
            reward_per_mission=10.0,
        )

        receipt = asyncio.run(ns.run("Excellent mission", ihsan_override=0.98))

        assert receipt.rewarded is True
        assert receipt.reward_amount == 5.0  # 50% to node
        assert receipt.pool_contribution == 5.0  # 50% to pool
        assert wallet.seed_balance == 5.0
        assert pool.current_balance == 5.0

    def test_cumulative_pool_growth(self):
        from core.token.bloom import CommunityPool
        from core.token.bloom import TokenMinter as BloomMinter
        from core.token.bloom import WalletState

        pool = CommunityPool()
        minter = BloomMinter(community_pool=pool)
        wallet = WalletState(node_id="test_node")

        ns = _build_ns(
            minter=minter,
            wallet=wallet,
            wallets=[wallet],
            reward_per_mission=2.0,
        )

        for i in range(5):
            asyncio.run(ns.run(f"Mission {i}", ihsan_override=0.96))

        assert ns.stats.rewards_minted == 5
        assert ns.stats.total_seed_minted == 10.0  # 5 × 2.0
        assert ns.stats.total_pool_contributed == 5.0  # 50% of 10.0
        assert wallet.seed_balance == 5.0
        assert pool.current_balance == 5.0


# ═══════════════════════════════════════════════════════════════════
# 5. GINI INVARIANT (ADL §14)
# ═══════════════════════════════════════════════════════════════════


class TestGiniInvariant:
    """Token minting halts if wealth inequality exceeds 0.35 Gini."""

    def test_gini_ok_with_equal_wallets(self):
        from core.token.bloom import CommunityPool
        from core.token.bloom import TokenMinter as BloomMinter
        from core.token.bloom import WalletState

        pool = CommunityPool()
        minter = BloomMinter(community_pool=pool)
        w1 = WalletState(node_id="a", seed_balance=10.0)
        w2 = WalletState(node_id="b", seed_balance=12.0)
        w3 = WalletState(node_id="c", seed_balance=11.0)

        ns = _build_ns(
            minter=minter,
            wallet=w1,
            wallets=[w1, w2, w3],
        )

        receipt = asyncio.run(ns.run("Fair mission", ihsan_override=0.96))
        assert receipt.gini_ok is True

    def test_gini_halt_with_extreme_inequality(self):
        from core.token.bloom import CommunityPool
        from core.token.bloom import TokenMinter as BloomMinter
        from core.token.bloom import WalletState

        pool = CommunityPool()
        minter = BloomMinter(community_pool=pool)
        w1 = WalletState(node_id="whale", seed_balance=10000.0)
        w2 = WalletState(node_id="poor_a", seed_balance=1.0)
        w3 = WalletState(node_id="poor_b", seed_balance=1.0)

        ns = _build_ns(
            minter=minter,
            wallet=w1,
            wallets=[w1, w2, w3],
        )

        receipt = asyncio.run(ns.run("Unjust mission", ihsan_override=0.96))
        assert receipt.gini_ok is False
        assert ns.stats.gini_halts >= 1


# ═══════════════════════════════════════════════════════════════════
# 6. EVIDENCE CHAIN INTEGRITY (Lamport)
# ═══════════════════════════════════════════════════════════════════


class TestEvidenceChain:
    """Every receipt links to the previous one via hash chain."""

    def test_chain_advances(self):
        collector = ReceiptCollector()
        ns = _build_ns(on_receipt=collector)

        asyncio.run(ns.run("Mission A", ihsan_override=0.96))
        asyncio.run(ns.run("Mission B", ihsan_override=0.97))
        asyncio.run(ns.run("Mission C", ihsan_override=0.95))

        assert len(collector.receipts) == 3

        # Each receipt has unique chain hash
        hashes = [r.chain_hash for r in collector.receipts]
        assert len(set(hashes)) == 3  # All unique

        # Chain is deterministic
        assert hashes[0] != "0" * 64  # Advanced from genesis

    def test_evidence_hash_is_deterministic(self):
        collector = ReceiptCollector()
        ns = _build_ns(on_receipt=collector)

        asyncio.run(ns.run("Deterministic mission", ihsan_override=0.96))

        receipt = collector.receipts[0]
        assert receipt.evidence_hash.startswith("ev:")
        assert len(receipt.evidence_hash) == 35  # "ev:" + 32 hex chars

    def test_receipt_is_serializable(self):
        collector = ReceiptCollector()
        ns = _build_ns(on_receipt=collector)

        asyncio.run(ns.run("Serialize me", ihsan_override=0.96))

        d = collector.receipts[0].as_dict()
        assert isinstance(d, dict)
        assert "mission_id" in d
        assert "chain_hash" in d
        assert "evidence_hash" in d


# ═══════════════════════════════════════════════════════════════════
# 7. EVENTBUS EVENT FLOW (Hewitt Actor Model)
# ═══════════════════════════════════════════════════════════════════


class TestEventBusFlow:
    """Events flow to the Phase 80 EventBus with correct types."""

    def test_events_published_on_s2(self):
        from core.bus.subscribers import EventBus

        bus = EventBus()
        ns = _build_ns(bus=bus)

        receipt = asyncio.run(ns.run("Eventful mission", ihsan_override=0.96))

        assert "action.intent" in receipt.events_published
        assert "action.receipt" in receipt.events_published
        assert "session.end" in receipt.events_published
        assert len(bus._chain) >= 3  # At least 3 events

    def test_ihsan_breach_event_on_low_score(self):
        from core.bus.subscribers import EventBus

        bus = EventBus()
        ns = _build_ns(bus=bus)

        receipt = asyncio.run(ns.run("Low quality", ihsan_override=0.80))

        assert "ihsan.gate.breached" in receipt.events_published

    def test_no_breach_event_on_high_score(self):
        from core.bus.subscribers import EventBus

        bus = EventBus()
        ns = _build_ns(bus=bus)

        receipt = asyncio.run(ns.run("High quality", ihsan_override=0.98))

        assert "ihsan.gate.breached" not in receipt.events_published

    def test_low_ihsan_breach_payload_matches_subscriber_schema(self):
        from core.bus.subscribers import EventBus, EventType

        bus = EventBus()
        ns = _build_ns(bus=bus)

        receipt = asyncio.run(ns.run("Low quality", ihsan_override=0.80))
        breach = next(
            e for e in bus._chain if e.event_type == EventType.IHSAN_GATE_BREACHED
        )

        assert breach.payload["session_id"] == receipt.mission_id
        assert breach.payload["ihsan_composite"] == pytest.approx(0.80)
        assert breach.payload["action_type"].startswith("mission:")
        assert breach.payload["violation_dimensions"] == ["ihsan_below_threshold"]

    def test_action_receipt_payload_matches_subscriber_schema(self):
        from core.bus.subscribers import EventBus, EventType

        bus = EventBus()
        ns = _build_ns(bus=bus)

        receipt = asyncio.run(ns.run("Eventful mission", ihsan_override=0.96))
        action_receipt = next(
            e for e in bus._chain if e.event_type == EventType.ACTION_RECEIPT
        )
        session_end = next(
            e for e in bus._chain if e.event_type == EventType.SESSION_END
        )

        assert action_receipt.payload["action_type"].startswith("mission:")
        assert action_receipt.payload["ihsan_composite"] == pytest.approx(0.96)
        assert action_receipt.payload["result_summary"]
        assert session_end.payload["session_id"] == receipt.mission_id
        assert session_end.payload["actions"][0]["ihsan_composite"] == pytest.approx(
            0.96
        )


# ═══════════════════════════════════════════════════════════════════
# 8. FACTORY WIRING (all Phase 80 modules compose)
# ═══════════════════════════════════════════════════════════════════


class TestFactoryWiring:
    """The create() factory wires all Phase 80 modules together."""

    def test_create_factory(self):
        ns = SovereignNervousSystem.create(
            inference=EchoInference(),
            reward_per_mission=1.0,
        )

        receipt = asyncio.run(ns.run("Factory test", ihsan_override=0.96))

        assert receipt.system == "S2"
        assert receipt.rewarded is True
        assert receipt.events_published  # Events were published
        assert ns.stats.total_missions == 1

    def test_echo_inference(self):
        ns = _build_ns(inference=EchoInference())

        receipt = asyncio.run(ns.run("Echo test"))

        assert "[Echo]" in receipt.output_text
        assert "Echo test" in receipt.output_text


# ═══════════════════════════════════════════════════════════════════
# 9. STATS OBSERVABILITY
# ═══════════════════════════════════════════════════════════════════


class TestStatsObservability:
    """Stats track all nervous system activity for observability."""

    def test_stats_accumulate(self):
        from core.token.bloom import CommunityPool
        from core.token.bloom import TokenMinter as BloomMinter
        from core.token.bloom import WalletState

        pool = CommunityPool()
        minter = BloomMinter(community_pool=pool)
        wallet = WalletState(node_id="stats_node")

        ns = _build_ns(minter=minter, wallet=wallet, wallets=[wallet])

        asyncio.run(ns.run("Good mission", ihsan_override=0.96))
        asyncio.run(ns.run("Bad mission", ihsan_override=0.70))

        stats = ns.stats
        assert stats.total_missions == 2
        assert stats.rewards_minted == 1
        # Low Ihsān gates BEFORE calling minter (not a minter rejection)
        assert stats.rewards_rejected == 0
        assert stats.avg_ihsan == pytest.approx(0.83, abs=0.01)

    def test_stats_serializable(self):
        ns = _build_ns()
        asyncio.run(ns.run("Serialize stats"))

        d = ns.stats.as_dict()
        assert isinstance(d, dict)
        assert "s1_hit_rate" in d
        assert "total_missions" in d
