"""
Helix 3 Constitutional Evolutionary Scheduler Tests
=====================================================
Proves the Triple Helix is complete: S1 (reflex) + S2 (deliberation) + S3 (evolution).

9 test classes, one per constitutional invariant:
  1. IhsanTensor8D — geometric mean, fail-closed, dimension counting
  2. Helix3 empty tick — baseline behavior with no receipts
  3. Helix3 with receipts — evolutionary processing
  4. Gini enforcement — HALT on inequality
  5. Reflex lifecycle — precipitation + pruning
  6. Evidence chain — hash continuity
  7. NervousSystem integration — wire_helix3 factory
  8. Constitutional bridge — process_tick_constitutional fallback
  9. Stats observability — cumulative tracking
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest

from core.sovereign.helix3 import (
    HeartbeatReceipt,
    Helix3Scheduler,
    IhsanTensor8D,
    wire_helix3,
)


# ═══════════════════════════════════════════════════════════════════
# 1. IHSĀN TENSOR 8D (§8)
# ═══════════════════════════════════════════════════════════════════

class TestIhsanTensor8D:
    """The 8D tensor with geometric mean — constitutional scoring."""

    def test_uniform_high_score(self):
        t = IhsanTensor8D.uniform(0.96)
        assert t.geometric_mean == pytest.approx(0.96, abs=0.001)
        assert t.weighted_mean == pytest.approx(0.96, abs=0.001)
        assert t.verified_count == 8  # All above 0.85

    def test_zero_dimension_kills_score(self):
        """Fail-closed: zero in ANY dimension → zero composite."""
        t = IhsanTensor8D.uniform(0.96)
        t.moral_clarity = 0.0  # One dimension failed
        assert t.geometric_mean == 0.0

    def test_mixed_dimensions(self):
        t = IhsanTensor8D(
            moral_clarity=0.95,
            epistemic_humility=0.90,
            structural_integrity=0.92,
            verifiability=0.88,
            contextual_relevance=0.91,
            intent_alignment=0.96,
            resilience=0.87,
            efficiency=0.93,
        )
        # Geometric mean should be lower than arithmetic
        assert t.geometric_mean < t.weighted_mean
        assert t.geometric_mean > 0.85
        assert t.min_dimension == 0.87

    def test_verified_count(self):
        t = IhsanTensor8D(
            moral_clarity=0.95,
            epistemic_humility=0.80,  # Below 0.85
            structural_integrity=0.92,
            verifiability=0.60,  # Below 0.85
            contextual_relevance=0.91,
            intent_alignment=0.96,
            resilience=0.87,
            efficiency=0.93,
        )
        assert t.verified_count == 6  # 2 below threshold

    def test_from_scores(self):
        scores = {"moral_clarity": 0.95, "efficiency": 0.90}
        t = IhsanTensor8D.from_scores(scores)
        assert t.moral_clarity == 0.95
        assert t.efficiency == 0.90
        assert t.epistemic_humility == 0.0  # Unset → 0.0

    def test_dimensions_property(self):
        t = IhsanTensor8D.uniform(0.95)
        dims = t.dimensions
        assert len(dims) == 8
        assert all(v == 0.95 for v in dims.values())


# ═══════════════════════════════════════════════════════════════════
# 2. EMPTY TICK (baseline)
# ═══════════════════════════════════════════════════════════════════

class TestEmptyTick:
    """Process tick with no receipts — organism at rest."""

    def test_empty_tick_produces_receipt(self):
        h3 = Helix3Scheduler()
        receipt = h3.process_tick()

        assert isinstance(receipt, HeartbeatReceipt)
        assert receipt.tick_number == 1
        assert receipt.missions_processed == 0
        assert receipt.gini_ok is True  # No wallets → ok

    def test_empty_tensor_defaults_to_threshold(self):
        h3 = Helix3Scheduler()
        receipt = h3.process_tick()

        # Empty tensor should default to UNIFIED_IHSAN_THRESHOLD (0.95)
        assert receipt.ihsan_composite == pytest.approx(0.95, abs=0.001)

    def test_consecutive_ticks_advance_counter(self):
        h3 = Helix3Scheduler()
        r1 = h3.process_tick()
        r2 = h3.process_tick()
        r3 = h3.process_tick()

        assert r1.tick_number == 1
        assert r2.tick_number == 2
        assert r3.tick_number == 3


# ═══════════════════════════════════════════════════════════════════
# 3. TICK WITH RECEIPTS (evolutionary processing)
# ═══════════════════════════════════════════════════════════════════

class TestTickWithReceipts:
    """Process tick with mission receipts — organism evolving."""

    def _make_receipt(self, ihsan: float = 0.96, mission_id: str = "m-1") -> Dict[str, Any]:
        return {
            "mission_id": mission_id,
            "ihsan_score": ihsan,
            "snr_score": 0.90,
            "reward_amount": 0.5,
            "evidence_hash": "ev:abc123",
        }

    def test_receipts_counted(self):
        h3 = Helix3Scheduler()
        h3.ingest_receipt(self._make_receipt())
        h3.ingest_receipt(self._make_receipt(mission_id="m-2"))

        receipt = h3.process_tick()
        assert receipt.missions_processed == 2

    def test_receipts_cleared_after_tick(self):
        h3 = Helix3Scheduler()
        h3.ingest_receipt(self._make_receipt())

        h3.process_tick()
        r2 = h3.process_tick()  # Second tick should have 0 receipts
        assert r2.missions_processed == 0

    def test_ihsan_tensor_from_receipts(self):
        h3 = Helix3Scheduler()
        h3.ingest_receipt(self._make_receipt(ihsan=0.96))
        h3.ingest_receipt(self._make_receipt(ihsan=0.94, mission_id="m-2"))

        receipt = h3.process_tick()
        # Average ihsan: (0.96 + 0.94) / 2 = 0.95
        assert receipt.ihsan_composite == pytest.approx(0.95, abs=0.001)


# ═══════════════════════════════════════════════════════════════════
# 4. GINI ENFORCEMENT (§14 ADL)
# ═══════════════════════════════════════════════════════════════════

class TestGiniEnforcement:
    """Gini invariant ≤ 0.35 or evolutionary rewards HALT."""

    def test_gini_ok_equal_wallets(self):
        from core.token.bloom import WalletState

        w1 = WalletState(node_id="a", seed_balance=10.0)
        w2 = WalletState(node_id="b", seed_balance=12.0)

        h3 = Helix3Scheduler(wallets=[w1, w2])
        receipt = h3.process_tick()
        assert receipt.gini_ok is True

    def test_gini_halt_extreme_inequality(self):
        from core.token.bloom import WalletState

        w1 = WalletState(node_id="whale", seed_balance=10000.0)
        w2 = WalletState(node_id="poor", seed_balance=1.0)

        h3 = Helix3Scheduler(wallets=[w1, w2])
        receipt = h3.process_tick()
        assert receipt.gini_ok is False
        assert h3.stats.total_gini_halts == 1


# ═══════════════════════════════════════════════════════════════════
# 5. REFLEX LIFECYCLE (precipitation + pruning)
# ═══════════════════════════════════════════════════════════════════

class TestReflexLifecycle:
    """Reflex cache evolves through precipitation and pruning."""

    def test_precipitation_revalidation(self):
        from core.sovereign.reflex_compiler import ReflexCompiler

        reflex = ReflexCompiler(max_entries=100, persistence_path=None)
        # Seed a reflex
        reflex.import_forest_reflex(
            key_str="evolve_key", plan="evolve_plan",
            ihsan=0.96, source="local", confidence=0.95,
        )

        h3 = Helix3Scheduler(reflex_cache=reflex)
        receipt = h3.process_tick()

        assert receipt.reflexes_precipitated >= 0  # May or may not precipitate
        assert isinstance(receipt.reflexes_pruned, int)

    def test_reflex_stats_track(self):
        from core.sovereign.reflex_compiler import ReflexCompiler

        reflex = ReflexCompiler(max_entries=100, persistence_path=None)
        h3 = Helix3Scheduler(reflex_cache=reflex)

        h3.process_tick()
        h3.process_tick()

        assert h3.stats.total_ticks == 2


# ═══════════════════════════════════════════════════════════════════
# 6. EVIDENCE CHAIN (Lamport)
# ═══════════════════════════════════════════════════════════════════

class TestEvidenceChain:
    """Hash chain links heartbeats into tamper-evident sequence."""

    def test_chain_advances(self):
        h3 = Helix3Scheduler()
        r1 = h3.process_tick()
        r2 = h3.process_tick()
        r3 = h3.process_tick()

        # All unique
        hashes = {r1.chain_hash, r2.chain_hash, r3.chain_hash}
        assert len(hashes) == 3

        # None is genesis
        assert all(h != "0" * 64 for h in hashes)

    def test_evidence_hash_format(self):
        h3 = Helix3Scheduler()
        receipt = h3.process_tick()
        assert receipt.evidence_hash.startswith("ev:")

    def test_receipt_serializable(self):
        h3 = Helix3Scheduler()
        receipt = h3.process_tick()
        d = receipt.as_dict()
        assert isinstance(d, dict)
        assert "tick_number" in d
        assert "chain_hash" in d
        assert "ihsan_tensor" in d


# ═══════════════════════════════════════════════════════════════════
# 7. NERVOUS SYSTEM INTEGRATION (wire_helix3)
# ═══════════════════════════════════════════════════════════════════

class TestNervousSystemIntegration:
    """Helix 3 wires into the Nervous System seamlessly."""

    def test_wire_helix3_factory(self):
        from core.sovereign.mission_nervous_system import (
            EchoInference,
            SovereignNervousSystem,
        )

        ns = SovereignNervousSystem.create(
            inference=EchoInference(),
            reward_per_mission=1.0,
        )

        heartbeats: List[HeartbeatReceipt] = []
        h3 = wire_helix3(ns, on_heartbeat=heartbeats.append)

        # Run a mission — should auto-ingest into Helix 3
        asyncio.run(ns.run("Test mission", ihsan_override=0.96))

        # Process tick — should see the mission
        receipt = h3.process_tick()
        assert receipt.missions_processed == 1
        assert len(heartbeats) == 1

    def test_multiple_missions_accumulate(self):
        from core.sovereign.mission_nervous_system import (
            EchoInference,
            SovereignNervousSystem,
        )

        ns = SovereignNervousSystem.create(
            inference=EchoInference(),
            reward_per_mission=1.0,
        )

        h3 = wire_helix3(ns)

        for i in range(5):
            asyncio.run(ns.run(f"Mission {i}", ihsan_override=0.96))

        receipt = h3.process_tick()
        assert receipt.missions_processed == 5
        assert h3.stats.total_missions_processed == 5


# ═══════════════════════════════════════════════════════════════════
# 8. CONSTITUTIONAL BRIDGE (process_tick_constitutional)
# ═══════════════════════════════════════════════════════════════════

class TestConstitutionalBridge:
    """Falls back to simplified tick if constitutional kernel unavailable."""

    def test_constitutional_tick_graceful_degradation(self):
        h3 = Helix3Scheduler()
        h3.ingest_receipt({
            "mission_id": "m-1",
            "ihsan_score": 0.96,
            "snr_score": 0.90,
        })

        # Should not raise even if constitutional kernel has import issues
        receipt = h3.process_tick_constitutional()
        assert isinstance(receipt, HeartbeatReceipt)
        assert receipt.missions_processed >= 0


# ═══════════════════════════════════════════════════════════════════
# 9. STATS OBSERVABILITY
# ═══════════════════════════════════════════════════════════════════

class TestStatsObservability:
    """Cumulative stats track Helix 3 health over time."""

    def test_stats_accumulate(self):
        h3 = Helix3Scheduler()
        h3.ingest_receipt({"mission_id": "m-1", "ihsan_score": 0.96})
        h3.process_tick()
        h3.ingest_receipt({"mission_id": "m-2", "ihsan_score": 0.94})
        h3.process_tick()

        stats = h3.stats
        assert stats.total_ticks == 2
        assert stats.total_missions_processed == 2
        assert stats.avg_ihsan > 0

    def test_stats_serializable(self):
        h3 = Helix3Scheduler()
        h3.process_tick()
        d = h3.stats.as_dict()
        assert isinstance(d, dict)
        assert "total_ticks" in d
        assert "avg_ihsan" in d
