"""
SAT Controller Tests — POI-008 Comprehensive Suite.

Standing on Giants:
- Ostrom (1990): Commons governance
- Gini (1912): Inequality measurement
- Al-Ghazali (1097): Proportional justice (zakat)
- Axelrod (1984): Cooperation dynamics
"""

import pytest
from datetime import datetime, timezone

from core.proof_engine.poi_engine import (
    ContributionMetadata,
    ContributionType,
    PoIConfig,
    PoIOrchestrator,
    compute_gini,
)
from core.sovereign.sat_controller import (
    RebalancingEvent,
    SATController,
    URPSnapshot,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def config():
    """Default PoI config."""
    return PoIConfig()


@pytest.fixture
def orchestrator(config):
    """PoI orchestrator with config."""
    return PoIOrchestrator(config)


@pytest.fixture
def sat(orchestrator, config):
    """SAT controller with orchestrator."""
    return SATController(poi_orchestrator=orchestrator, config=config)


def _make_contribution(
    contributor_id: str,
    content_hash: str = "",
    snr: float = 0.95,
    ihsan: float = 0.96,
) -> ContributionMetadata:
    """Helper to create contributions."""
    if not content_hash:
        content_hash = f"hash_{contributor_id}_{snr}"
    return ContributionMetadata(
        contributor_id=contributor_id,
        contribution_type=ContributionType.CODE,
        content_hash=content_hash,
        snr_score=snr,
        ihsan_score=ihsan,
    )


# =============================================================================
# URP SNAPSHOT
# =============================================================================

class TestURPSnapshot:
    """Tests for URPSnapshot."""

    def test_to_dict(self):
        """to_dict() includes all fields."""
        snap = URPSnapshot(
            total_compute_credits=1000,
            allocated_credits=800,
            available_credits=200,
            holder_credits={"alice": 500, "bob": 300},
            gini_coefficient=0.25,
        )
        d = snap.to_dict()
        assert d["total_compute_credits"] == 1000
        assert d["num_holders"] == 2
        assert d["gini_coefficient"] == 0.25

    def test_default_values(self):
        """Default snapshot has zero values."""
        snap = URPSnapshot()
        assert snap.total_compute_credits == 0
        assert snap.gini_coefficient == 0.0


# =============================================================================
# REBALANCING EVENT
# =============================================================================

class TestRebalancingEvent:
    """Tests for RebalancingEvent."""

    def test_to_dict(self):
        """to_dict() includes all fields."""
        event = RebalancingEvent(
            event_id="sat_001",
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            reason="test",
            strategy="computational_zakat",
            gini_before=0.55,
            gini_after=0.40,
            credits_redistributed=100.0,
            contributors_affected=3,
        )
        d = event.to_dict()
        assert d["event_id"] == "sat_001"
        assert d["strategy"] == "computational_zakat"
        assert d["gini_before"] == 0.55
        assert d["gini_after"] == 0.40


# =============================================================================
# SAT CONTROLLER — CREDIT MANAGEMENT
# =============================================================================

class TestSATCreditManagement:
    """Tests for URP credit allocation."""

    def test_allocate_credits(self, sat):
        """Credits are allocated correctly."""
        sat.allocate_credits("alice", 1000)
        assert sat.get_credits("alice") == 1000

    def test_allocate_cumulative(self, sat):
        """Multiple allocations accumulate."""
        sat.allocate_credits("alice", 500)
        sat.allocate_credits("alice", 300)
        assert sat.get_credits("alice") == 800

    def test_adjust_positive(self, sat):
        """Positive adjustment increases credits."""
        sat.allocate_credits("alice", 500)
        sat.adjust_credits("alice", 200)
        assert sat.get_credits("alice") == 700

    def test_adjust_negative(self, sat):
        """Negative adjustment decreases credits."""
        sat.allocate_credits("alice", 500)
        sat.adjust_credits("alice", -200)
        assert sat.get_credits("alice") == 300

    def test_adjust_floor_zero(self, sat):
        """Credits cannot go below zero."""
        sat.allocate_credits("alice", 100)
        sat.adjust_credits("alice", -500)
        assert sat.get_credits("alice") == 0

    def test_get_credits_unknown(self, sat):
        """Unknown contributor returns 0."""
        assert sat.get_credits("unknown") == 0

    def test_urp_snapshot(self, sat):
        """URP snapshot reflects current state."""
        sat.allocate_credits("alice", 500)
        sat.allocate_credits("bob", 300)
        snap = sat.get_urp_snapshot()
        assert snap.total_compute_credits == 800
        assert len(snap.holder_credits) == 2


# =============================================================================
# SAT CONTROLLER — REBALANCING
# =============================================================================

class TestSATRebalancing:
    """Tests for SAT rebalancing logic."""

    def test_no_rebalance_when_equal(self, sat):
        """Equal distribution doesn't trigger rebalancing."""
        sat.allocate_credits("alice", 100)
        sat.allocate_credits("bob", 100)
        sat.allocate_credits("carol", 100)
        event = sat.check_and_rebalance()
        assert event is None

    def test_rebalance_triggered_by_inequality(self, sat):
        """High inequality triggers rebalancing."""
        sat.allocate_credits("alice", 10000)
        sat.allocate_credits("bob", 100)
        sat.allocate_credits("carol", 50)
        event = sat.check_and_rebalance()
        assert event is not None
        assert event.gini_after <= event.gini_before

    def test_zakat_strategy(self, sat):
        """Computational zakat collects from excess."""
        sat.allocate_credits("alice", 10000)
        sat.allocate_credits("bob", 100)
        event = sat.rebalance(
            reason="test",
            strategy="computational_zakat",
        )
        assert event.credits_redistributed > 0

    def test_progressive_strategy(self, sat):
        """Progressive redistribution works."""
        sat.allocate_credits("alice", 10000)
        sat.allocate_credits("bob", 100)
        event = sat.rebalance(
            reason="test",
            strategy="progressive_redistribution",
        )
        assert event.credits_redistributed > 0
        assert event.gini_after <= event.gini_before

    def test_rebalancing_preserves_total(self, sat):
        """Total credits are conserved during rebalancing."""
        sat.allocate_credits("alice", 10000)
        sat.allocate_credits("bob", 1000)
        sat.allocate_credits("carol", 100)
        total_before = sum(sat._urp_credits.values())
        sat.rebalance(reason="test", strategy="computational_zakat")
        total_after = sum(sat._urp_credits.values())
        # Allow rounding tolerance (int conversion)
        assert abs(total_before - total_after) <= len(sat._urp_credits)

    def test_unknown_strategy(self, sat):
        """Unknown strategy doesn't crash."""
        sat.allocate_credits("alice", 1000)
        event = sat.rebalance(reason="test", strategy="unknown_strategy")
        assert event.credits_redistributed == 0.0

    def test_rebalancing_history(self, sat):
        """Rebalancing events are recorded."""
        sat.allocate_credits("alice", 10000)
        sat.allocate_credits("bob", 100)
        sat.rebalance(reason="test_1", strategy="computational_zakat")
        sat.rebalance(reason="test_2", strategy="progressive_redistribution")
        history = sat.get_rebalancing_history()
        assert len(history) == 2
        assert history[0]["reason"] == "test_1"
        assert history[1]["reason"] == "test_2"

    def test_empty_credits_no_crash(self, sat):
        """Rebalancing with no credits doesn't crash."""
        event = sat.rebalance(reason="test", strategy="computational_zakat")
        assert event.credits_redistributed == 0.0


# =============================================================================
# SAT CONTROLLER — EPOCH FINALIZATION
# =============================================================================

class TestSATEpochFinalization:
    """Tests for SAT epoch finalization."""

    def test_finalize_empty_epoch(self, sat):
        """Empty epoch finalizes without error."""
        result = sat.finalize_epoch()
        assert result["total_contributors"] == 0
        assert result["epochs_finalized"] == 1

    def test_finalize_with_contributions(self, sat, orchestrator):
        """Epoch with contributions produces tokens and credits."""
        for name in ["alice", "bob", "carol"]:
            meta = _make_contribution(name, content_hash=f"epoch_test_{name}")
            orchestrator.register_contribution(meta)

        result = sat.finalize_epoch(epoch_reward=1000.0)
        assert result["total_contributors"] == 3
        assert result["tokens_distributed"] > 0

        # Contributors should have URP credits
        for name in ["alice", "bob", "carol"]:
            assert sat.get_credits(name) > 0

    def test_multiple_epochs(self, sat, orchestrator):
        """Multiple epochs accumulate credits."""
        meta = _make_contribution("alice", content_hash="epoch_1")
        orchestrator.register_contribution(meta)
        sat.finalize_epoch()

        meta2 = _make_contribution("alice", content_hash="epoch_2")
        orchestrator.register_contribution(meta2)
        sat.finalize_epoch()

        assert sat._epochs_finalized == 2
        assert sat.get_credits("alice") > 0

    def test_no_orchestrator(self, config):
        """Finalize without orchestrator returns error."""
        sat = SATController(poi_orchestrator=None, config=config)
        result = sat.finalize_epoch()
        assert "error" in result

    def test_epoch_triggers_rebalancing(self):
        """Epoch with high inequality triggers automatic rebalancing."""
        config = PoIConfig(gini_rebalance_threshold=0.01)  # Very low threshold
        orch = PoIOrchestrator(config)
        sat = SATController(poi_orchestrator=orch, config=config)

        # Create unequal contributions
        for i in range(5):
            meta = _make_contribution(
                f"contributor_{i}",
                content_hash=f"trigger_test_{i}",
                snr=0.85 + i * 0.03,
                ihsan=0.90 + i * 0.02,
            )
            orch.register_contribution(meta)

        result = sat.finalize_epoch()
        # With very low threshold, rebalancing should trigger
        assert result["rebalancing_triggered"] is True
        assert "rebalancing" in result


# =============================================================================
# SAT CONTROLLER — STATS AND AUDIT
# =============================================================================

class TestSATStats:
    """Tests for SAT statistics and audit."""

    def test_initial_stats(self, sat):
        """Initial stats are zero."""
        stats = sat.get_stats()
        assert stats["total_holders"] == 0
        assert stats["total_credits"] == 0
        assert stats["rebalancing_events"] == 0
        assert stats["epochs_finalized"] == 0

    def test_stats_after_operations(self, sat):
        """Stats reflect operations."""
        sat.allocate_credits("alice", 500)
        sat.allocate_credits("bob", 300)
        stats = sat.get_stats()
        assert stats["total_holders"] == 2
        assert stats["total_credits"] == 800
        assert stats["gini_threshold"] == 0.45

    def test_top_contributors(self, sat):
        """Top contributors are sorted by credits."""
        sat.allocate_credits("alice", 1000)
        sat.allocate_credits("bob", 500)
        sat.allocate_credits("carol", 100)
        top = sat.get_top_contributors(limit=2)
        assert len(top) == 2
        assert top[0]["contributor_id"] == "alice"
        assert top[0]["credits"] == 1000
        assert top[1]["contributor_id"] == "bob"

    def test_empty_rebalancing_history(self, sat):
        """Empty history returns empty list."""
        history = sat.get_rebalancing_history()
        assert history == []


# =============================================================================
# INTEGRATION: PoI + SAT PIPELINE
# =============================================================================

class TestPoISATPipeline:
    """Integration tests for the full PoI + SAT pipeline."""

    def test_full_pipeline(self):
        """Full pipeline: register -> epoch -> distribute -> rebalance."""
        config = PoIConfig()
        orch = PoIOrchestrator(config)
        sat = SATController(poi_orchestrator=orch, config=config)

        # Register contributions
        for i, name in enumerate(["alice", "bob", "carol", "dave"]):
            meta = _make_contribution(
                name,
                content_hash=f"pipeline_test_{name}",
                snr=0.90 + i * 0.02,
                ihsan=0.92 + i * 0.01,
            )
            orch.register_contribution(meta)

        # Add some citations
        orch.add_citation("bob", "alice")
        orch.add_citation("carol", "alice")
        orch.add_citation("dave", "alice")

        # Finalize epoch
        result = sat.finalize_epoch(epoch_reward=1000.0)
        assert result["total_contributors"] == 4
        assert result["tokens_distributed"] > 0

        # Check credits were distributed
        assert sat.get_credits("alice") > 0
        assert sat.get_credits("bob") > 0

        # Alice should have higher credits (most cited)
        assert sat.get_credits("alice") >= sat.get_credits("dave")

    def test_pipeline_gini_tracking(self):
        """Gini coefficient is tracked across epochs."""
        config = PoIConfig()
        orch = PoIOrchestrator(config)
        sat = SATController(poi_orchestrator=orch, config=config)

        # Epoch 1
        meta = _make_contribution("alice", content_hash="gini_1")
        orch.register_contribution(meta)
        result1 = sat.finalize_epoch()

        stats = sat.get_stats()
        assert "gini_coefficient" in stats

    def test_pipeline_multi_epoch_growth(self):
        """Credits grow across epochs."""
        config = PoIConfig()
        orch = PoIOrchestrator(config)
        sat = SATController(poi_orchestrator=orch, config=config)

        for epoch in range(3):
            meta = _make_contribution(
                "alice", content_hash=f"growth_{epoch}"
            )
            orch.register_contribution(meta)
            sat.finalize_epoch()

        assert sat.get_credits("alice") > 0
        assert sat._epochs_finalized == 3
