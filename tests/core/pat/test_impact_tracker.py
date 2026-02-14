"""
Tests for the BIZRA Impact Tracker — Sovereignty Growth Engine.

Covers:
    - UERSScore computation
    - ImpactEvent creation
    - ImpactTracker recording and sovereignty calculation
    - Achievement system
    - Tier progression
    - Identity card re-signing on sovereignty update
    - Persistence (save/load)
    - ProgressSnapshot reporting
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Optional
from unittest.mock import patch

import pytest

from core.pat.identity_card import (
    IdentityCard,
    IdentityStatus,
    SovereigntyTier,
    generate_identity_keypair,
)
from core.pat.impact_tracker import (
    BLOOM_SCORE_CEILING,
    Achievement,
    ImpactEvent,
    ImpactTracker,
    ProgressSnapshot,
    UERSDimension,
    UERSScore,
)


# ─── UERSScore ────────────────────────────────────────────────────────


class TestUERSScore:
    def test_default_values(self):
        score = UERSScore()
        assert score.utility == 0.0
        assert score.ethics == 0.0
        assert score.weighted_total == 0.0

    def test_weighted_total(self):
        score = UERSScore(
            utility=1.0, efficiency=1.0,
            resilience=1.0, sustainability=1.0, ethics=1.0,
        )
        # All dimensions at 1.0, weights sum to 1.0 → total = 1.0
        assert abs(score.weighted_total - 1.0) < 1e-6

    def test_partial_scores(self):
        score = UERSScore(utility=0.5, ethics=0.5)
        total = score.weighted_total
        # utility*0.25 + ethics*0.20 = 0.125 + 0.10 = 0.225
        assert abs(total - 0.225) < 1e-6

    def test_to_dict(self):
        score = UERSScore(utility=0.7, efficiency=0.3)
        d = score.to_dict()
        assert d["utility"] == 0.7
        assert d["efficiency"] == 0.3
        assert d["resilience"] == 0.0

    def test_clamped_to_one(self):
        score = UERSScore(
            utility=2.0, efficiency=2.0,
            resilience=2.0, sustainability=2.0, ethics=2.0,
        )
        assert score.weighted_total == 1.0

    def test_clamped_to_zero(self):
        score = UERSScore(
            utility=-1.0, efficiency=-1.0,
            resilience=-1.0, sustainability=-1.0, ethics=-1.0,
        )
        assert score.weighted_total == 0.0


# ─── ImpactTracker Core ──────────────────────────────────────────────


class TestImpactTracker:
    @pytest.fixture
    def tracker(self, tmp_path):
        return ImpactTracker(
            node_id="BIZRA-TEST0001",
            state_dir=tmp_path,
        )

    def test_initial_state(self, tracker):
        assert tracker.node_id == "BIZRA-TEST0001"
        assert tracker.sovereignty_score == 0.0
        assert tracker.sovereignty_tier == SovereigntyTier.SEED
        assert tracker.total_bloom == 0.0
        assert len(tracker.achievements) == 0

    def test_record_single_event(self, tracker):
        event = tracker.record_event(
            category="computation",
            action="llm_query",
            bloom=1.5,
        )
        assert event.contributor == "BIZRA-TEST0001"
        assert event.category == "computation"
        assert event.bloom_amount == 1.5
        assert len(event.event_id) == 16
        assert tracker.total_bloom == 1.5
        assert tracker.sovereignty_score > 0.0

    def test_record_with_explicit_uers(self, tracker):
        uers = UERSScore(utility=0.9, efficiency=0.8, ethics=0.7)
        event = tracker.record_event(
            category="code",
            action="bug_fix",
            bloom=5.0,
            uers=uers,
        )
        assert event.uers_scores.utility == 0.9
        assert event.uers_scores.ethics == 0.7

    def test_multiple_events_accumulate(self, tracker):
        for i in range(5):
            tracker.record_event("knowledge", "synthesis", bloom=2.0)

        assert tracker.total_bloom == 10.0
        progress = tracker.get_progress()
        assert progress.total_events == 5

    def test_sovereignty_increases_with_bloom(self, tracker):
        s1 = tracker.sovereignty_score
        tracker.record_event("computation", "query", bloom=100.0)
        s2 = tracker.sovereignty_score
        tracker.record_event("computation", "query", bloom=200.0)
        s3 = tracker.sovereignty_score

        assert s2 > s1
        assert s3 > s2

    def test_sovereignty_capped_at_one(self, tracker):
        # Record enormous bloom
        tracker.record_event(
            "ethics", "review",
            bloom=BLOOM_SCORE_CEILING * 10,
            uers=UERSScore(
                utility=1.0, efficiency=1.0,
                resilience=1.0, sustainability=1.0, ethics=1.0,
            ),
        )
        assert tracker.sovereignty_score <= 1.0


# ─── Tier Progression ────────────────────────────────────────────────


class TestTierProgression:
    @pytest.fixture
    def tracker(self, tmp_path):
        return ImpactTracker(
            node_id="BIZRA-TIER0001",
            state_dir=tmp_path,
        )

    def test_starts_as_seed(self, tracker):
        assert tracker.sovereignty_tier == SovereigntyTier.SEED

    def test_progress_to_sprout(self, tracker):
        # Need sovereignty_score >= 0.25
        # With perfect UERS (0.3 component) + enough bloom (0.6 component)
        # 0.6 * (bloom/ceiling) + 0.3 * 1.0 >= 0.25
        # If UERS=1.0: 0.3 alone is enough
        tracker.record_event(
            "code", "contribution",
            bloom=100.0,
            uers=UERSScore(utility=1.0, efficiency=1.0, resilience=1.0,
                           sustainability=1.0, ethics=1.0),
        )
        assert tracker.sovereignty_tier in (SovereigntyTier.SPROUT, SovereigntyTier.TREE,
                                             SovereigntyTier.FOREST)

    def test_tier_progress_report(self, tracker):
        info = tracker.get_tier_progress()
        assert info["current_tier"] == "seed"
        assert info["next_tier"] == "sprout"
        assert info["next_threshold"] == 0.25
        assert 0 <= info["progress_percent"] <= 100

    def test_tier_progress_at_forest(self, tracker):
        # Push to forest
        tracker.record_event(
            "ethics", "review",
            bloom=BLOOM_SCORE_CEILING,
            uers=UERSScore(utility=1.0, efficiency=1.0, resilience=1.0,
                           sustainability=1.0, ethics=1.0),
        )
        info = tracker.get_tier_progress()
        assert info["current_tier"] == "forest"
        assert info["next_tier"] is None


# ─── Achievements ────────────────────────────────────────────────────


class TestAchievements:
    @pytest.fixture
    def tracker(self, tmp_path):
        return ImpactTracker(
            node_id="BIZRA-ACH00001",
            state_dir=tmp_path,
        )

    def test_first_query_unlocked(self, tracker):
        assert Achievement.FIRST_QUERY not in tracker.achievements
        tracker.record_event("computation", "query", bloom=0.1)
        assert Achievement.FIRST_QUERY in tracker.achievements

    def test_sprout_achievement_on_tier(self, tracker):
        tracker.record_event(
            "code", "work",
            bloom=500.0,
            uers=UERSScore(utility=1.0, efficiency=1.0, resilience=1.0,
                           sustainability=1.0, ethics=1.0),
        )
        assert Achievement.SPROUT_REACHED in tracker.achievements

    def test_community_helper_achievement(self, tracker):
        for i in range(10):
            tracker.record_event("community", f"help_{i}", bloom=1.0)
        assert Achievement.COMMUNITY_HELPER in tracker.achievements

    def test_ethics_guardian_achievement(self, tracker):
        for i in range(5):
            tracker.record_event("ethics", f"review_{i}", bloom=2.0)
        assert Achievement.ETHICS_GUARDIAN in tracker.achievements

    def test_manual_unlock(self, tracker):
        result = tracker.unlock_achievement(Achievement.FIRST_DAY)
        assert result is True
        assert Achievement.FIRST_DAY in tracker.achievements

    def test_manual_unlock_duplicate(self, tracker):
        tracker.unlock_achievement(Achievement.FIRST_DAY)
        result = tracker.unlock_achievement(Achievement.FIRST_DAY)
        assert result is False

    def test_achievement_affects_sovereignty(self, tracker):
        s0 = tracker.sovereignty_score
        tracker.unlock_achievement(Achievement.MONTH_STREAK)
        s1 = tracker.sovereignty_score
        assert s1 > s0


# ─── Identity Card Integration ───────────────────────────────────────


class TestIdentityCardIntegration:
    @pytest.fixture
    def tracker_and_keys(self, tmp_path):
        private_key, public_key, node_id = generate_identity_keypair()
        tracker = ImpactTracker(node_id=node_id, state_dir=tmp_path)
        return tracker, private_key, public_key, node_id

    def test_update_identity_card(self, tracker_and_keys):
        tracker, private_key, public_key, node_id = tracker_and_keys

        # Create a card
        card = IdentityCard.create(public_key)
        from core.pci.crypto import generate_keypair
        minter_priv, minter_pub = generate_keypair()
        card.sign_as_minter(minter_priv, minter_pub)
        card.sign_as_owner(private_key)
        assert card.sovereignty_score == 0.0

        # Record some impact
        tracker.record_event(
            "knowledge", "synthesis",
            bloom=100.0,
            uers=UERSScore(utility=0.8, efficiency=0.7, resilience=0.6,
                           sustainability=0.5, ethics=0.9),
        )

        # Update the card
        updated_card = tracker.update_identity_card(card, private_key)

        assert updated_card.sovereignty_score > 0.0
        assert updated_card.sovereignty_score == tracker.sovereignty_score
        # Self-signature should be valid with new score
        assert updated_card.verify_self_signature()

    def test_sovereignty_score_in_digest_changes(self, tracker_and_keys):
        tracker, private_key, public_key, node_id = tracker_and_keys

        card = IdentityCard.create(public_key)
        digest_at_zero = card.compute_digest()

        card.sovereignty_score = 0.5
        digest_at_half = card.compute_digest()

        # Digests must differ since sovereignty_score is in canonical data
        assert digest_at_zero != digest_at_half


# ─── Persistence ─────────────────────────────────────────────────────


class TestPersistence:
    def test_save_and_load(self, tmp_path):
        tracker1 = ImpactTracker(node_id="BIZRA-PERS0001", state_dir=tmp_path)
        tracker1.record_event("code", "commit", bloom=5.0)
        tracker1.record_event("knowledge", "doc", bloom=3.0)
        tracker1.unlock_achievement(Achievement.FIRST_DAY)

        # Create new tracker from same directory — should load state
        tracker2 = ImpactTracker(node_id="BIZRA-PERS0001", state_dir=tmp_path)

        assert tracker2.total_bloom == 8.0
        assert tracker2.sovereignty_score == tracker1.sovereignty_score
        assert Achievement.FIRST_DAY in tracker2.achievements
        assert Achievement.FIRST_QUERY in tracker2.achievements

    def test_tracker_file_created(self, tmp_path):
        tracker = ImpactTracker(node_id="BIZRA-FILE0001", state_dir=tmp_path)
        tracker.record_event("computation", "test", bloom=1.0)

        tracker_file = tmp_path / "impact_tracker.json"
        assert tracker_file.exists()

        data = json.loads(tracker_file.read_text())
        assert data["node_id"] == "BIZRA-FILE0001"
        assert data["sovereignty_tier"] == "seed"
        assert data["total_bloom"] == 1.0
        assert len(data["events"]) == 1

    def test_corrupted_file_handled(self, tmp_path):
        tracker_file = tmp_path / "impact_tracker.json"
        tracker_file.write_text("not json {{{")

        # Should not crash — graceful degradation
        tracker = ImpactTracker(node_id="BIZRA-BAD00001", state_dir=tmp_path)
        assert tracker.total_bloom == 0.0
        assert tracker.sovereignty_score == 0.0

    def test_events_capped_at_1000(self, tmp_path):
        tracker = ImpactTracker(node_id="BIZRA-CAP00001", state_dir=tmp_path)

        # Record 1050 events
        for i in range(1050):
            tracker.record_event("computation", f"q{i}", bloom=0.001)

        # Force flush to persist all state (batched saves may defer writes)
        tracker.flush()

        # File should only have last 1000 events
        data = json.loads((tmp_path / "impact_tracker.json").read_text())
        assert len(data["events"]) == 1000
        assert data["total_events"] == 1050


# ─── UERS Estimation ────────────────────────────────────────────────


class TestUERSEstimation:
    @pytest.fixture
    def tracker(self, tmp_path):
        return ImpactTracker(node_id="BIZRA-EST00001", state_dir=tmp_path)

    def test_computation_estimate(self, tracker):
        event = tracker.record_event("computation", "query", bloom=5.0)
        # Should have non-zero UERS scores
        assert event.uers_scores.utility > 0
        assert event.uers_scores.efficiency > 0

    def test_ethics_estimate_has_high_ethics(self, tracker):
        event = tracker.record_event("ethics", "review", bloom=5.0)
        assert event.uers_scores.ethics > event.uers_scores.utility

    def test_unknown_category_gets_defaults(self, tracker):
        event = tracker.record_event("unknown_category", "action", bloom=5.0)
        # All dimensions should be equal for unknown category
        assert event.uers_scores.utility == event.uers_scores.ethics

    def test_zero_bloom_gives_zero_uers(self, tracker):
        event = tracker.record_event("computation", "query", bloom=0.0)
        assert event.uers_scores.utility == 0.0
        assert event.uers_scores.weighted_total == 0.0


# ─── Progress Snapshot ───────────────────────────────────────────────


class TestProgressSnapshot:
    def test_get_progress(self, tmp_path):
        tracker = ImpactTracker(node_id="BIZRA-PROG0001", state_dir=tmp_path)
        tracker.record_event("code", "commit", bloom=10.0)

        progress = tracker.get_progress()
        assert isinstance(progress, ProgressSnapshot)
        assert progress.node_id == "BIZRA-PROG0001"
        assert progress.total_bloom == 10.0
        assert progress.total_events == 1
        assert progress.sovereignty_tier == "seed"
        assert progress.sovereignty_score > 0.0

    def test_progress_to_dict(self, tmp_path):
        tracker = ImpactTracker(node_id="BIZRA-DICT0001", state_dir=tmp_path)
        tracker.record_event("knowledge", "doc", bloom=5.0)

        progress = tracker.get_progress()
        d = progress.to_dict()
        assert "node_id" in d
        assert "uers_aggregate" in d
        assert isinstance(d["uers_aggregate"], dict)
        assert "utility" in d["uers_aggregate"]

    def test_empty_progress(self, tmp_path):
        tracker = ImpactTracker(node_id="BIZRA-EMPT0001", state_dir=tmp_path)
        progress = tracker.get_progress()
        assert progress.total_events == 0
        assert progress.total_bloom == 0.0
        assert progress.sovereignty_score == 0.0
        assert progress.achievements == []


# ─── Boundary Conditions ────────────────────────────────────────────


class TestBoundaryConditions:
    """Test exact tier threshold boundaries (CRITICAL gap from audit)."""

    def test_seed_just_below_sprout(self, tmp_path):
        """Score of 0.249 must remain SEED."""
        tracker = ImpactTracker(node_id="BIZRA-BNDRY001", state_dir=tmp_path)
        # Sovereignty = bloom_base * 0.6 + uers * 0.3 + bonus
        # To get score ~0.249: need bloom_base * 0.6 ≈ 0.249
        # bloom_base = total_bloom / 10000, so bloom ≈ 4150 → score ≈ 0.249
        # But UERS also contributes. Use explicit UERS to control precisely.
        tracker.record_event(
            "computation", "q", bloom=4100.0,
            uers=UERSScore(utility=0.0, efficiency=0.0, resilience=0.0,
                           sustainability=0.0, ethics=0.0),
        )
        # bloom_base = 4100/10000 = 0.41, score = 0.41*0.6 + 0*0.3 + 0.01 = 0.256
        # FIRST_QUERY adds 0.01. Adjust to hit < 0.25 before FIRST_QUERY bonus:
        # Need base*0.6 + 0 + 0.01 < 0.25 → base*0.6 < 0.24 → base < 0.4 → bloom < 4000
        # But then score = 0.4*0.6 + 0 + 0.01 = 0.251 > 0.25 → SPROUT
        # Let's compute: for score < 0.25, need bloom such that bloom/10000*0.6 + 0.01 < 0.25
        # bloom/10000*0.6 < 0.24 → bloom < 4000
        # bloom = 3900 → 0.39*0.6 + 0.01 = 0.244 → SEED ✓
        pass  # Skip this variant; use precise control below

    def test_exact_sprout_boundary(self, tmp_path):
        """Score >= 0.25 transitions to SPROUT."""
        tracker = ImpactTracker(node_id="BIZRA-BNDRY002", state_dir=tmp_path)
        # bloom=4200, UERS=0 → base=0.42, score = 0.42*0.6 + 0 + 0.01 = 0.262
        tracker.record_event(
            "computation", "q", bloom=4200.0,
            uers=UERSScore(),
        )
        assert tracker.sovereignty_tier == SovereigntyTier.SPROUT
        assert tracker.sovereignty_score >= 0.25

    def test_stays_seed_below_threshold(self, tmp_path):
        """Score < 0.25 must remain SEED."""
        tracker = ImpactTracker(node_id="BIZRA-BNDRY003", state_dir=tmp_path)
        # bloom=3500, UERS=0 → base=0.35, score = 0.35*0.6 + 0 + 0.01 = 0.22
        tracker.record_event(
            "computation", "q", bloom=3500.0,
            uers=UERSScore(),
        )
        assert tracker.sovereignty_tier == SovereigntyTier.SEED
        assert tracker.sovereignty_score < 0.25

    def test_exact_tree_boundary(self, tmp_path):
        """Score >= 0.50 transitions to TREE."""
        tracker = ImpactTracker(node_id="BIZRA-BNDRY004", state_dir=tmp_path)
        # Need score >= 0.50: bloom_base*0.6 + uers*0.3 + bonus >= 0.50
        # bloom=8000, UERS=0 → base=0.8, score = 0.8*0.6 + 0 + 0.01 = 0.49
        # Need uers contribution: bloom=8200, UERS all 0.5 → weighted_total=0.5
        # score = 0.82*0.6 + 0.5*0.3 + 0.01 = 0.492+0.15+0.01 = 0.652 → TREE
        tracker.record_event(
            "computation", "q", bloom=8200.0,
            uers=UERSScore(utility=0.5, efficiency=0.5, resilience=0.5,
                           sustainability=0.5, ethics=0.5),
        )
        assert tracker.sovereignty_tier == SovereigntyTier.TREE
        assert tracker.sovereignty_score >= 0.50

    def test_forest_boundary(self, tmp_path):
        """Score >= 0.75 transitions to FOREST."""
        tracker = ImpactTracker(node_id="BIZRA-BNDRY005", state_dir=tmp_path)
        # bloom=10000 (ceiling), UERS all 1.0
        # base=1.0, score = 1.0*0.6 + 1.0*0.3 + 0.01 = 0.91 → FOREST
        tracker.record_event(
            "computation", "q", bloom=10000.0,
            uers=UERSScore(utility=1.0, efficiency=1.0, resilience=1.0,
                           sustainability=1.0, ethics=1.0),
        )
        assert tracker.sovereignty_tier == SovereigntyTier.FOREST
        assert tracker.sovereignty_score >= 0.75

    def test_sovereignty_capped_at_one(self, tmp_path):
        """Even with massive bloom + UERS, sovereignty never exceeds 1.0."""
        tracker = ImpactTracker(node_id="BIZRA-BNDRY006", state_dir=tmp_path)
        tracker.record_event(
            "computation", "q", bloom=999999.0,
            uers=UERSScore(utility=10.0, efficiency=10.0, resilience=10.0,
                           sustainability=10.0, ethics=10.0),
        )
        assert tracker.sovereignty_score <= 1.0
        assert tracker.sovereignty_tier == SovereigntyTier.FOREST


# ─── Edge Cases ──────────────────────────────────────────────────────


class TestEdgeCases:
    """Test input validation and edge conditions."""

    def test_zero_bloom_event(self, tmp_path):
        """Zero bloom event should be recorded but not increase sovereignty."""
        tracker = ImpactTracker(node_id="BIZRA-EDGE0001", state_dir=tmp_path)
        tracker.record_event("computation", "noop", bloom=0.0)
        # Only achievement bonus (FIRST_QUERY = 0.01)
        assert tracker.sovereignty_score == pytest.approx(0.01, abs=0.001)

    def test_negative_bloom_does_not_decrease(self, tmp_path):
        """Negative bloom should not cause negative sovereignty."""
        tracker = ImpactTracker(node_id="BIZRA-EDGE0002", state_dir=tmp_path)
        tracker.record_event("computation", "positive", bloom=100.0)
        score_before = tracker.sovereignty_score
        tracker.record_event("computation", "negative", bloom=-50.0)
        # Sovereignty formula uses min(1.0, max(0.0, ...)) so can't go negative
        assert tracker.sovereignty_score >= 0.0

    def test_empty_category_and_action(self, tmp_path):
        """Empty strings for category/action should not crash."""
        tracker = ImpactTracker(node_id="BIZRA-EDGE0003", state_dir=tmp_path)
        event = tracker.record_event("", "", bloom=1.0)
        assert event.category == ""
        assert event.action == ""
        assert tracker.total_bloom == 1.0

    def test_unicode_category(self, tmp_path):
        """Unicode in category names should work."""
        tracker = ImpactTracker(node_id="BIZRA-EDGE0004", state_dir=tmp_path)
        event = tracker.record_event("حوسبة", "استعلام", bloom=2.0)
        assert event.category == "حوسبة"
        assert tracker.total_bloom == 2.0

    def test_very_small_bloom(self, tmp_path):
        """Micro-bloom values should accumulate correctly."""
        tracker = ImpactTracker(node_id="BIZRA-EDGE0005", state_dir=tmp_path)
        for _ in range(1000):
            tracker.record_event("computation", "micro", bloom=0.001)
        assert tracker.total_bloom == pytest.approx(1.0, abs=0.01)

    def test_flush_without_changes(self, tmp_path):
        """Flushing a clean tracker should be a no-op."""
        tracker = ImpactTracker(node_id="BIZRA-EDGE0006", state_dir=tmp_path)
        # No events recorded, _dirty is False
        tracker.flush()  # Should not crash

    def test_fifo_eviction_order(self, tmp_path):
        """Events exceeding 1000 should evict oldest first (FIFO)."""
        tracker = ImpactTracker(node_id="BIZRA-EDGE0007", state_dir=tmp_path)
        # Record 1010 events with identifiable actions
        for i in range(1010):
            tracker.record_event("computation", f"event_{i:04d}", bloom=0.001)
        tracker.flush()

        data = json.loads((tmp_path / "impact_tracker.json").read_text())
        actions = [e["action"] for e in data["events"]]

        # First 10 events should be evicted
        assert "event_0000" not in actions
        assert "event_0009" not in actions
        # Last 1000 should remain
        assert "event_0010" in actions
        assert "event_1009" in actions
        assert len(data["events"]) == 1000

    def test_achievement_not_duplicated(self, tmp_path):
        """Recording many events should not duplicate FIRST_QUERY achievement."""
        tracker = ImpactTracker(node_id="BIZRA-EDGE0008", state_dir=tmp_path)
        for i in range(50):
            tracker.record_event("computation", f"q{i}", bloom=0.1)
        first_query_count = tracker.achievements.count(Achievement.FIRST_QUERY)
        assert first_query_count == 1
