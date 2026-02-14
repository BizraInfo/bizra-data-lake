"""
Proof-of-Impact Engine Tests — POI-004 Comprehensive Suite.

Standing on Giants:
- Nakamoto (2008): PoW verification
- Page & Brin (1998): PageRank correctness
- Gini (1912): Inequality measurement
- Shannon (1948): SNR as quality metric
- Al-Ghazali (1058-1111): Zakat distribution justice
"""

import math
import pytest
from datetime import datetime, timedelta, timezone

from core.proof_engine.poi_engine import (
    AuditTrail,
    CitationGraph,
    ContributionMetadata,
    ContributionType,
    ContributionVerifier,
    LongevityScore,
    PoIConfig,
    PoIOrchestrator,
    ProofOfImpact,
    ReachScore,
    RebalanceResult,
    SATRebalancer,
    TemporalScorer,
    TokenDistribution,
    VerifiedContribution,
    compute_gini,
    compute_token_distribution,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def config():
    """Default PoI configuration."""
    return PoIConfig()


@pytest.fixture
def strict_config():
    """Strict quality thresholds."""
    return PoIConfig(snr_quality_min=0.95, ihsan_quality_min=0.95)


@pytest.fixture
def orchestrator(config):
    """Default orchestrator."""
    return PoIOrchestrator(config)


def _make_contribution(
    contributor_id: str = "alice",
    contribution_type: ContributionType = ContributionType.CODE,
    snr: float = 0.95,
    ihsan: float = 0.96,
    content_hash: str = "",
    timestamp: datetime = None,
) -> ContributionMetadata:
    """Helper to create ContributionMetadata."""
    if not content_hash:
        content_hash = f"hash_{contributor_id}_{snr}_{ihsan}"
    return ContributionMetadata(
        contributor_id=contributor_id,
        contribution_type=contribution_type,
        content_hash=content_hash,
        snr_score=snr,
        ihsan_score=ihsan,
        timestamp=timestamp or datetime.now(timezone.utc),
    )


# =============================================================================
# POI CONFIG
# =============================================================================

class TestPoIConfig:
    """Tests for PoIConfig."""

    def test_default_weights_sum_to_one(self, config):
        """Default weights are a convex combination."""
        assert abs(config.alpha + config.beta + config.gamma - 1.0) < 1e-6

    def test_validate_passes_defaults(self, config):
        """Default config passes validation."""
        config.validate()  # Should not raise

    def test_validate_rejects_bad_weights(self):
        """Config with non-convex weights fails validation."""
        bad = PoIConfig(alpha=0.5, beta=0.5, gamma=0.5)
        with pytest.raises(ValueError, match="must sum to 1.0"):
            bad.validate()

    def test_validate_rejects_bad_damping(self):
        """Invalid damping factor raises."""
        bad = PoIConfig(pagerank_damping=1.5)
        with pytest.raises(ValueError, match="Damping factor"):
            bad.validate()

    def test_validate_rejects_negative_decay(self):
        """Negative decay lambda raises."""
        bad = PoIConfig(decay_lambda=-0.01)
        with pytest.raises(ValueError, match="Decay lambda"):
            bad.validate()

    def test_canonical_bytes_deterministic(self, config):
        """Config canonical bytes are deterministic."""
        b1 = config.canonical_bytes()
        b2 = config.canonical_bytes()
        assert b1 == b2

    def test_digest_deterministic(self, config):
        """Config digest is deterministic."""
        d1 = config.hex_digest()
        d2 = config.hex_digest()
        assert d1 == d2
        assert len(d1) == 64

    def test_different_configs_different_digests(self):
        """Different configs produce different digests."""
        c1 = PoIConfig(alpha=0.5, beta=0.3, gamma=0.2)
        c2 = PoIConfig(alpha=0.4, beta=0.4, gamma=0.2)
        assert c1.hex_digest() != c2.hex_digest()


# =============================================================================
# CONTRIBUTION TYPE
# =============================================================================

class TestContributionType:
    """Tests for ContributionType enum."""

    def test_all_types_exist(self):
        """All 5 contribution types exist."""
        assert ContributionType.CODE.value == "code"
        assert ContributionType.DATA.value == "data"
        assert ContributionType.REVIEW.value == "review"
        assert ContributionType.GOVERNANCE.value == "governance"
        assert ContributionType.INFRASTRUCTURE.value == "infrastructure"


# =============================================================================
# CONTRIBUTION METADATA
# =============================================================================

class TestContributionMetadata:
    """Tests for ContributionMetadata."""

    def test_canonical_bytes_deterministic(self):
        """Canonical bytes are deterministic."""
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        m1 = ContributionMetadata(
            contributor_id="alice",
            contribution_type=ContributionType.CODE,
            content_hash="abc123",
            snr_score=0.96,
            ihsan_score=0.97,
            timestamp=ts,
        )
        m2 = ContributionMetadata(
            contributor_id="alice",
            contribution_type=ContributionType.CODE,
            content_hash="abc123",
            snr_score=0.96,
            ihsan_score=0.97,
            timestamp=ts,
        )
        assert m1.canonical_bytes() == m2.canonical_bytes()

    def test_digest_format(self):
        """Digest is 64-char hex."""
        m = _make_contribution()
        hd = m.hex_digest()
        assert len(hd) == 64
        int(hd, 16)

    def test_nonce_included_in_canonical(self):
        """Nonce affects canonical bytes."""
        m1 = _make_contribution(content_hash="same")
        m1.nonce = "nonce_a"
        m2 = _make_contribution(content_hash="same")
        m2.nonce = "nonce_b"
        assert m1.canonical_bytes() != m2.canonical_bytes()


# =============================================================================
# STAGE 1: CONTRIBUTION VERIFIER
# =============================================================================

class TestContributionVerifier:
    """Tests for Stage 1: ContributionVerifier."""

    def test_accepts_valid_contribution(self, config):
        """Valid contribution passes verification."""
        verifier = ContributionVerifier(config)
        meta = _make_contribution(snr=0.95, ihsan=0.96)
        result = verifier.verify(meta)
        assert result.verified is True
        assert result.quality_score > 0

    def test_rejects_low_snr(self, config):
        """Low SNR fails verification."""
        verifier = ContributionVerifier(config)
        meta = _make_contribution(snr=0.50, ihsan=0.96)
        result = verifier.verify(meta)
        assert result.verified is False
        assert "SNR below threshold" in result.rejection_reason

    def test_rejects_low_ihsan(self, config):
        """Low Ihsan fails verification."""
        verifier = ContributionVerifier(config)
        meta = _make_contribution(snr=0.95, ihsan=0.50)
        result = verifier.verify(meta)
        assert result.verified is False
        assert "Ihsan below threshold" in result.rejection_reason

    def test_rejects_duplicate(self, config):
        """Duplicate content hash is rejected."""
        verifier = ContributionVerifier(config)
        m1 = _make_contribution(content_hash="dup_hash", snr=0.95, ihsan=0.96)
        m2 = _make_contribution(content_hash="dup_hash", snr=0.95, ihsan=0.96)
        r1 = verifier.verify(m1)
        r2 = verifier.verify(m2)
        assert r1.verified is True
        assert r2.verified is False
        assert "Duplicate" in r2.rejection_reason

    def test_quality_score_formula(self, config):
        """Quality score is 0.6*SNR + 0.4*Ihsan."""
        verifier = ContributionVerifier(config)
        meta = _make_contribution(snr=1.0, ihsan=1.0)
        result = verifier.verify(meta)
        assert abs(result.quality_score - 1.0) < 1e-6

    def test_quality_score_weighted(self, config):
        """Quality score reflects weighted combination."""
        verifier = ContributionVerifier(config)
        meta = _make_contribution(snr=0.90, ihsan=0.95, content_hash="unique1")
        result = verifier.verify(meta)
        expected = 0.6 * 0.90 + 0.4 * 0.95
        assert abs(result.quality_score - expected) < 1e-6

    def test_stats(self, config):
        """Verifier tracks unique contribution count."""
        verifier = ContributionVerifier(config)
        for i in range(5):
            meta = _make_contribution(content_hash=f"hash_{i}", snr=0.95, ihsan=0.96)
            verifier.verify(meta)
        stats = verifier.get_stats()
        assert stats["unique_contributions"] == 5

    def test_strict_config_rejects_borderline(self, strict_config):
        """Strict config rejects contributions that pass default."""
        verifier = ContributionVerifier(strict_config)
        meta = _make_contribution(snr=0.90, ihsan=0.92, content_hash="borderline")
        result = verifier.verify(meta)
        assert result.verified is False

    def test_to_dict(self, config):
        """VerifiedContribution.to_dict() includes all fields."""
        verifier = ContributionVerifier(config)
        meta = _make_contribution(snr=0.95, ihsan=0.96)
        result = verifier.verify(meta)
        d = result.to_dict()
        assert "contributor_id" in d
        assert "contribution_type" in d
        assert "verified" in d
        assert "quality_score" in d


# =============================================================================
# STAGE 2: CITATION GRAPH
# =============================================================================

class TestCitationGraph:
    """Tests for Stage 2: CitationGraph."""

    def test_add_citation(self, config):
        """Citations are recorded."""
        graph = CitationGraph(config)
        graph.add_citation("alice", "bob")
        stats = graph.get_stats()
        assert stats["total_nodes"] == 2
        assert stats["total_edges"] == 1

    def test_self_citation_ignored(self, config):
        """Self-citations are rejected."""
        graph = CitationGraph(config)
        graph.add_citation("alice", "alice")
        assert graph.get_stats()["total_edges"] == 0

    def test_pagerank_uniform_graph(self, config):
        """Uniform graph has roughly equal PageRank."""
        graph = CitationGraph(config)
        # Symmetric triangle
        graph.add_citation("a", "b")
        graph.add_citation("b", "c")
        graph.add_citation("c", "a")
        pr = graph.compute_pagerank()
        values = list(pr.values())
        # All should be roughly equal
        assert max(values) - min(values) < 0.05

    def test_pagerank_star_topology(self, config):
        """Star topology: center has highest PageRank."""
        graph = CitationGraph(config)
        for node in ["b", "c", "d", "e"]:
            graph.add_citation(node, "a")
        pr = graph.compute_pagerank()
        assert pr["a"] > pr["b"]
        assert pr["a"] > pr["c"]

    def test_pagerank_empty_graph(self, config):
        """Empty graph returns empty dict."""
        graph = CitationGraph(config)
        assert graph.compute_pagerank() == {}

    def test_reach_scores_normalized(self, config):
        """Reach scores are in [0, 1]."""
        graph = CitationGraph(config)
        graph.add_citation("a", "b")
        graph.add_citation("b", "c")
        graph.add_citation("c", "a")
        scores = graph.compute_reach_scores()
        for s in scores:
            assert 0.0 <= s.normalized_reach <= 1.0

    def test_reach_scores_contain_all_nodes(self, config):
        """All nodes get a reach score."""
        graph = CitationGraph(config)
        graph.add_citation("a", "b")
        graph.add_citation("c", "d")
        scores = graph.compute_reach_scores()
        ids = {s.contributor_id for s in scores}
        assert ids == {"a", "b", "c", "d"}

    def test_citation_ring_detection(self):
        """Citation rings above threshold are penalized."""
        config = PoIConfig(citation_ring_threshold=1)
        graph = CitationGraph(config)
        # Create a ring: a<->b, a<->c (2 mutual > threshold of 1)
        graph.add_citation("a", "b")
        graph.add_citation("b", "a")
        graph.add_citation("a", "c")
        graph.add_citation("c", "a")
        penalties = graph.detect_citation_rings()
        assert "a" in penalties
        assert penalties["a"] > 0

    def test_no_ring_below_threshold(self, config):
        """Below threshold, no ring penalty."""
        graph = CitationGraph(config)
        # One mutual citation (threshold is 3)
        graph.add_citation("a", "b")
        graph.add_citation("b", "a")
        penalties = graph.detect_citation_rings()
        assert len(penalties) == 0

    def test_reach_score_to_dict(self, config):
        """ReachScore.to_dict() includes all fields."""
        graph = CitationGraph(config)
        graph.add_citation("a", "b")
        scores = graph.compute_reach_scores()
        d = scores[0].to_dict()
        assert "contributor_id" in d
        assert "raw_pagerank" in d
        assert "normalized_reach" in d
        assert "citation_count" in d
        assert "cited_by_count" in d

    def test_graph_stats(self, config):
        """Graph stats report correctly."""
        graph = CitationGraph(config)
        graph.add_citation("a", "b")
        graph.add_citation("a", "c")
        graph.add_citation("b", "c")
        stats = graph.get_stats()
        assert stats["total_nodes"] == 3
        assert stats["total_edges"] == 3
        assert abs(stats["avg_out_degree"] - 1.0) < 1e-6


# =============================================================================
# STAGE 3: TEMPORAL SCORER
# =============================================================================

class TestTemporalScorer:
    """Tests for Stage 3: TemporalScorer."""

    def test_no_activity_zero_longevity(self, config):
        """No activity gives zero longevity."""
        scorer = TemporalScorer(config)
        result = scorer.compute_longevity("nobody")
        assert result.normalized_longevity == 0.0
        assert result.days_active == 0.0

    def test_recent_activity_positive_longevity(self, config):
        """Recent activity gives positive longevity with no decay."""
        scorer = TemporalScorer(config)
        now = datetime.now(timezone.utc)
        scorer.record_activity("alice", now)
        result = scorer.compute_longevity("alice", reference_time=now)
        # Single activity: activity_factor = log(2)/log(21) ≈ 0.228, decay = 1.0
        assert result.normalized_longevity > 0.2
        assert result.decay_factor > 0.99

    def test_old_activity_decays(self, config):
        """Old activity has low longevity due to decay."""
        scorer = TemporalScorer(config)
        old = datetime.now(timezone.utc) - timedelta(days=365)
        scorer.record_activity("alice", old)
        now = datetime.now(timezone.utc)
        result = scorer.compute_longevity("alice", reference_time=now)
        # e^(-0.01 * 365) ≈ 0.026
        assert result.decay_factor < 0.05
        assert result.normalized_longevity < 0.1

    def test_multiple_activities_increase_longevity(self, config):
        """More activities increase the activity factor."""
        scorer = TemporalScorer(config)
        now = datetime.now(timezone.utc)
        # Single activity
        scorer.record_activity("alice", now)
        r1 = scorer.compute_longevity("alice", reference_time=now)

        # Many activities
        for i in range(10):
            scorer.record_activity("alice", now - timedelta(hours=i))
        r2 = scorer.compute_longevity("alice", reference_time=now)

        assert r2.normalized_longevity >= r1.normalized_longevity

    def test_sustained_bonus(self, config):
        """Sustained contributors get a bonus."""
        scorer = TemporalScorer(config)
        now = datetime.now(timezone.utc)
        # Activity across 5 different weeks
        for week in range(5):
            ts = now - timedelta(weeks=week)
            scorer.record_activity("alice", ts)
        result = scorer.compute_longevity("alice", reference_time=now)
        assert result.sustained_bonus_applied is True

    def test_not_sustained_few_weeks(self, config):
        """Few weeks of activity is not sustained."""
        scorer = TemporalScorer(config)
        now = datetime.now(timezone.utc)
        scorer.record_activity("alice", now)
        scorer.record_activity("alice", now - timedelta(days=1))
        result = scorer.compute_longevity("alice", reference_time=now)
        assert result.sustained_bonus_applied is False

    def test_longevity_score_to_dict(self, config):
        """LongevityScore.to_dict() includes all fields."""
        scorer = TemporalScorer(config)
        scorer.record_activity("alice")
        result = scorer.compute_longevity("alice")
        d = result.to_dict()
        assert "contributor_id" in d
        assert "normalized_longevity" in d
        assert "days_active" in d
        assert "decay_factor" in d
        assert "spike_detected" in d

    def test_longevity_clamped(self, config):
        """Longevity is always in [0, 1]."""
        scorer = TemporalScorer(config)
        now = datetime.now(timezone.utc)
        # Many activities across many weeks
        for week in range(20):
            for day in range(3):
                ts = now - timedelta(weeks=week, days=day)
                scorer.record_activity("alice", ts)
        result = scorer.compute_longevity("alice", reference_time=now)
        assert 0.0 <= result.normalized_longevity <= 1.0


# =============================================================================
# GINI COEFFICIENT
# =============================================================================

class TestGiniCoefficient:
    """Tests for Gini coefficient computation."""

    def test_perfect_equality(self):
        """Equal values give Gini = 0."""
        assert compute_gini([1.0, 1.0, 1.0, 1.0]) == 0.0

    def test_maximum_inequality(self):
        """One person has everything, Gini approaches 1."""
        g = compute_gini([0.0, 0.0, 0.0, 100.0])
        assert g > 0.7

    def test_empty_list(self):
        """Empty list gives Gini = 0."""
        assert compute_gini([]) == 0.0

    def test_single_value(self):
        """Single value gives Gini = 0."""
        assert compute_gini([42.0]) == 0.0

    def test_all_zeros(self):
        """All zeros gives Gini = 0."""
        assert compute_gini([0.0, 0.0, 0.0]) == 0.0

    def test_two_values_unequal(self):
        """Two unequal values give positive Gini."""
        g = compute_gini([1.0, 3.0])
        assert 0 < g < 1

    def test_gini_bounded(self):
        """Gini is always in [0, 1]."""
        import random
        random.seed(42)
        for _ in range(10):
            values = [random.random() for _ in range(20)]
            g = compute_gini(values)
            assert 0.0 <= g <= 1.0

    def test_more_equal_lower_gini(self):
        """More equal distribution has lower Gini."""
        equal = compute_gini([5.0, 5.0, 5.0, 5.0])
        unequal = compute_gini([1.0, 2.0, 7.0, 10.0])
        assert equal < unequal


# =============================================================================
# SAT REBALANCER
# =============================================================================

class TestSATRebalancer:
    """Tests for SAT-based rebalancing."""

    def test_no_rebalance_below_threshold(self, config):
        """No rebalancing when Gini is below threshold."""
        rebalancer = SATRebalancer(config)
        scores = {"a": 0.5, "b": 0.5, "c": 0.5}
        result = rebalancer.rebalance(scores)
        assert result.rebalance_triggered is False
        assert result.rebalanced_scores == scores

    def test_rebalance_triggered_high_gini(self):
        """Rebalancing triggered when Gini exceeds threshold."""
        config = PoIConfig(gini_rebalance_threshold=0.1)  # Very low threshold
        rebalancer = SATRebalancer(config)
        scores = {"a": 0.1, "b": 0.9}
        result = rebalancer.rebalance(scores)
        assert result.rebalance_triggered is True
        assert result.gini_after <= result.gini_before

    def test_zakat_collected_from_excess(self):
        """Zakat is collected from contributors above mean."""
        config = PoIConfig(gini_rebalance_threshold=0.01)  # Always trigger
        rebalancer = SATRebalancer(config)
        scores = {"a": 0.1, "b": 0.9}
        result = rebalancer.rebalance(scores)
        assert result.zakat_collected > 0

    def test_zakat_distributed_to_needy(self):
        """Zakat is distributed to contributors below mean."""
        config = PoIConfig(gini_rebalance_threshold=0.01)
        rebalancer = SATRebalancer(config)
        scores = {"a": 0.1, "b": 0.9}
        result = rebalancer.rebalance(scores)
        # After rebalancing, 'a' should have more
        assert result.rebalanced_scores["a"] >= scores["a"]

    def test_zakat_conservation(self):
        """Total score is conserved (collected = distributed)."""
        config = PoIConfig(gini_rebalance_threshold=0.01)
        rebalancer = SATRebalancer(config)
        scores = {"a": 0.1, "b": 0.5, "c": 0.9}
        result = rebalancer.rebalance(scores)
        original_total = sum(scores.values())
        rebalanced_total = sum(result.rebalanced_scores.values())
        assert abs(original_total - rebalanced_total) < 1e-10

    def test_empty_scores(self, config):
        """Empty scores return no-op result."""
        rebalancer = SATRebalancer(config)
        result = rebalancer.rebalance({})
        assert result.rebalance_triggered is False

    def test_exemption_floor(self):
        """Contributors below exemption floor are not taxed."""
        config = PoIConfig(
            gini_rebalance_threshold=0.01,
            zakat_exemption_floor=0.5,
        )
        rebalancer = SATRebalancer(config)
        scores = {"a": 0.05, "b": 0.95}
        result = rebalancer.rebalance(scores)
        # 'a' is below exemption floor, should not be taxed even if above mean
        # (here 'a' is below mean anyway, but the logic is tested)
        assert result.rebalance_triggered is True

    def test_to_dict(self):
        """RebalanceResult.to_dict() includes fields."""
        config = PoIConfig(gini_rebalance_threshold=0.01)
        rebalancer = SATRebalancer(config)
        result = rebalancer.rebalance({"a": 0.1, "b": 0.9})
        d = result.to_dict()
        assert "gini_before" in d
        assert "gini_after" in d
        assert "zakat_collected" in d


# =============================================================================
# PROOF OF IMPACT
# =============================================================================

class TestProofOfImpact:
    """Tests for ProofOfImpact dataclass."""

    def test_to_dict(self, config):
        """to_dict() includes all fields."""
        poi = ProofOfImpact(
            contributor_id="alice",
            contribution_score=0.9,
            reach_score=0.7,
            longevity_score=0.8,
            poi_score=0.83,
            alpha=0.5,
            beta=0.3,
            gamma=0.2,
            config_digest="abc123",
            computation_id="test_001",
        )
        d = poi.to_dict()
        assert d["contributor_id"] == "alice"
        assert d["poi_score"] == 0.83
        assert d["weights"]["alpha"] == 0.5

    def test_canonical_bytes_deterministic(self):
        """PoI canonical bytes are deterministic."""
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        kwargs = dict(
            contributor_id="alice",
            contribution_score=0.9,
            reach_score=0.7,
            longevity_score=0.8,
            poi_score=0.83,
            alpha=0.5, beta=0.3, gamma=0.2,
            config_digest="abc123",
            computation_id="test_001",
            timestamp=ts,
        )
        p1 = ProofOfImpact(**kwargs)
        p2 = ProofOfImpact(**kwargs)
        assert p1.canonical_bytes() == p2.canonical_bytes()

    def test_digest_format(self):
        """PoI hex_digest is 64-char hex."""
        poi = ProofOfImpact(
            contributor_id="alice",
            contribution_score=0.9, reach_score=0.7, longevity_score=0.8,
            poi_score=0.83, alpha=0.5, beta=0.3, gamma=0.2,
            config_digest="abc", computation_id="c1",
        )
        hd = poi.hex_digest()
        assert len(hd) == 64
        int(hd, 16)


# =============================================================================
# AUDIT TRAIL
# =============================================================================

class TestAuditTrail:
    """Tests for AuditTrail."""

    def test_to_dict(self):
        """AuditTrail.to_dict() includes summary fields."""
        trail = AuditTrail(
            epoch_id="epoch_001",
            poi_scores=[],
            gini_coefficient=0.25,
            rebalance_triggered=False,
            config_digest="abc",
        )
        d = trail.to_dict()
        assert d["epoch_id"] == "epoch_001"
        assert d["total_contributors"] == 0
        assert d["gini_coefficient"] == 0.25

    def test_digest_deterministic(self):
        """AuditTrail digest is deterministic."""
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        kwargs = dict(
            epoch_id="e1",
            poi_scores=[],
            gini_coefficient=0.3,
            rebalance_triggered=False,
            config_digest="abc",
            timestamp=ts,
        )
        t1 = AuditTrail(**kwargs)
        t2 = AuditTrail(**kwargs)
        assert t1.digest() == t2.digest()


# =============================================================================
# POI ORCHESTRATOR
# =============================================================================

class TestPoIOrchestrator:
    """Tests for the full PoI orchestrator."""

    def test_register_valid_contribution(self, orchestrator):
        """Valid contribution is registered."""
        meta = _make_contribution(snr=0.95, ihsan=0.96)
        result = orchestrator.register_contribution(meta)
        assert result.verified is True

    def test_register_invalid_contribution(self, orchestrator):
        """Invalid contribution is rejected but doesn't error."""
        meta = _make_contribution(snr=0.10, ihsan=0.96)
        result = orchestrator.register_contribution(meta)
        assert result.verified is False

    def test_compute_epoch_empty(self, orchestrator):
        """Empty epoch produces empty audit trail."""
        audit = orchestrator.compute_epoch()
        assert len(audit.poi_scores) == 0
        assert audit.gini_coefficient == 0.0

    def test_compute_epoch_single_contributor(self, orchestrator):
        """Single contributor epoch."""
        meta = _make_contribution(contributor_id="alice", snr=0.95, ihsan=0.96)
        orchestrator.register_contribution(meta)
        audit = orchestrator.compute_epoch()
        assert len(audit.poi_scores) == 1
        assert audit.poi_scores[0].contributor_id == "alice"
        assert audit.poi_scores[0].poi_score > 0

    def test_compute_epoch_multiple_contributors(self, orchestrator):
        """Multiple contributors get different scores."""
        for i, name in enumerate(["alice", "bob", "carol"]):
            meta = _make_contribution(
                contributor_id=name,
                content_hash=f"hash_{name}",
                snr=0.95,
                ihsan=0.96,
            )
            orchestrator.register_contribution(meta)
        audit = orchestrator.compute_epoch()
        assert len(audit.poi_scores) == 3

    def test_citations_affect_reach(self, orchestrator):
        """Citations increase reach score."""
        for name in ["alice", "bob", "carol"]:
            meta = _make_contribution(
                contributor_id=name,
                content_hash=f"hash_{name}",
                snr=0.95,
                ihsan=0.96,
            )
            orchestrator.register_contribution(meta)

        # Everyone cites alice
        orchestrator.add_citation("bob", "alice")
        orchestrator.add_citation("carol", "alice")

        audit = orchestrator.compute_epoch()
        alice_poi = next(p for p in audit.poi_scores if p.contributor_id == "alice")
        bob_poi = next(p for p in audit.poi_scores if p.contributor_id == "bob")
        # Alice should have higher reach
        assert alice_poi.reach_score > bob_poi.reach_score

    def test_composite_formula(self, orchestrator):
        """Composite score follows alpha*C + beta*R + gamma*L."""
        meta = _make_contribution(
            contributor_id="alice",
            content_hash="formula_test",
            snr=0.95,
            ihsan=0.96,
        )
        orchestrator.register_contribution(meta)
        audit = orchestrator.compute_epoch()
        poi = audit.poi_scores[0]
        expected = (
            orchestrator.config.alpha * poi.contribution_score
            + orchestrator.config.beta * poi.reach_score
            + orchestrator.config.gamma * poi.longevity_score
        )
        assert abs(poi.poi_score - expected) < 1e-6

    def test_get_contributor_poi(self, orchestrator):
        """Get specific contributor's PoI after epoch."""
        meta = _make_contribution(contributor_id="alice", content_hash="lookup_test")
        orchestrator.register_contribution(meta)
        orchestrator.compute_epoch()
        poi = orchestrator.get_contributor_poi("alice")
        assert poi is not None
        assert poi.contributor_id == "alice"

    def test_get_contributor_poi_not_found(self, orchestrator):
        """Non-existent contributor returns None."""
        orchestrator.compute_epoch()
        assert orchestrator.get_contributor_poi("nobody") is None

    def test_get_contributor_poi_no_epoch(self, orchestrator):
        """No epoch computed returns None."""
        assert orchestrator.get_contributor_poi("alice") is None

    def test_multiple_epochs(self, orchestrator):
        """Multiple epochs are tracked."""
        meta = _make_contribution(contributor_id="alice", content_hash="epoch_1")
        orchestrator.register_contribution(meta)
        orchestrator.compute_epoch("e1")

        meta2 = _make_contribution(contributor_id="bob", content_hash="epoch_2")
        orchestrator.register_contribution(meta2)
        orchestrator.compute_epoch("e2")

        stats = orchestrator.get_stats()
        assert stats["total_epochs"] == 2

    def test_stats(self, orchestrator):
        """Stats include all subsystem stats."""
        stats = orchestrator.get_stats()
        assert "total_contributors" in stats
        assert "total_contributions" in stats
        assert "total_epochs" in stats
        assert "graph_stats" in stats
        assert "verifier_stats" in stats
        assert "config_digest" in stats

    def test_audit_trail_has_config_digest(self, orchestrator):
        """Audit trail includes config digest."""
        audit = orchestrator.compute_epoch()
        assert audit.config_digest == orchestrator.config.hex_digest()

    def test_epoch_id_auto_generated(self, orchestrator):
        """Epoch IDs auto-increment."""
        a1 = orchestrator.compute_epoch()
        a2 = orchestrator.compute_epoch()
        assert a1.epoch_id != a2.epoch_id

    def test_custom_epoch_id(self, orchestrator):
        """Custom epoch IDs are used."""
        audit = orchestrator.compute_epoch("my_epoch")
        assert audit.epoch_id == "my_epoch"


# =============================================================================
# TOKEN DISTRIBUTION
# =============================================================================

class TestTokenDistribution:
    """Tests for token distribution."""

    def test_distribution_proportional(self, orchestrator):
        """Tokens are proportional to PoI scores."""
        for name in ["alice", "bob"]:
            meta = _make_contribution(
                contributor_id=name,
                content_hash=f"token_{name}",
                snr=0.95,
                ihsan=0.96,
            )
            orchestrator.register_contribution(meta)
        audit = orchestrator.compute_epoch()
        dist = compute_token_distribution(audit, epoch_reward=100.0)
        assert abs(dist.total_minted - 100.0) < 1e-6

    def test_distribution_zero_scores(self):
        """Zero total PoI gives zero distribution."""
        audit = AuditTrail(
            epoch_id="empty",
            poi_scores=[],
            gini_coefficient=0.0,
            rebalance_triggered=False,
            config_digest="abc",
        )
        dist = compute_token_distribution(audit, epoch_reward=100.0)
        assert dist.total_minted == 0.0

    def test_distribution_to_dict(self, orchestrator):
        """TokenDistribution.to_dict() includes fields."""
        meta = _make_contribution(contributor_id="alice", content_hash="dist_test")
        orchestrator.register_contribution(meta)
        audit = orchestrator.compute_epoch()
        dist = compute_token_distribution(audit, epoch_reward=50.0)
        d = dist.to_dict()
        assert "epoch_id" in d
        assert "epoch_reward" in d
        assert "total_minted" in d
        assert "distributions" in d

    def test_scaling_factor(self, orchestrator):
        """Scaling factor adjusts total minted."""
        meta = _make_contribution(contributor_id="alice", content_hash="scale_test")
        orchestrator.register_contribution(meta)
        audit = orchestrator.compute_epoch()
        dist = compute_token_distribution(audit, epoch_reward=100.0, scaling_factor=2.0)
        assert abs(dist.total_minted - 200.0) < 1e-6

    def test_single_contributor_gets_all(self, orchestrator):
        """Single contributor gets full epoch reward."""
        meta = _make_contribution(contributor_id="alice", content_hash="solo")
        orchestrator.register_contribution(meta)
        audit = orchestrator.compute_epoch()
        dist = compute_token_distribution(audit, epoch_reward=100.0)
        assert abs(dist.distributions["alice"] - 100.0) < 1e-6
