"""Tests for Pool-Mediated Consensus and Propagation.

Phase 61, Step 3 -- 11 TDD anchors from the spec.
Validates Amended Theorem 2.4 (BFT) and Amended Lemma 3.3 (Propagation).
"""

from __future__ import annotations

import math

from core.federation.pool_consensus import (
    ConsensusParams,
    ConsensusRound,
    PropagationParams,
    attack_cost_at_scale,
    propagation_time_table,
)

# ---------------------------------------------------------------------------
# Amended Theorem 2.4 -- BFT Consensus Parameters
# ---------------------------------------------------------------------------


class TestConsensusParams:
    """BFT parameter derivation from validator count."""

    def test_bft_params_100_validators(self) -> None:
        """100 validators -> max_byzantine=33, quorum=67."""
        params = ConsensusParams.from_validator_count(100)
        assert params.total_validators == 100
        assert params.max_byzantine == 33
        assert params.quorum_size == 67

    def test_bft_params_1000_validators(self) -> None:
        """1000 validators -> max_byzantine=333, quorum=667."""
        params = ConsensusParams.from_validator_count(1000)
        assert params.total_validators == 1000
        assert params.max_byzantine == 333
        assert params.quorum_size == 667

    def test_bft_params_single_validator(self) -> None:
        """Single validator -> quorum=1, max_byzantine=0."""
        params = ConsensusParams.from_validator_count(1)
        assert params.quorum_size == 1
        assert params.max_byzantine == 0

    def test_equivocation_impossible(self) -> None:
        """Equivocation is always False under Pool mediation (Axiom 1.6)."""
        for n in [1, 7, 100, 100_000]:
            params = ConsensusParams.from_validator_count(n)
            assert params.equivocation_possible is False

    def test_safety_margin_scales(self) -> None:
        """Safety margin approaches 1/3 as validator count grows."""
        params_small = ConsensusParams.from_validator_count(7)
        params_large = ConsensusParams.from_validator_count(100_000)
        # Both should be close to 1/3
        assert 0.2 < params_small.safety_margin < 0.34
        assert 0.33 < params_large.safety_margin < 0.34

    def test_is_safe_within_tolerance(self) -> None:
        """is_safe returns True when observed Byzantine <= max."""
        params = ConsensusParams.from_validator_count(100)
        assert params.is_safe(0) is True
        assert params.is_safe(33) is True
        assert params.is_safe(34) is False


# ---------------------------------------------------------------------------
# Consensus Round
# ---------------------------------------------------------------------------


class TestConsensusRound:
    """Quorum-based voting logic."""

    def test_consensus_round_reaches_quorum(self) -> None:
        """Enough approval votes -> quorum reached and finalized."""
        params = ConsensusParams.from_validator_count(7)
        # 7 validators -> f=2, quorum=5
        rnd = ConsensusRound(evidence_hash="abc123", params=params)
        for i in range(5):
            rnd.add_vote(f"validator_{i}", True)
        assert rnd.is_quorum_reached is True
        assert rnd.finalized is True
        assert rnd.honest_votes == 5
        assert rnd.total_votes == 5

    def test_consensus_round_rejects_insufficient(self) -> None:
        """Fewer than quorum approval votes -> not finalized."""
        params = ConsensusParams.from_validator_count(7)
        rnd = ConsensusRound(evidence_hash="abc123", params=params)
        # Submit 4 approvals (quorum is 5)
        for i in range(4):
            rnd.add_vote(f"validator_{i}", True)
        assert rnd.is_quorum_reached is False
        assert rnd.finalized is False

    def test_consensus_round_no_double_vote(self) -> None:
        """Second vote from the same validator is silently ignored."""
        params = ConsensusParams.from_validator_count(7)
        rnd = ConsensusRound(evidence_hash="abc123", params=params)
        rnd.add_vote("validator_0", True)
        rnd.add_vote("validator_0", False)  # Should be ignored
        assert rnd.total_votes == 1
        assert rnd.votes["validator_0"] is True

    def test_no_votes_after_finalization(self) -> None:
        """Once finalized, no further votes are accepted."""
        params = ConsensusParams.from_validator_count(4)
        # 4 validators -> f=1, quorum=3
        rnd = ConsensusRound(evidence_hash="hash", params=params)
        for i in range(3):
            rnd.add_vote(f"v_{i}", True)
        assert rnd.finalized is True
        rnd.add_vote("late_voter", True)
        assert rnd.total_votes == 3  # Late vote not counted


# ---------------------------------------------------------------------------
# Attack Cost
# ---------------------------------------------------------------------------


class TestAttackCost:
    """Byzantine attack cost scaling."""

    def test_attack_cost_scales_linearly(self) -> None:
        """Cost of attack grows linearly with network size."""
        cost_1k = attack_cost_at_scale(1_000)
        cost_10k = attack_cost_at_scale(10_000)
        # 10x more nodes should yield roughly 10x more Byzantine tolerance
        ratio = cost_10k["max_byzantine"] / cost_1k["max_byzantine"]
        assert ratio > 5  # Conservative: at least 5x growth for 10x nodes

    def test_attack_cost_returns_expected_keys(self) -> None:
        """Returned dict has the documented keys."""
        result = attack_cost_at_scale(1_000)
        assert "validators" in result
        assert "max_byzantine" in result
        assert "attack_cost_description" in result


# ---------------------------------------------------------------------------
# Amended Lemma 3.3 -- Propagation
# ---------------------------------------------------------------------------


class TestPropagationParams:
    """Pool CacheCoordinator propagation parameters."""

    def test_propagation_fanout_log_n(self) -> None:
        """Fanout is O(log N) -- specifically ceil(log2(N))."""
        params = PropagationParams.for_network(1_000)
        expected_fanout = math.ceil(math.log2(1_000))  # 10
        assert params.fanout == expected_fanout

    def test_propagation_rounds_small(self) -> None:
        """Rounds <= 6 for 1M nodes."""
        params = PropagationParams.for_network(1_000_000)
        assert params.rounds <= 6
        # Fanout should be ceil(log2(1_000_000)) = 20
        assert params.fanout == 20

    def test_propagation_coverage_100_percent(self) -> None:
        """Deterministic tree guarantees 100% coverage."""
        for n in [10, 1_000, 1_000_000]:
            params = PropagationParams.for_network(n)
            assert params.coverage_guarantee == 1.0

    def test_propagation_single_node(self) -> None:
        """Edge case: single node needs zero rounds."""
        params = PropagationParams.for_network(1)
        assert params.rounds == 0
        assert params.fanout == 1
        assert params.total_messages == 0

    def test_propagation_message_complexity_scales(self) -> None:
        """Total messages = N * fanout for each scale."""
        for n in [100, 1_000, 10_000]:
            params = PropagationParams.for_network(n)
            assert params.total_messages == n * params.fanout

    def test_propagation_time_table(self) -> None:
        """propagation_time_table returns correct entries."""
        table = propagation_time_table([100, 1_000, 1_000_000])
        assert len(table) == 3
        assert table[0]["nodes"] == 100
        assert table[1]["nodes"] == 1_000
        assert table[2]["nodes"] == 1_000_000
        # All entries have required keys
        for entry in table:
            assert "fanout" in entry
            assert "rounds" in entry
            assert "total_messages" in entry
            assert "coverage_guarantee" in entry
