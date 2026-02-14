"""
Tests for the BIZRA Unified Constitutional Engine (Omega Point).

Tests cover all four gap solutions:
- GAP-C1: IhsanProjector (8D -> 3D in O(1))
- GAP-C2: AdlInvariant (Protocol-level rejection gate)
- GAP-C3: ByzantineConsensus (f < n/3 proven)
- GAP-C4: TreasuryController (Graceful degradation)

Standing on Giants: Shannon, Lamport, Landauer, Al-Ghazali
"""

import pytest
import numpy as np
from datetime import datetime, timezone

from core.constitutional import (
    # Core Types
    IhsanVector,
    NTUState,
    # GAP-C1
    IhsanProjector,
    # GAP-C2
    AdlInvariant,
    AdlInvariantResult,
    AdlViolationType,
    AdlViolationError,
    # GAP-C3
    ByzantineConsensus,
    ByzantineVoteType,
    ConsensusState,
    # GAP-C4
    TreasuryMode,
    TreasuryController,
    TREASURY_MODES,
    # Unified
    ConstitutionalEngine,
    create_constitutional_engine,
    # Constants
    ADL_GINI_THRESHOLD,
    BFT_QUORUM_FRACTION,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def high_ihsan_vector():
    """Create a high-quality Ihsan vector that passes threshold (>= 0.95)."""
    return IhsanVector(
        correctness=0.99,
        safety=0.99,
        user_benefit=0.97,
        efficiency=0.95,
        auditability=0.96,
        anti_centralization=0.92,
        robustness=0.93,
        adl_fairness=0.98,
    )


@pytest.fixture
def low_ihsan_vector():
    """Create a low-quality Ihsan vector that fails threshold."""
    return IhsanVector(
        correctness=0.70,
        safety=0.75,
        user_benefit=0.65,
        efficiency=0.60,
        auditability=0.70,
        anti_centralization=0.50,
        robustness=0.55,
        adl_fairness=0.60,
    )


@pytest.fixture
def fair_distribution():
    """Create a fair resource distribution (low Gini)."""
    return {
        "node_001": 100.0,
        "node_002": 120.0,
        "node_003": 110.0,
        "node_004": 90.0,
        "node_005": 115.0,
    }


@pytest.fixture
def unfair_distribution():
    """Create an unfair resource distribution (high Gini)."""
    return {
        "node_001": 900.0,  # Monopoly
        "node_002": 50.0,
        "node_003": 25.0,
        "node_004": 15.0,
        "node_005": 10.0,
    }


@pytest.fixture
def mock_keys():
    """Create mock cryptographic keys for testing."""
    from core.pci.crypto import generate_keypair
    private_key_hex, public_key_hex = generate_keypair()
    return {
        "private_key": private_key_hex,
        "public_key": public_key_hex,
    }


# =============================================================================
# GAP-C1: IHSAN PROJECTOR TESTS
# =============================================================================

class TestIhsanProjector:
    """Tests for O(1) Ihsan to NTU projection."""

    def test_projection_produces_valid_ntu(self, high_ihsan_vector):
        """Projection should produce NTU state with valid ranges."""
        projector = IhsanProjector()
        ntu = projector.project(high_ihsan_vector)

        assert 0.0 <= ntu.belief <= 1.0
        assert 0.0 <= ntu.entropy <= 1.0
        assert 0.0 <= ntu.lambda_lr <= 1.0

    def test_high_ihsan_produces_stable_ntu(self, high_ihsan_vector):
        """High Ihsan should produce stable NTU state."""
        projector = IhsanProjector()
        ntu = projector.project(high_ihsan_vector)

        # High quality input should yield high belief
        assert ntu.belief > 0.6
        # Stable state has low entropy
        assert ntu.is_stable(threshold=0.6)

    def test_low_ihsan_produces_unstable_ntu(self, low_ihsan_vector):
        """Low Ihsan should produce less stable NTU state."""
        projector = IhsanProjector()
        ntu = projector.project(low_ihsan_vector)

        # Low quality input should yield lower belief
        assert ntu.belief < 0.8

    def test_projection_is_deterministic(self, high_ihsan_vector):
        """Same input should always produce same output."""
        projector = IhsanProjector()

        ntu1 = projector.project(high_ihsan_vector)
        ntu2 = projector.project(high_ihsan_vector)

        assert ntu1.belief == ntu2.belief
        assert ntu1.entropy == ntu2.entropy
        assert ntu1.lambda_lr == ntu2.lambda_lr

    def test_batch_projection(self, high_ihsan_vector, low_ihsan_vector):
        """Batch projection should process multiple vectors."""
        projector = IhsanProjector()
        batch = [high_ihsan_vector, low_ihsan_vector, high_ihsan_vector]

        results = projector.project_batch(batch)

        assert len(results) == 3
        assert results[0].belief > results[1].belief  # High > Low

    def test_projection_matrix_constraints(self):
        """Projection matrix rows should sum to 1.0."""
        projector = IhsanProjector()
        row_sums = projector.projection_matrix.sum(axis=1)

        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)

    def test_ihsan_weighted_score(self, high_ihsan_vector):
        """Weighted score should use constitutional weights."""
        score = high_ihsan_vector.weighted_score()

        # With our test values, should be > 0.95
        assert score > 0.9
        assert score <= 1.0


# =============================================================================
# GAP-C2: ADL INVARIANT TESTS
# =============================================================================

class TestAdlInvariant:
    """Tests for protocol-level Adl (Justice) enforcement."""

    def test_fair_distribution_passes(self, fair_distribution):
        """Fair distribution should pass Adl invariant."""
        adl = AdlInvariant()
        result = adl.check(fair_distribution)

        assert result.passed
        assert result.gini < ADL_GINI_THRESHOLD
        assert len(result.violations) == 0

    def test_unfair_distribution_fails(self, unfair_distribution):
        """Unfair distribution should fail Adl invariant."""
        adl = AdlInvariant()
        result = adl.check(unfair_distribution)

        assert not result.passed
        assert result.gini > ADL_GINI_THRESHOLD
        assert len(result.violations) > 0

    def test_concentration_detected(self, unfair_distribution):
        """Should detect when one entity controls >50%."""
        adl = AdlInvariant()
        result = adl.check(unfair_distribution)

        # Find concentration violation
        concentration_violations = [
            v for v in result.violations
            if v.violation_type == AdlViolationType.CONCENTRATION_DETECTED
        ]
        assert len(concentration_violations) > 0
        assert concentration_violations[0].violator_id == "node_001"

    def test_preemptive_check_blocks_monopoly(self, fair_distribution):
        """Preemptive check should block changes that would violate Adl."""
        adl = AdlInvariant()

        # Proposed change that would create monopoly
        proposed_change = {
            "node_001": 1000.0,  # Give node_001 massive resources
        }

        result = adl.check(fair_distribution, proposed_change)

        # Current distribution is fair, but proposed change is not
        assert not result.passed
        monopoly_violations = [
            v for v in result.violations
            if v.violation_type == AdlViolationType.MONOPOLY_ATTEMPT
        ]
        assert len(monopoly_violations) > 0

    def test_gini_calculation_perfect_equality(self):
        """Perfect equality should have Gini = 0."""
        adl = AdlInvariant()
        equal_dist = {
            "a": 100.0,
            "b": 100.0,
            "c": 100.0,
            "d": 100.0,
        }

        gini = adl.compute_gini(equal_dist)
        assert gini < 0.01

    def test_gini_calculation_perfect_inequality(self):
        """Near-perfect inequality should have high Gini."""
        adl = AdlInvariant()
        unequal_dist = {
            "monopolist": 1000.0,
            "pauper1": 1.0,
            "pauper2": 1.0,
            "pauper3": 1.0,
        }

        gini = adl.compute_gini(unequal_dist)
        assert gini > 0.7

    def test_must_pass_raises_on_violation(self, unfair_distribution):
        """must_pass should raise AdlViolationError on failure."""
        adl = AdlInvariant()

        with pytest.raises(AdlViolationError) as exc_info:
            adl.must_pass(unfair_distribution)

        assert len(exc_info.value.violations) > 0

    def test_redistribution_suggestion(self, unfair_distribution):
        """Should suggest redistribution to achieve target Gini."""
        adl = AdlInvariant()
        changes = adl.suggest_redistribution(unfair_distribution, target_gini=0.35)

        # Should suggest taking from node_001 (the monopolist)
        assert "node_001" in changes
        assert changes["node_001"] < 0  # Negative = take


# =============================================================================
# GAP-C3: BYZANTINE CONSENSUS TESTS
# =============================================================================

class TestByzantineConsensus:
    """Tests for Byzantine fault tolerant consensus."""

    def test_quorum_calculation_7_nodes(self, mock_keys):
        """7 nodes: f=2, quorum=5."""
        consensus = ByzantineConsensus(
            node_id="node_001",
            private_key=mock_keys["private_key"],
            public_key=mock_keys["public_key"],
            total_nodes=7,
        )

        assert consensus.fault_tolerance == 2
        assert consensus.quorum_size == 5

    def test_quorum_calculation_4_nodes(self, mock_keys):
        """4 nodes: f=1, quorum=3."""
        consensus = ByzantineConsensus(
            node_id="node_001",
            private_key=mock_keys["private_key"],
            public_key=mock_keys["public_key"],
            total_nodes=4,
        )

        assert consensus.fault_tolerance == 1
        assert consensus.quorum_size == 3

    def test_bft_property_verification(self, mock_keys):
        """BFT property n >= 3f + 1 should be verified."""
        consensus = ByzantineConsensus(
            node_id="node_001",
            private_key=mock_keys["private_key"],
            public_key=mock_keys["public_key"],
            total_nodes=7,
        )

        assert consensus.verify_bft_property()

    def test_proposal_creation_with_high_ihsan(self, mock_keys, high_ihsan_vector):
        """Proposals should be created with sufficient Ihsan."""
        consensus = ByzantineConsensus(
            node_id="node_001",
            private_key=mock_keys["private_key"],
            public_key=mock_keys["public_key"],
            total_nodes=7,
        )

        ihsan_score = high_ihsan_vector.weighted_score()
        proposal = consensus.create_proposal(b"test_value", ihsan_score)

        assert proposal is not None
        assert proposal.proposer_id == "node_001"
        assert proposal.state == ConsensusState.PREPARING

    def test_proposal_rejected_with_low_ihsan(self, mock_keys, low_ihsan_vector):
        """Proposals should be rejected with insufficient Ihsan."""
        consensus = ByzantineConsensus(
            node_id="node_001",
            private_key=mock_keys["private_key"],
            public_key=mock_keys["public_key"],
            total_nodes=7,
        )

        ihsan_score = low_ihsan_vector.weighted_score()
        proposal = consensus.create_proposal(b"test_value", ihsan_score)

        assert proposal is None

    def test_vote_signing(self, mock_keys, high_ihsan_vector):
        """Votes should be signed with Ed25519."""
        consensus = ByzantineConsensus(
            node_id="node_001",
            private_key=mock_keys["private_key"],
            public_key=mock_keys["public_key"],
            total_nodes=7,
        )

        ihsan_score = high_ihsan_vector.weighted_score()
        proposal = consensus.create_proposal(b"test_value", ihsan_score)

        vote = consensus.sign_vote(
            ByzantineVoteType.PREPARE,
            proposal.proposal_id,
            proposal.value,
        )

        assert vote is not None
        assert vote.voter_id == "node_001"
        assert vote.signature != ""

    def test_peer_registration_required(self, mock_keys, high_ihsan_vector):
        """Votes from unregistered peers should be rejected."""
        consensus = ByzantineConsensus(
            node_id="node_001",
            private_key=mock_keys["private_key"],
            public_key=mock_keys["public_key"],
            total_nodes=7,
        )

        ihsan_score = high_ihsan_vector.weighted_score()
        proposal = consensus.create_proposal(b"test_value", ihsan_score)

        # Create vote from unregistered peer
        vote = consensus.sign_vote(
            ByzantineVoteType.PREPARE,
            proposal.proposal_id,
            proposal.value,
        )
        vote.voter_id = "unknown_peer"  # Forge voter ID

        # Should fail verification
        assert not consensus.verify_vote(vote)


# =============================================================================
# GAP-C4: TREASURY CONTROLLER TESTS
# =============================================================================

class TestTreasuryController:
    """Tests for treasury with graceful degradation."""

    def test_initial_mode_is_ethical(self):
        """Treasury should start in Ethical mode."""
        treasury = TreasuryController(initial_treasury=1000.0)

        assert treasury.mode == TreasuryMode.ETHICAL

    def test_deposit_increases_balance(self):
        """Deposits should increase treasury balance."""
        treasury = TreasuryController(initial_treasury=100.0)

        new_balance = treasury.deposit(500.0, "test")

        assert new_balance == 600.0
        assert treasury.treasury == 600.0

    def test_withdraw_decreases_balance(self):
        """Withdrawals should decrease treasury balance."""
        treasury = TreasuryController(initial_treasury=1000.0)

        amount = treasury.withdraw(300.0, "test")

        assert amount == 300.0
        assert treasury.treasury == 700.0

    def test_withdraw_fails_on_insufficient_funds(self):
        """Withdrawal should fail if insufficient funds."""
        treasury = TreasuryController(initial_treasury=100.0)

        amount = treasury.withdraw(500.0, "test")

        assert amount is None
        assert treasury.treasury == 100.0

    def test_mode_degrades_to_hibernation(self):
        """Treasury should degrade to Hibernation when low."""
        treasury = TreasuryController(initial_treasury=1000.0)
        treasury._target_treasury = 1000.0

        # Withdraw to 40% of target
        treasury.withdraw(600.0, "trigger_hibernation")

        assert treasury.mode == TreasuryMode.HIBERNATION

    def test_mode_degrades_to_emergency(self):
        """Treasury should degrade to Emergency when critical."""
        treasury = TreasuryController(initial_treasury=1000.0)
        treasury._target_treasury = 1000.0

        # Withdraw to <20% of target
        treasury.withdraw(850.0, "trigger_emergency")

        assert treasury.mode == TreasuryMode.EMERGENCY

    def test_mode_recovers_to_ethical(self):
        """Treasury should recover to Ethical when healthy."""
        treasury = TreasuryController(initial_treasury=100.0)
        treasury._target_treasury = 1000.0
        treasury.set_mode(TreasuryMode.HIBERNATION, force=True)

        # Deposit to >60% of target
        treasury.deposit(700.0, "recovery")

        assert treasury.mode == TreasuryMode.ETHICAL

    def test_effective_thresholds_vary_by_mode(self):
        """Different modes should have different thresholds."""
        treasury = TreasuryController(initial_treasury=1000.0)

        ethical_thresholds = treasury.get_effective_thresholds()

        treasury.set_mode(TreasuryMode.HIBERNATION, force=True)
        hibernation_thresholds = treasury.get_effective_thresholds()

        treasury.set_mode(TreasuryMode.EMERGENCY, force=True)
        emergency_thresholds = treasury.get_effective_thresholds()

        # Ethical has strictest thresholds
        assert ethical_thresholds["gini_threshold"] < hibernation_thresholds["gini_threshold"]
        assert hibernation_thresholds["gini_threshold"] < emergency_thresholds["gini_threshold"]

    def test_operation_execution_check(self):
        """Should check if operations can be executed."""
        treasury = TreasuryController(initial_treasury=100.0)

        can_cheap, _ = treasury.can_execute_operation(10.0)
        can_expensive, reason = treasury.can_execute_operation(500.0)

        assert can_cheap
        assert not can_expensive
        assert "Insufficient" in reason


# =============================================================================
# UNIFIED CONSTITUTIONAL ENGINE TESTS
# =============================================================================

class TestConstitutionalEngine:
    """Tests for the unified Constitutional Engine."""

    def test_engine_creation(self, mock_keys):
        """Engine should be created with valid configuration."""
        engine = create_constitutional_engine(
            node_id="test_node",
            private_key=mock_keys["private_key"],
            public_key=mock_keys["public_key"],
            total_nodes=7,
            initial_treasury=1000.0,
        )

        assert engine.node_id == "test_node"
        assert engine.consensus.total_nodes == 7
        assert engine.treasury.treasury == 1000.0

    def test_evaluate_action_all_pass(
        self, mock_keys, high_ihsan_vector, fair_distribution
    ):
        """Action should be permitted when all constraints pass."""
        engine = create_constitutional_engine(
            node_id="test_node",
            private_key=mock_keys["private_key"],
            public_key=mock_keys["public_key"],
            total_nodes=7,
            initial_treasury=1000.0,
        )

        permitted, details = engine.evaluate_action(
            ihsan_vector=high_ihsan_vector,
            distribution=fair_distribution,
            operation_cost=50.0,
        )

        assert permitted
        assert details["ihsan"]["passed"]
        assert details["adl"]["passed"]
        assert details["treasury"]["can_execute"]

    def test_evaluate_action_ihsan_fails(
        self, mock_keys, low_ihsan_vector, fair_distribution
    ):
        """Action should be rejected when Ihsan fails."""
        engine = create_constitutional_engine(
            node_id="test_node",
            private_key=mock_keys["private_key"],
            public_key=mock_keys["public_key"],
            total_nodes=7,
            initial_treasury=1000.0,
        )

        permitted, details = engine.evaluate_action(
            ihsan_vector=low_ihsan_vector,
            distribution=fair_distribution,
            operation_cost=50.0,
        )

        assert not permitted
        assert not details["ihsan"]["passed"]
        assert "Ihsan" in details["rejection_reason"]

    def test_evaluate_action_adl_fails(
        self, mock_keys, high_ihsan_vector, unfair_distribution
    ):
        """Action should be rejected when Adl fails."""
        engine = create_constitutional_engine(
            node_id="test_node",
            private_key=mock_keys["private_key"],
            public_key=mock_keys["public_key"],
            total_nodes=7,
            initial_treasury=1000.0,
        )

        permitted, details = engine.evaluate_action(
            ihsan_vector=high_ihsan_vector,
            distribution=unfair_distribution,
            operation_cost=50.0,
        )

        assert not permitted
        assert not details["adl"]["passed"]
        assert "Adl" in details["rejection_reason"]

    def test_evaluate_action_treasury_fails(
        self, mock_keys, high_ihsan_vector, fair_distribution
    ):
        """Action should be rejected when treasury insufficient."""
        engine = create_constitutional_engine(
            node_id="test_node",
            private_key=mock_keys["private_key"],
            public_key=mock_keys["public_key"],
            total_nodes=7,
            initial_treasury=10.0,  # Low treasury
        )

        permitted, details = engine.evaluate_action(
            ihsan_vector=high_ihsan_vector,
            distribution=fair_distribution,
            operation_cost=500.0,  # High cost
        )

        assert not permitted
        assert not details["treasury"]["can_execute"]
        assert "Treasury" in details["rejection_reason"]

    def test_engine_status(self, mock_keys):
        """Engine should provide comprehensive status."""
        engine = create_constitutional_engine(
            node_id="test_node",
            private_key=mock_keys["private_key"],
            public_key=mock_keys["public_key"],
            total_nodes=7,
            initial_treasury=1000.0,
        )

        status = engine.get_status()

        assert status["node_id"] == "test_node"
        assert "components" in status
        assert "projector" in status["components"]
        assert "adl_invariant" in status["components"]
        assert "consensus" in status["components"]
        assert "treasury" in status["components"]

    def test_mode_change_updates_adl_threshold(self, mock_keys):
        """Treasury mode change should update Adl threshold."""
        engine = create_constitutional_engine(
            node_id="test_node",
            private_key=mock_keys["private_key"],
            public_key=mock_keys["public_key"],
            total_nodes=7,
            initial_treasury=1000.0,
        )

        initial_threshold = engine.adl_invariant.gini_threshold

        # Trigger mode change
        engine.treasury.set_mode(TreasuryMode.HIBERNATION, force=True)

        # Manually trigger the callback (in production this is automatic)
        engine._on_treasury_mode_change(TreasuryMode.ETHICAL, TreasuryMode.HIBERNATION)

        new_threshold = engine.adl_invariant.gini_threshold

        assert new_threshold > initial_threshold


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for the complete Constitutional Engine."""

    def test_full_consensus_flow(self, mock_keys, high_ihsan_vector, fair_distribution):
        """Test complete flow from evaluation to consensus."""
        engine = create_constitutional_engine(
            node_id="leader",
            private_key=mock_keys["private_key"],
            public_key=mock_keys["public_key"],
            total_nodes=3,  # Small for testing
            initial_treasury=1000.0,
        )

        # Execute with consensus
        success, proposal_id, details = engine.execute_with_consensus(
            value=b"test_transaction",
            ihsan_vector=high_ihsan_vector,
            distribution=fair_distribution,
            operation_cost=10.0,
        )

        assert success
        assert proposal_id is not None
        assert "consensus" in details

    def test_graceful_degradation_flow(self, mock_keys, high_ihsan_vector, fair_distribution):
        """Test graceful degradation under resource pressure."""
        engine = create_constitutional_engine(
            node_id="test_node",
            private_key=mock_keys["private_key"],
            public_key=mock_keys["public_key"],
            total_nodes=7,
            initial_treasury=1000.0,
        )

        # Start in Ethical mode
        assert engine.treasury.mode == TreasuryMode.ETHICAL

        # Drain treasury
        engine.treasury.withdraw(600.0, "pressure_test")

        # Should now be in Hibernation
        assert engine.treasury.mode == TreasuryMode.HIBERNATION

        # Action with relaxed thresholds should still work
        permitted, details = engine.evaluate_action(
            ihsan_vector=high_ihsan_vector,
            distribution=fair_distribution,
            operation_cost=10.0,
        )

        assert permitted
        assert details["treasury_mode"] == "HIBERNATION"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
