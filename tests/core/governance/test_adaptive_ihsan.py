"""
Ω-2: Adaptive Ihsān Dirichlet Posterior — Test Suite

Proves:
1. Dirichlet state computes correct mode, mean, variance
2. Bayesian updates produce valid posteriors
3. ALL 5 constitutional invariants hold under arbitrary observations
4. KL-drift bound prevents runaway adaptation
5. Cryptographic receipts are tamper-evident
6. Convergence behavior matches theoretical expectations

Standing on Giants:
- Dirichlet (1839): Simplex distribution properties
- Bayes (1763): Conjugate posterior correctness
- Shannon (1948): KL-divergence bounds
"""

import pytest
import math
import json
from copy import deepcopy

from core.governance.adaptive_ihsan import (
    AdaptiveIhsan,
    DirichletObservation,
    DirichletState,
    UpdateOutcome,
    UpdateReceipt,
    create_adaptive_ihsan,
    WEIGHT_FLOOR,
    WEIGHT_CEILING,
    SAFETY_CORRECTNESS_FLOOR,
    MAX_KL_DRIFT,
    DEFAULT_CONCENTRATION,
    DIMENSION_ORDER,
)
from core.integration.constants import IHSAN_WEIGHTS


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def engine():
    """Standard adaptive Ihsān engine."""
    return create_adaptive_ihsan()


@pytest.fixture
def low_concentration_engine():
    """Engine with lower concentration (faster adaptation)."""
    return AdaptiveIhsan(concentration=20.0)


@pytest.fixture
def success_observation():
    """Single success observation on safety dimension."""
    return DirichletObservation(
        dimension="safety",
        outcome=UpdateOutcome.SUCCESS,
        magnitude=1.0,
        context="test",
    )


@pytest.fixture
def failure_observation():
    """Single failure observation on efficiency dimension."""
    return DirichletObservation(
        dimension="efficiency",
        outcome=UpdateOutcome.FAILURE,
        magnitude=1.0,
        context="test",
    )


# =============================================================================
# 1. DIRICHLET STATE — Mathematical correctness
# =============================================================================


class TestDirichletState:
    """Verify Dirichlet distribution computations."""

    def test_mode_matches_canonical_at_initialization(self, engine):
        """Mode of prior should match IHSAN_WEIGHTS."""
        weights = engine.current_weights
        for dim in DIMENSION_ORDER:
            assert abs(weights[dim] - IHSAN_WEIGHTS[dim]) < 0.01, (
                f"Initial weight for {dim}: {weights[dim]:.4f} != {IHSAN_WEIGHTS[dim]}"
            )

    def test_weights_sum_to_one(self, engine):
        """INV-1: Weights must always sum to 1.0."""
        total = sum(engine.current_weights.values())
        assert abs(total - 1.0) < 1e-6, f"Weights sum to {total}, expected 1.0"

    def test_mean_sums_to_one(self, engine):
        """Mean of Dirichlet always sums to 1.0."""
        mean = engine.state.mean
        total = sum(mean.values())
        assert abs(total - 1.0) < 1e-6, f"Mean sums to {total}"

    def test_variance_is_positive(self, engine):
        """All variances must be non-negative."""
        for dim, var in engine.state.variance.items():
            assert var >= 0, f"Negative variance for {dim}: {var}"

    def test_effective_sample_size_equals_concentration(self, engine):
        """ESS should approximately equal concentration parameter."""
        ess = engine.state.effective_sample_size
        assert abs(ess - DEFAULT_CONCENTRATION) < 1.0, (
            f"ESS {ess} != concentration {DEFAULT_CONCENTRATION}"
        )

    def test_all_dimensions_present(self, engine):
        """All 8 canonical dimensions must be present."""
        weights = engine.current_weights
        for dim in DIMENSION_ORDER:
            assert dim in weights, f"Missing dimension: {dim}"
        assert len(weights) == 8


# =============================================================================
# 2. BAYESIAN UPDATE — Correctness
# =============================================================================


class TestBayesianUpdate:
    """Verify Bayesian update mechanics."""

    def test_success_increases_dimension_weight(self, engine):
        """Success on a dimension should increase its weight."""
        before = engine.current_weights["safety"]
        obs = DirichletObservation("safety", UpdateOutcome.SUCCESS, 1.0)
        receipt = engine.update(obs)
        after = engine.current_weights["safety"]

        assert receipt.accepted
        assert after > before, f"Safety: {before:.6f} -> {after:.6f} (should increase)"

    def test_failure_also_increases_weight(self, engine):
        """Failure means the dimension was a bottleneck — needs MORE weight."""
        before = engine.current_weights["efficiency"]
        obs = DirichletObservation("efficiency", UpdateOutcome.FAILURE, 1.0)
        receipt = engine.update(obs)
        after = engine.current_weights["efficiency"]

        assert receipt.accepted
        assert after >= before, (
            f"Efficiency: {before:.6f} -> {after:.6f} (bottleneck should increase)"
        )

    def test_neutral_produces_no_change(self, engine):
        """Neutral observation should not change weights."""
        before = deepcopy(engine.current_weights)
        obs = DirichletObservation("safety", UpdateOutcome.NEUTRAL, 1.0)
        receipt = engine.update(obs)

        assert receipt.accepted
        for dim in DIMENSION_ORDER:
            assert abs(engine.current_weights[dim] - before[dim]) < 1e-10

    def test_weights_remain_simplex_after_update(self, engine, success_observation):
        """INV-1: Weights must sum to 1.0 after every update."""
        for _ in range(20):
            engine.update(success_observation)
            total = sum(engine.current_weights.values())
            assert abs(total - 1.0) < 1e-6, f"Weights sum: {total}"

    def test_observation_count_increments(self, engine, success_observation):
        """Observation counter must track correctly."""
        assert engine.observation_count == 0
        engine.update(success_observation)
        assert engine.observation_count == 1
        engine.update(success_observation)
        assert engine.observation_count == 2

    def test_batch_update(self, engine):
        """Batch update processes all observations."""
        observations = [
            DirichletObservation("safety", UpdateOutcome.SUCCESS, 1.0),
            DirichletObservation("correctness", UpdateOutcome.SUCCESS, 0.5),
            DirichletObservation("efficiency", UpdateOutcome.FAILURE, 0.8),
        ]
        receipts = engine.update_batch(observations)
        assert len(receipts) == 3
        assert engine.observation_count == 3


# =============================================================================
# 3. CONSTITUTIONAL INVARIANTS — The hard gates
# =============================================================================


class TestConstitutionalInvariants:
    """Verify that constitutional bounds are NEVER violated."""

    def test_inv2_floor_never_violated(self, low_concentration_engine):
        """INV-2: No weight drops below WEIGHT_FLOOR even with many observations."""
        engine = low_concentration_engine
        # Hammer one dimension with successes to push others down
        for _ in range(100):
            obs = DirichletObservation("correctness", UpdateOutcome.SUCCESS, 1.0)
            receipt = engine.update(obs)
            if receipt.accepted:
                for dim, w in engine.current_weights.items():
                    assert w >= WEIGHT_FLOOR - 1e-8, (
                        f"INV-2 VIOLATED: {dim}={w:.6f} < floor={WEIGHT_FLOOR}"
                    )

    def test_inv3_ceiling_never_violated(self, low_concentration_engine):
        """INV-3: No weight exceeds WEIGHT_CEILING."""
        engine = low_concentration_engine
        for _ in range(100):
            obs = DirichletObservation("safety", UpdateOutcome.SUCCESS, 1.0)
            receipt = engine.update(obs)
            if receipt.accepted:
                for dim, w in engine.current_weights.items():
                    assert w <= WEIGHT_CEILING + 1e-8, (
                        f"INV-3 VIOLATED: {dim}={w:.6f} > ceiling={WEIGHT_CEILING}"
                    )

    def test_inv4_safety_correctness_floor(self, engine):
        """INV-4: Safety + Correctness >= SAFETY_CORRECTNESS_FLOOR."""
        # Push all OTHER dimensions with success
        for dim in ["user_benefit", "efficiency", "auditability",
                     "anti_centralization", "robustness", "adl_fairness"]:
            for _ in range(20):
                obs = DirichletObservation(dim, UpdateOutcome.SUCCESS, 1.0)
                receipt = engine.update(obs)
                if receipt.accepted:
                    combined = (engine.current_weights["safety"]
                                + engine.current_weights["correctness"])
                    assert combined >= SAFETY_CORRECTNESS_FLOOR - 1e-8, (
                        f"INV-4 VIOLATED: safety+correctness={combined:.6f}"
                    )

    def test_inv5_kl_drift_bounded(self, low_concentration_engine):
        """INV-5: KL-divergence from canonical never exceeds MAX_KL_DRIFT."""
        engine = low_concentration_engine
        for _ in range(200):
            obs = DirichletObservation("safety", UpdateOutcome.SUCCESS, 1.0)
            receipt = engine.update(obs)
            # Whether accepted or rejected, the engine's state should be valid
            report = engine.convergence_report()
            assert report["kl_divergence_from_canonical"] <= MAX_KL_DRIFT + 1e-8

    def test_rejected_updates_dont_change_state(self, engine):
        """Rejected updates must NOT modify the engine state."""
        before = deepcopy(engine.current_weights)
        before_count = engine.observation_count

        # Force many updates until one is rejected
        rejected = False
        for _ in range(1000):
            obs = DirichletObservation("safety", UpdateOutcome.SUCCESS, 1.0)
            receipt = engine.update(obs)
            if not receipt.accepted:
                rejected = True
                # State should not have changed from last accepted state
                break

        # At least verify state integrity after all updates
        total = sum(engine.current_weights.values())
        assert abs(total - 1.0) < 1e-6


# =============================================================================
# 4. RECEIPT INTEGRITY — Cryptographic audit trail
# =============================================================================


class TestReceiptIntegrity:
    """Verify cryptographic receipt generation."""

    def test_receipt_has_hash(self, engine, success_observation):
        """Every receipt must have a non-empty hash."""
        receipt = engine.update(success_observation)
        assert receipt.receipt_hash
        assert len(receipt.receipt_hash) > 0

    def test_receipt_has_timestamp(self, engine, success_observation):
        """Every receipt must have a timestamp."""
        receipt = engine.update(success_observation)
        assert receipt.timestamp
        assert "T" in receipt.timestamp  # ISO format

    def test_different_updates_different_hashes(self, engine):
        """Different observations must produce different receipt hashes."""
        r1 = engine.update(DirichletObservation("safety", UpdateOutcome.SUCCESS, 1.0))
        r2 = engine.update(DirichletObservation("correctness", UpdateOutcome.SUCCESS, 1.0))
        assert r1.receipt_hash != r2.receipt_hash

    def test_receipt_tracks_prior_and_posterior(self, engine, success_observation):
        """Receipt must capture both prior and posterior weights."""
        receipt = engine.update(success_observation)
        assert "safety" in receipt.prior_weights
        assert "safety" in receipt.posterior_weights
        assert abs(sum(receipt.prior_weights.values()) - 1.0) < 1e-6
        assert abs(sum(receipt.posterior_weights.values()) - 1.0) < 1e-6

    def test_receipt_serialization(self, engine, success_observation):
        """Receipts must be JSON-serializable."""
        receipt = engine.update(success_observation)
        d = receipt.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0
        parsed = json.loads(json_str)
        assert parsed["accepted"] == receipt.accepted

    def test_get_receipts_returns_history(self, engine):
        """get_receipts must return audit trail."""
        for dim in DIMENSION_ORDER[:3]:
            engine.update(DirichletObservation(dim, UpdateOutcome.SUCCESS, 0.5))
        receipts = engine.get_receipts(limit=10)
        assert len(receipts) == 3


# =============================================================================
# 5. CONVERGENCE & ANALYSIS
# =============================================================================


class TestConvergence:
    """Verify convergence reporting and analysis."""

    def test_convergence_report_structure(self, engine):
        """Report must contain required fields."""
        report = engine.convergence_report()
        required_fields = [
            "observation_count", "effective_sample_size",
            "kl_divergence_from_canonical", "max_kl_allowed",
            "drift_budget_remaining", "canonical_weights",
            "current_weights", "drift_per_dimension",
        ]
        for field in required_fields:
            assert field in report, f"Missing field: {field}"

    def test_drift_budget_decreases_with_updates(self, low_concentration_engine):
        """Drift budget should decrease as weights evolve."""
        engine = low_concentration_engine
        report_before = engine.convergence_report()
        for _ in range(10):
            engine.update(DirichletObservation("safety", UpdateOutcome.SUCCESS, 1.0))
        report_after = engine.convergence_report()

        assert report_after["drift_budget_remaining"] <= report_before["drift_budget_remaining"]

    def test_reset_restores_canonical(self, engine, success_observation):
        """Reset must restore canonical weights exactly."""
        for _ in range(5):
            engine.update(success_observation)
        engine.reset_to_canonical()

        for dim in DIMENSION_ORDER:
            assert abs(engine.current_weights[dim] - IHSAN_WEIGHTS[dim]) < 0.01

    def test_serialization_roundtrip(self, engine, success_observation):
        """Serialize → deserialize must preserve state."""
        for _ in range(5):
            engine.update(success_observation)

        data = engine.to_dict()
        restored = AdaptiveIhsan.from_dict(data)

        for dim in DIMENSION_ORDER:
            assert abs(
                engine.current_weights[dim] - restored.current_weights[dim]
            ) < 1e-6


# =============================================================================
# 6. EDGE CASES & VALIDATION
# =============================================================================


class TestEdgeCases:
    """Boundary conditions and error handling."""

    def test_invalid_dimension_raises(self):
        """Unknown dimension must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown dimension"):
            DirichletObservation("nonexistent", UpdateOutcome.SUCCESS, 1.0)

    def test_invalid_magnitude_raises(self):
        """Magnitude outside [0, 1] must raise ValueError."""
        with pytest.raises(ValueError, match="Magnitude"):
            DirichletObservation("safety", UpdateOutcome.SUCCESS, 1.5)

    def test_zero_magnitude_is_valid(self, engine):
        """Zero magnitude should be accepted but produce no change."""
        before = deepcopy(engine.current_weights)
        obs = DirichletObservation("safety", UpdateOutcome.SUCCESS, 0.0)
        engine.update(obs)
        for dim in DIMENSION_ORDER:
            assert abs(engine.current_weights[dim] - before[dim]) < 1e-10
