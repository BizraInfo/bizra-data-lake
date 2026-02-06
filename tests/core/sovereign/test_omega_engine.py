"""
BIZRA Omega Engine Test Suite
Comprehensive testing for GAP-C1, GAP-C2, GAP-C4 implementations.

Standing on Giants: TDD (Kent Beck), Property-Based Testing (QuickCheck)
"""

import math
import pytest
import numpy as np
from datetime import datetime, timezone, timedelta

from core.sovereign.omega_engine import (
    # GAP-C1: Ihsan Projector
    NTUState,
    IhsanVector,
    IhsanProjector,
    # GAP-C2: Adl Invariant
    AdlInvariant,
    AdlViolation,
    # GAP-C4: Treasury Mode
    TreasuryMode,
    TreasuryState,
    TreasuryController,
    # Unified
    OmegaEngine,
    create_omega_engine,
    ihsan_from_scores,
)


# ═══════════════════════════════════════════════════════════════════════════════
# GAP-C1: IHSAN PROJECTOR TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestNTUState:
    """Tests for NeuroTemporal Unit State."""

    def test_ntu_valid_ranges(self):
        """NTU should clamp values to valid ranges."""
        ntu = NTUState(belief=1.5, entropy=-0.5, potential=2.0)
        assert ntu.belief == 1.0
        assert ntu.entropy == 0.0
        assert ntu.potential == 1.0

    def test_ntu_negative_clamp(self):
        """NTU should clamp negative values."""
        ntu = NTUState(belief=-0.5, entropy=-1.0, potential=-1.5)
        assert ntu.belief == 0.0
        assert ntu.entropy == 0.0
        assert ntu.potential == -1.0

    def test_ntu_magnitude(self):
        """NTU magnitude calculation."""
        ntu = NTUState(belief=0.6, entropy=0.8, potential=0.0)
        expected = math.sqrt(0.6**2 + 0.8**2)
        assert abs(ntu.magnitude - expected) < 1e-9

    def test_ntu_to_dict(self):
        """NTU serialization."""
        ntu = NTUState(belief=0.5, entropy=0.3, potential=-0.2)
        d = ntu.to_dict()
        assert d["belief"] == 0.5
        assert d["entropy"] == 0.3
        assert d["potential"] == -0.2


class TestIhsanVector:
    """Tests for 8D Ihsan Vector."""

    def test_ihsan_to_array(self):
        """Ihsan should convert to numpy array."""
        ihsan = IhsanVector(
            truthfulness=0.9,
            trustworthiness=0.8,
            justice=0.95,
            excellence=0.85,
            wisdom=0.9,
            compassion=0.88,
            patience=0.92,
            gratitude=0.87,
        )
        arr = ihsan.to_array()
        assert arr.shape == (8,)
        assert arr[0] == 0.9
        assert arr[2] == 0.95

    def test_ihsan_from_array(self):
        """Ihsan should reconstruct from array."""
        arr = np.array([0.9, 0.8, 0.95, 0.85, 0.9, 0.88, 0.92, 0.87])
        ihsan = IhsanVector.from_array(arr)
        assert ihsan.truthfulness == 0.9
        assert ihsan.justice == 0.95

    def test_ihsan_minimum(self):
        """Ihsan minimum should find lowest dimension."""
        ihsan = IhsanVector(
            truthfulness=0.9,
            trustworthiness=0.3,  # Minimum
            justice=0.95,
            excellence=0.85,
            wisdom=0.9,
            compassion=0.88,
            patience=0.92,
            gratitude=0.87,
        )
        assert ihsan.minimum == 0.3

    def test_ihsan_geometric_mean(self):
        """Ihsan geometric mean calculation."""
        ihsan = IhsanVector(
            truthfulness=0.9,
            trustworthiness=0.9,
            justice=0.9,
            excellence=0.9,
            wisdom=0.9,
            compassion=0.9,
            patience=0.9,
            gratitude=0.9,
        )
        # All 0.9 → geometric mean = 0.9
        assert abs(ihsan.geometric_mean - 0.9) < 1e-9


class TestIhsanProjector:
    """Tests for GAP-C1: 8D → 3D Projection."""

    def test_projector_output_shape(self):
        """Projector should output valid NTU state."""
        projector = IhsanProjector()
        ihsan = IhsanVector(
            truthfulness=0.9,
            trustworthiness=0.9,
            justice=0.9,
            excellence=0.9,
            wisdom=0.9,
            compassion=0.9,
            patience=0.9,
            gratitude=0.9,
        )
        ntu = projector.project(ihsan)
        assert isinstance(ntu, NTUState)
        assert 0 <= ntu.belief <= 1
        assert ntu.entropy >= 0
        assert -1 <= ntu.potential <= 1

    def test_projector_o1_complexity(self):
        """Projector should complete in O(1) time (constant ops)."""
        import time

        projector = IhsanProjector()
        ihsan = IhsanVector(0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9)

        # Run 1000 times, should be fast
        start = time.perf_counter()
        for _ in range(1000):
            projector.project(ihsan)
        elapsed = time.perf_counter() - start

        # Should complete in < 100ms for 1000 iterations
        assert elapsed < 0.1, f"Projection too slow: {elapsed:.3f}s for 1000 iters"

    def test_projector_doubt_on_low_dimension(self):
        """Projector should reduce belief if any dimension < 0.5."""
        projector = IhsanProjector()

        # High all dimensions
        high = IhsanVector(0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9)
        ntu_high = projector.project(high)

        # Low trustworthiness
        low = IhsanVector(0.9, 0.3, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9)
        ntu_low = projector.project(low)

        # Belief should be lower when a dimension is compromised
        assert ntu_low.belief < ntu_high.belief

    def test_projector_justice_affects_potential(self):
        """Justice (عدل) should have highest weight on potential."""
        projector = IhsanProjector()

        # High justice
        high_justice = IhsanVector(0.5, 0.5, 0.95, 0.5, 0.5, 0.5, 0.5, 0.5)
        ntu_high = projector.project(high_justice)

        # Low justice
        low_justice = IhsanVector(0.5, 0.5, 0.3, 0.5, 0.5, 0.5, 0.5, 0.5)
        ntu_low = projector.project(low_justice)

        # Potential should differ significantly
        assert ntu_high.potential > ntu_low.potential

    def test_projector_excellence_affects_belief(self):
        """Excellence (إحسان) should primarily affect belief."""
        projector = IhsanProjector()

        # High excellence
        high = IhsanVector(0.5, 0.5, 0.5, 0.95, 0.5, 0.5, 0.5, 0.5)
        ntu_high = projector.project(high)

        # Low excellence
        low = IhsanVector(0.5, 0.5, 0.5, 0.3, 0.5, 0.5, 0.5, 0.5)
        ntu_low = projector.project(low)

        # Belief should differ (accounting for doubt penalty on low)
        assert ntu_high.belief > ntu_low.belief

    def test_projector_inverse_approximate(self):
        """Inverse projection should approximate original (lossy)."""
        projector = IhsanProjector()
        original = IhsanVector(0.8, 0.85, 0.9, 0.88, 0.82, 0.79, 0.86, 0.84)

        ntu = projector.project(original)
        recovered = projector.inverse_project(ntu, prior=original)

        # Should be within 20% of original on average
        diff = np.abs(original.to_array() - recovered.to_array())
        assert diff.mean() < 0.2


# ═══════════════════════════════════════════════════════════════════════════════
# GAP-C2: ADL INVARIANT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestAdlInvariant:
    """Tests for GAP-C2: Economic Justice Enforcement."""

    def test_gini_empty_holdings(self):
        """Gini of empty holdings is 0."""
        assert AdlInvariant.calculate_gini({}) == 0.0

    def test_gini_single_holder(self):
        """Gini of single holder is 0 (perfect equality with self)."""
        assert AdlInvariant.calculate_gini({"alice": 1000}) == 0.0

    def test_gini_perfect_equality(self):
        """Gini of equal holdings is ~0."""
        holdings = {f"user_{i}": 100 for i in range(10)}
        gini = AdlInvariant.calculate_gini(holdings)
        assert gini < 0.01  # Should be 0 for perfect equality

    def test_gini_perfect_inequality(self):
        """Gini approaches 1 for extreme inequality."""
        # One person has everything, rest have 0
        holdings = {"billionaire": 1_000_000}
        holdings.update({f"pauper_{i}": 0.01 for i in range(1000)})
        gini = AdlInvariant.calculate_gini(holdings)
        assert gini > 0.9  # Should be close to 1

    def test_gini_moderate_inequality(self):
        """Gini calculation for realistic distribution."""
        holdings = {
            "alice": 500,
            "bob": 300,
            "charlie": 150,
            "diana": 50,
        }
        gini = AdlInvariant.calculate_gini(holdings)
        # Should be around 0.25-0.35 for this distribution
        assert 0.2 < gini < 0.4

    def test_validate_transaction_valid(self):
        """Valid transaction should pass."""
        adl = AdlInvariant(gini_threshold=0.40)

        pre_state = {"alice": 500, "bob": 500}
        post_state = {"alice": 400, "bob": 600}  # Transfer 100

        valid, violation = adl.validate_transaction(pre_state, post_state, "tx_001")
        assert valid is True
        assert violation is None

    def test_validate_transaction_gini_violation(self):
        """Transaction exceeding Gini threshold should be rejected."""
        adl = AdlInvariant(gini_threshold=0.40)

        pre_state = {"alice": 500, "bob": 500}
        post_state = {"alice": 950, "bob": 50}  # Extreme transfer

        valid, violation = adl.validate_transaction(pre_state, post_state, "tx_002")
        assert valid is False
        assert violation is not None
        assert "Gini violation" in violation.reason
        assert violation.post_gini > 0.40

    def test_validate_transaction_conservation_violation(self):
        """Transaction violating conservation should be rejected."""
        adl = AdlInvariant(gini_threshold=0.40)

        pre_state = {"alice": 500, "bob": 500}
        post_state = {"alice": 600, "bob": 600}  # Created value!

        valid, violation = adl.validate_transaction(pre_state, post_state, "tx_003")
        assert valid is False
        assert violation is not None
        assert "Conservation violation" in violation.reason

    def test_harberger_redistribution(self):
        """Harberger tax should redistribute wealth."""
        adl = AdlInvariant(harberger_rate=0.05)

        holdings = {
            "rich": 900,
            "poor": 100,
        }

        # Apply 1 year of tax
        new_holdings = adl.redistribute_harberger_tax(holdings, period_days=365)

        # Total should be conserved
        assert abs(sum(new_holdings.values()) - 1000) < 1e-9

        # Gap should be smaller
        old_gap = holdings["rich"] - holdings["poor"]
        new_gap = new_holdings["rich"] - new_holdings["poor"]
        assert new_gap < old_gap

    def test_violation_logging(self):
        """Violations should be logged."""
        violations_received = []

        def on_violation(v):
            violations_received.append(v)

        adl = AdlInvariant(gini_threshold=0.40, on_violation=on_violation)

        pre_state = {"alice": 500, "bob": 500}
        post_state = {"alice": 950, "bob": 50}

        adl.validate_transaction(pre_state, post_state, "tx_004")

        assert len(violations_received) == 1
        assert len(adl.get_violation_history()) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# GAP-C4: TREASURY MODE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestTreasuryMode:
    """Tests for GAP-C4: Treasury Graceful Degradation."""

    def test_initial_state_ethical(self):
        """Treasury should start in ETHICAL mode."""
        controller = TreasuryController()
        assert controller.state.mode == TreasuryMode.ETHICAL

    def test_transition_to_hibernation(self):
        """Low ethics should trigger HIBERNATION."""
        controller = TreasuryController(
            initial_reserves_days=90,
            initial_ethical_score=0.80,
        )

        # Update with low ethics market
        market_data = {
            "volatility": 0.8,  # High volatility
            "manipulation_score": 0.5,
            "compliance_score": 0.3,
            "social_impact": 0.2,
        }

        state = controller.update(market_data, elapsed_days=1)
        assert state.mode == TreasuryMode.HIBERNATION

    def test_transition_to_emergency(self):
        """Low reserves should trigger EMERGENCY."""
        controller = TreasuryController(
            initial_reserves_days=5,  # Below 7-day threshold
            initial_ethical_score=0.80,
        )

        market_data = {"volatility": 0.2, "manipulation_score": 0.1}
        state = controller.update(market_data, elapsed_days=1)
        assert state.mode == TreasuryMode.EMERGENCY

    def test_burn_rate_by_mode(self):
        """Burn rate should vary by mode."""
        controller = TreasuryController()

        # ETHICAL mode
        assert controller.calculate_burn_rate() == 1.0

        # Force HIBERNATION
        controller.state = TreasuryState(
            mode=TreasuryMode.HIBERNATION,
            reserves_days=50,
            ethical_score=0.5,
            burn_rate=0.2,
            last_transition=datetime.now(timezone.utc),
            transition_reason="test",
        )
        assert controller.calculate_burn_rate() == 0.2

        # Force EMERGENCY
        controller.state = TreasuryState(
            mode=TreasuryMode.EMERGENCY,
            reserves_days=3,
            ethical_score=0.3,
            burn_rate=0.05,
            last_transition=datetime.now(timezone.utc),
            transition_reason="test",
        )
        assert controller.calculate_burn_rate() == 0.05

    def test_recovery_from_hibernation(self):
        """Good ethics should allow recovery to ETHICAL."""
        transitions = []

        def on_transition(old, new, reason):
            transitions.append((old, new))

        controller = TreasuryController(
            initial_reserves_days=90,
            initial_ethical_score=0.55,  # Below threshold
            on_transition=on_transition,
        )

        # Force hibernation first
        bad_market = {
            "volatility": 0.8,
            "manipulation_score": 0.5,
            "compliance_score": 0.3,
            "social_impact": 0.2,
        }
        controller.update(bad_market, elapsed_days=1)
        assert controller.state.mode == TreasuryMode.HIBERNATION

        # Now improve market
        good_market = {
            "volatility": 0.1,
            "manipulation_score": 0.05,
            "compliance_score": 0.95,
            "social_impact": 0.9,
        }
        controller.update(good_market, elapsed_days=1)
        assert controller.state.mode == TreasuryMode.ETHICAL

    def test_ethics_evaluation(self):
        """Ethics score should reflect market conditions."""
        controller = TreasuryController()

        # Perfect market
        perfect = {
            "volatility": 0.0,
            "manipulation_score": 0.0,
            "compliance_score": 1.0,
            "social_impact": 1.0,
        }
        assert controller.evaluate_market_ethics(perfect) == 1.0

        # Terrible market
        terrible = {
            "volatility": 1.0,
            "manipulation_score": 1.0,
            "compliance_score": 0.0,
            "social_impact": 0.0,
        }
        assert controller.evaluate_market_ethics(terrible) == 0.0

    def test_transition_history(self):
        """Transitions should be logged."""
        controller = TreasuryController(
            initial_reserves_days=90,
            initial_ethical_score=0.80,
        )

        # Trigger hibernation
        bad_market = {"volatility": 0.9, "manipulation_score": 0.7}
        controller.update(bad_market, elapsed_days=1)

        history = controller.get_transition_history()
        assert len(history) == 1
        assert history[0][1] == TreasuryMode.ETHICAL
        assert history[0][2] == TreasuryMode.HIBERNATION


# ═══════════════════════════════════════════════════════════════════════════════
# OMEGA ENGINE INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestOmegaEngine:
    """Tests for unified Omega Engine."""

    def test_create_omega_engine(self):
        """Factory should create configured engine."""
        engine = create_omega_engine(gini_threshold=0.35, initial_reserves_days=60)

        assert engine.adl.gini_threshold == 0.35
        assert engine.treasury.state.reserves_days == 60

    def test_evaluate_ihsan(self):
        """Engine should evaluate Ihsan and return NTU."""
        engine = OmegaEngine()

        ihsan = IhsanVector(0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9)
        score, ntu = engine.evaluate_ihsan(ihsan)

        assert abs(score - 0.9) < 0.01  # Geometric mean of 0.9s
        assert isinstance(ntu, NTUState)

    def test_validate_economic_action(self):
        """Engine should validate economic actions."""
        engine = OmegaEngine()

        pre = {"alice": 500, "bob": 500}
        post = {"alice": 400, "bob": 600}

        valid, error = engine.validate_economic_action(pre, post, "action_001")
        assert valid is True
        assert error is None

    def test_validate_economic_action_rejection(self):
        """Engine should reject Adl violations."""
        engine = OmegaEngine()

        pre = {"alice": 500, "bob": 500}
        post = {"alice": 950, "bob": 50}

        valid, error = engine.validate_economic_action(pre, post, "action_002")
        assert valid is False
        assert "Gini" in error

    def test_update_treasury(self):
        """Engine should update treasury state."""
        engine = OmegaEngine()

        market_data = {"volatility": 0.2, "compliance_score": 0.9}
        state = engine.update_treasury(market_data, elapsed_days=1)

        assert isinstance(state, TreasuryState)

    def test_get_operational_mode(self):
        """Engine should report operational mode."""
        engine = OmegaEngine()
        mode = engine.get_operational_mode()
        assert mode == TreasuryMode.ETHICAL

    def test_get_status(self):
        """Engine should report comprehensive status."""
        engine = OmegaEngine()
        status = engine.get_status()

        assert "ihsan_projector" in status
        assert "adl_invariant" in status
        assert "treasury" in status
        assert status["treasury"]["mode"] == "ETHICAL"


class TestIhsanFromScores:
    """Tests for simplified Ihsan creation."""

    def test_ihsan_from_scores_default(self):
        """Default scores should create valid Ihsan."""
        ihsan = ihsan_from_scores()
        assert ihsan.truthfulness == 0.9
        assert ihsan.excellence == 0.9

    def test_ihsan_from_scores_custom(self):
        """Custom scores should propagate."""
        ihsan = ihsan_from_scores(
            correctness=0.95,
            safety=0.90,
            user_benefit=0.85,
            efficiency=0.80,
        )
        assert ihsan.truthfulness == 0.95
        assert ihsan.justice == 0.90
        assert ihsan.compassion == 0.85
        assert ihsan.excellence == 0.80


# ═══════════════════════════════════════════════════════════════════════════════
# PROPERTY-BASED TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestProperties:
    """Property-based tests for invariants."""

    def test_gini_bounded(self):
        """Gini should always be in [0, 1]."""
        for _ in range(100):
            n = np.random.randint(1, 20)
            values = np.random.exponential(100, n)
            holdings = {f"user_{i}": v for i, v in enumerate(values)}

            gini = AdlInvariant.calculate_gini(holdings)
            assert 0.0 <= gini <= 1.0

    def test_ntu_projection_bounded(self):
        """NTU projection should always produce bounded values."""
        projector = IhsanProjector()

        for _ in range(100):
            values = np.random.random(8)
            ihsan = IhsanVector.from_array(values)
            ntu = projector.project(ihsan)

            assert 0.0 <= ntu.belief <= 1.0
            assert ntu.entropy >= 0.0
            assert -1.0 <= ntu.potential <= 1.0

    def test_harberger_conserves_value(self):
        """Harberger redistribution should conserve total value."""
        adl = AdlInvariant(harberger_rate=0.05)

        for _ in range(50):
            n = np.random.randint(2, 10)
            values = np.random.exponential(100, n)
            holdings = {f"user_{i}": v for i, v in enumerate(values)}

            original_total = sum(holdings.values())
            new_holdings = adl.redistribute_harberger_tax(holdings, period_days=30)
            new_total = sum(new_holdings.values())

            assert abs(original_total - new_total) < 1e-6
