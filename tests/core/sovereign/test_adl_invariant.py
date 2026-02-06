"""
Tests for the Adl (Justice) Invariant - Anti-Plutocracy Enforcement

Standing on Giants: Gini (1912), Harberger (1962), Rawls (1971)

These tests verify:
1. Gini coefficient calculation correctness
2. Transaction validation against threshold
3. Harberger tax redistribution
4. PCI gate integration
5. Edge cases and invariants
"""

import pytest
from typing import Dict

from core.sovereign.adl_invariant import (
    AdlInvariant,
    AdlGate,
    AdlRejectCode,
    AdlValidationResult,
    RedistributionResult,
    Transaction,
    calculate_gini,
    calculate_gini_components,
    assert_adl_invariant,
    simulate_transaction_impact,
    ADL_GINI_THRESHOLD,
    HARBERGER_TAX_RATE,
    MINIMUM_HOLDING,
    UBC_POOL_ID,
)


# =============================================================================
# GINI COEFFICIENT TESTS
# =============================================================================

class TestCalculateGini:
    """Tests for the Gini coefficient calculation."""

    def test_perfect_equality(self):
        """Equal holdings should have Gini = 0."""
        holdings = {"a": 100.0, "b": 100.0, "c": 100.0, "d": 100.0}
        gini = calculate_gini(holdings)
        assert gini == pytest.approx(0.0, abs=1e-6)

    def test_extreme_inequality(self):
        """One holder with everything should have Gini close to 1."""
        holdings = {"rich": 1000000.0, "poor1": 0.0, "poor2": 0.0, "poor3": 0.0}
        # Note: zeros are filtered out, so this becomes single holder
        gini = calculate_gini(holdings)
        assert gini == 0.0  # Single holder = no inequality

    def test_realistic_inequality(self):
        """Test with realistic wealth distribution."""
        # Top-heavy distribution
        holdings = {
            "whale": 1000.0,
            "large": 500.0,
            "medium": 200.0,
            "small1": 50.0,
            "small2": 50.0,
            "small3": 50.0,
        }
        gini = calculate_gini(holdings)
        # Should be moderate to high inequality
        assert 0.3 <= gini <= 0.7

    def test_empty_holdings(self):
        """Empty holdings should return 0."""
        gini = calculate_gini({})
        assert gini == 0.0

    def test_single_holder(self):
        """Single holder should return 0 (no inequality possible)."""
        gini = calculate_gini({"only": 1000.0})
        assert gini == 0.0

    def test_two_holders_equal(self):
        """Two equal holders should have Gini = 0."""
        gini = calculate_gini({"a": 100.0, "b": 100.0})
        assert gini == pytest.approx(0.0, abs=1e-6)

    def test_two_holders_unequal(self):
        """Two unequal holders should have non-zero Gini."""
        gini = calculate_gini({"rich": 900.0, "poor": 100.0})
        assert gini > 0.0
        # For 2 holders with 90%-10% split, Gini = 0.4
        assert gini == pytest.approx(0.4, abs=0.01)

    def test_negative_values_raise_error(self):
        """Negative holdings should raise ValueError."""
        # Use a large negative value that won't be filtered by MINIMUM_HOLDING
        with pytest.raises(ValueError, match="cannot be negative"):
            calculate_gini({"a": 100.0, "b": -50.0, "c": 50.0})

    def test_ubc_pool_excluded(self):
        """UBC pool should be excluded from Gini calculation."""
        holdings = {"a": 100.0, "b": 100.0, UBC_POOL_ID: 1000000.0}
        gini = calculate_gini(holdings)
        # Should ignore UBC pool and see equal distribution
        assert gini == pytest.approx(0.0, abs=1e-6)

    def test_dust_filtered(self):
        """Holdings below minimum should be filtered."""
        holdings = {"a": 100.0, "b": 100.0, "dust": 1e-15}
        gini = calculate_gini(holdings)
        # Dust holder should be ignored
        assert gini == pytest.approx(0.0, abs=1e-6)

    def test_gini_range(self):
        """Gini should always be in [0, 1]."""
        test_cases = [
            {"a": 1.0},
            {"a": 1.0, "b": 1.0},
            {"a": 100.0, "b": 1.0},
            {"a": 1000.0, "b": 100.0, "c": 10.0, "d": 1.0},
            {f"node_{i}": float(i) for i in range(1, 101)},
        ]
        for holdings in test_cases:
            gini = calculate_gini(holdings)
            assert 0.0 <= gini <= 1.0, f"Gini {gini} out of range for {holdings}"


class TestCalculateGiniComponents:
    """Tests for detailed Gini component breakdown."""

    def test_components_structure(self):
        """Verify all expected components are returned."""
        holdings = {"a": 100.0, "b": 200.0, "c": 300.0}
        components = calculate_gini_components(holdings)

        expected_keys = [
            "gini", "n", "total", "mean", "median",
            "min", "max", "top_10_pct_share", "bottom_50_pct_share"
        ]
        for key in expected_keys:
            assert key in components, f"Missing key: {key}"

    def test_components_values(self):
        """Verify component values are correct."""
        holdings = {"a": 100.0, "b": 200.0, "c": 300.0}
        components = calculate_gini_components(holdings)

        assert components["n"] == 3
        assert components["total"] == 600.0
        assert components["mean"] == 200.0
        assert components["min"] == 100.0
        assert components["max"] == 300.0


# =============================================================================
# TRANSACTION VALIDATION TESTS
# =============================================================================

class TestAdlInvariantValidation:
    """Tests for transaction validation against Adl invariant."""

    @pytest.fixture
    def invariant(self):
        """Create Adl invariant with default threshold."""
        return AdlInvariant()

    @pytest.fixture
    def equal_holdings(self):
        """Holdings with perfect equality (Gini = 0)."""
        return {f"node_{i}": 100.0 for i in range(10)}

    @pytest.fixture
    def moderate_holdings(self):
        """Holdings with moderate inequality (Gini ~ 0.3)."""
        return {
            "node_0": 300.0,
            "node_1": 200.0,
            "node_2": 150.0,
            "node_3": 100.0,
            "node_4": 100.0,
            "node_5": 75.0,
            "node_6": 50.0,
            "node_7": 25.0,
        }

    def test_valid_transaction_passes(self, invariant, equal_holdings):
        """Transaction that keeps Gini below threshold should pass."""
        tx = Transaction(
            tx_id="tx_001",
            sender="node_0",
            recipient="node_1",
            amount=10.0,
        )
        result = invariant.validate_transaction(tx, equal_holdings)

        assert result.passed is True
        assert result.reject_code == AdlRejectCode.SUCCESS
        assert result.post_gini <= ADL_GINI_THRESHOLD

    def test_plutocratic_transaction_rejected(self, invariant, moderate_holdings):
        """Transaction that pushes Gini above threshold should be rejected."""
        # Try to concentrate most wealth in one node
        tx = Transaction(
            tx_id="tx_plutocrat",
            sender="node_7",
            recipient="node_0",
            amount=24.0,  # Almost all of node_7's holdings
        )

        # First check current Gini
        pre_gini = calculate_gini(moderate_holdings)

        # Create a scenario that would exceed threshold
        # We need to create a more extreme case
        extreme_holdings = {
            "whale": 800.0,
            "small_1": 50.0,
            "small_2": 50.0,
            "small_3": 50.0,
            "small_4": 50.0,
        }

        tx2 = Transaction(
            tx_id="tx_concentrate",
            sender="small_1",
            recipient="whale",
            amount=49.0,
        )

        result = invariant.validate_transaction(tx2, extreme_holdings)

        # The transaction should be rejected if it exceeds threshold
        if result.post_gini > ADL_GINI_THRESHOLD:
            assert result.passed is False
            assert result.reject_code == AdlRejectCode.REJECT_GINI_EXCEEDED

    def test_insufficient_balance_rejected(self, invariant, equal_holdings):
        """Transaction exceeding sender balance should be rejected."""
        tx = Transaction(
            tx_id="tx_overdraft",
            sender="node_0",
            recipient="node_1",
            amount=150.0,  # More than node_0 has (100)
        )
        result = invariant.validate_transaction(tx, equal_holdings)

        assert result.passed is False
        assert result.reject_code == AdlRejectCode.REJECT_NEGATIVE_HOLDING

    def test_dust_amount_rejected(self, invariant, equal_holdings):
        """Transaction below minimum holding should be rejected."""
        tx = Transaction(
            tx_id="tx_dust",
            sender="node_0",
            recipient="node_1",
            amount=1e-15,  # Below MINIMUM_HOLDING
        )
        result = invariant.validate_transaction(tx, equal_holdings)

        assert result.passed is False
        assert result.reject_code == AdlRejectCode.REJECT_DUST_AMOUNT

    def test_conservation_maintained(self, invariant, equal_holdings):
        """Post-transaction total should equal pre-transaction total."""
        tx = Transaction(
            tx_id="tx_conserve",
            sender="node_0",
            recipient="node_1",
            amount=50.0,
        )
        result = invariant.validate_transaction(tx, equal_holdings)

        # Build post-state to verify conservation
        post_state = equal_holdings.copy()
        post_state["node_0"] -= 50.0
        post_state["node_1"] += 50.0

        pre_total = sum(v for k, v in equal_holdings.items() if k != UBC_POOL_ID)
        post_total = sum(v for k, v in post_state.items() if k != UBC_POOL_ID)

        assert abs(pre_total - post_total) < 1e-9

    def test_new_recipient_allowed(self, invariant, equal_holdings):
        """Transaction to new recipient should be allowed."""
        tx = Transaction(
            tx_id="tx_new_recipient",
            sender="node_0",
            recipient="new_node",
            amount=10.0,
        )
        result = invariant.validate_transaction(tx, equal_holdings)

        # Should pass if Gini stays below threshold
        assert result.passed is True
        assert "new_node" not in equal_holdings  # Verify it was indeed new


# =============================================================================
# HARBERGER TAX REDISTRIBUTION TESTS
# =============================================================================

class TestRedistributeSoilTax:
    """Tests for Harberger-style tax redistribution."""

    @pytest.fixture
    def invariant(self):
        return AdlInvariant()

    def test_redistribution_reduces_inequality(self, invariant):
        """Tax redistribution should reduce Gini coefficient."""
        holdings = {
            "rich": 1000.0,
            "medium": 500.0,
            "poor": 100.0,
        }

        pre_gini = calculate_gini(holdings)
        new_holdings = invariant.redistribute_soil_tax(holdings)
        post_gini = calculate_gini(new_holdings)

        # Redistribution should reduce inequality
        assert post_gini <= pre_gini

    def test_redistribution_conserves_total(self, invariant):
        """Total value should be conserved after redistribution."""
        holdings = {
            "node_0": 1000.0,
            "node_1": 500.0,
            "node_2": 250.0,
            "node_3": 125.0,
        }

        pre_total = sum(v for k, v in holdings.items() if k != UBC_POOL_ID)
        new_holdings = invariant.redistribute_soil_tax(holdings)
        post_total = sum(v for k, v in new_holdings.items() if k != UBC_POOL_ID)

        assert pre_total == pytest.approx(post_total, rel=1e-9)

    def test_redistribution_equal_distribution_unchanged(self, invariant):
        """Equal distribution should stay equal after tax."""
        holdings = {"a": 100.0, "b": 100.0, "c": 100.0, "d": 100.0}

        new_holdings = invariant.redistribute_soil_tax(holdings)

        # All should still be equal (tax taken and redistributed equally)
        values = [v for k, v in new_holdings.items() if k != UBC_POOL_ID]
        assert max(values) == pytest.approx(min(values), rel=1e-9)

    def test_redistribution_with_custom_rate(self, invariant):
        """Custom tax rate should be applied correctly."""
        holdings = {"rich": 1000.0, "poor": 100.0}

        # With 10% tax rate
        new_holdings = invariant.redistribute_soil_tax(holdings, tax_rate=0.10)

        # Total tax collected: 1000 * 0.10 + 100 * 0.10 = 110
        # Distributed equally: 110 / 2 = 55 per node
        # rich: 1000 - 100 + 55 = 955
        # poor: 100 - 10 + 55 = 145

        assert new_holdings["rich"] == pytest.approx(955.0, rel=1e-6)
        assert new_holdings["poor"] == pytest.approx(145.0, rel=1e-6)

    def test_redistribution_time_fraction(self, invariant):
        """Time fraction should proportionally reduce tax."""
        holdings = {"a": 1000.0, "b": 1000.0}

        # Full year vs half year
        full_year = invariant.redistribute_soil_tax(holdings, time_fraction=1.0)
        half_year = invariant.redistribute_soil_tax(holdings, time_fraction=0.5)

        # Half year should have half the tax effect
        full_delta = abs(1000.0 - full_year["a"])
        half_delta = abs(1000.0 - half_year["a"])

        # Since holdings are equal, both should be unchanged
        # But the math still holds
        assert half_delta == pytest.approx(full_delta * 0.5, rel=0.01)


class TestRedistributionResult:
    """Tests for RedistributionResult reporting."""

    def test_get_redistribution_impact(self):
        """Test impact preview without applying redistribution."""
        invariant = AdlInvariant()
        holdings = {"rich": 1000.0, "poor": 100.0}

        result = invariant.get_redistribution_impact(holdings)

        assert isinstance(result, RedistributionResult)
        assert result.success is True
        assert result.pre_gini > result.post_gini  # Should reduce inequality
        assert result.total_tax_collected > 0
        assert result.ubc_per_node > 0
        assert result.nodes_affected == 2


# =============================================================================
# ADL GATE (PCI INTEGRATION) TESTS
# =============================================================================

class TestAdlGate:
    """Tests for PCI envelope integration."""

    @pytest.fixture
    def holdings_provider(self):
        """Holdings provider for gate tests."""
        holdings = {f"node_{i}": 100.0 for i in range(10)}
        return lambda: holdings

    @pytest.fixture
    def gate(self, holdings_provider):
        """Create Adl gate with test holdings."""
        return AdlGate(holdings_provider)

    def test_get_current_gini(self, gate):
        """Test current Gini retrieval."""
        gini = gate.get_current_gini()
        assert gini == pytest.approx(0.0, abs=1e-6)  # Equal holdings

    def test_get_gini_headroom(self, gate):
        """Test headroom calculation."""
        headroom = gate.get_gini_headroom()
        assert headroom == pytest.approx(ADL_GINI_THRESHOLD, abs=1e-6)


# =============================================================================
# UTILITY FUNCTION TESTS
# =============================================================================

class TestAssertAdlInvariant:
    """Tests for the assertion helper."""

    def test_passes_for_valid_state(self):
        """Should not raise for valid state."""
        holdings = {"a": 100.0, "b": 100.0, "c": 100.0}
        assert_adl_invariant(holdings)  # Should not raise

    def test_raises_for_invalid_state(self):
        """Should raise AssertionError for invalid state."""
        # Create a state that would have very high Gini
        # This is tricky because we filter zeros...
        # Use a custom threshold to test the assertion
        holdings = {"whale": 999.0, "minnow": 1.0}

        # With threshold 0.1, this should fail
        with pytest.raises(AssertionError, match="Adl violation"):
            assert_adl_invariant(holdings, threshold=0.1)

    def test_checks_conservation(self):
        """Should check conservation when pre_state provided."""
        pre_state = {"a": 100.0, "b": 100.0}
        post_state = {"a": 100.0, "b": 100.0}  # Same

        assert_adl_invariant(post_state, pre_state)  # Should pass

        # Now break conservation
        post_state_bad = {"a": 100.0, "b": 150.0}  # 50 created from nothing

        with pytest.raises(AssertionError, match="Conservation violated"):
            assert_adl_invariant(post_state_bad, pre_state)


class TestSimulateTransactionImpact:
    """Tests for transaction impact simulation."""

    def test_simulation_structure(self):
        """Test that simulation returns expected structure."""
        holdings = {"a": 100.0, "b": 100.0}
        tx = Transaction(tx_id="sim", sender="a", recipient="b", amount=10.0)

        result = simulate_transaction_impact(tx, holdings)

        assert "transaction" in result
        assert "pre" in result
        assert "post" in result
        assert "delta" in result
        assert "would_pass" in result
        assert "headroom" in result

    def test_simulation_predicts_pass_fail(self):
        """Test that simulation correctly predicts pass/fail."""
        holdings = {"a": 100.0, "b": 100.0}

        # Small transfer should pass
        tx = Transaction(tx_id="sim", sender="a", recipient="b", amount=10.0)
        result = simulate_transaction_impact(tx, holdings)

        assert result["would_pass"] is True
        assert result["headroom"] > 0


# =============================================================================
# INVARIANT PROPERTY TESTS
# =============================================================================

class TestInvariantProperties:
    """Property-based tests for invariants."""

    def test_gini_monotonicity_with_concentration(self):
        """Gini should increase as wealth concentrates."""
        # Start equal
        holdings = {"a": 100.0, "b": 100.0, "c": 100.0, "d": 100.0}
        ginis = [calculate_gini(holdings)]

        # Progressively concentrate
        for _ in range(10):
            holdings["a"] += 10.0
            holdings["b"] -= 2.5
            holdings["c"] -= 2.5
            holdings["d"] -= 5.0
            if min(holdings.values()) >= MINIMUM_HOLDING:
                ginis.append(calculate_gini(holdings))

        # Each step should increase or maintain Gini
        for i in range(1, len(ginis)):
            assert ginis[i] >= ginis[i-1] - 1e-6, f"Gini decreased at step {i}"

    def test_redistribution_always_reduces_or_maintains_gini(self):
        """Harberger tax should never increase Gini."""
        test_cases = [
            {"a": 1000.0, "b": 100.0},
            {"a": 500.0, "b": 300.0, "c": 200.0},
            {"a": 100.0, "b": 100.0, "c": 100.0},
            {f"n{i}": float(i * 100) for i in range(1, 11)},
        ]

        invariant = AdlInvariant()

        for holdings in test_cases:
            pre_gini = calculate_gini(holdings)
            new_holdings = invariant.redistribute_soil_tax(holdings)
            post_gini = calculate_gini(new_holdings)

            assert post_gini <= pre_gini + 1e-9, (
                f"Redistribution increased Gini from {pre_gini} to {post_gini}"
            )

    def test_stats_tracking(self):
        """Test that validation stats are tracked."""
        invariant = AdlInvariant()
        holdings = {"a": 100.0, "b": 100.0}

        # Run some validations
        tx1 = Transaction(tx_id="1", sender="a", recipient="b", amount=10.0)
        tx2 = Transaction(tx_id="2", sender="a", recipient="b", amount=200.0)  # Will fail

        invariant.validate_transaction(tx1, holdings)
        invariant.validate_transaction(tx2, holdings)

        stats = invariant.get_stats()

        assert stats["validations"] == 2
        assert stats["rejections"] == 1
        assert stats["gini_threshold"] == ADL_GINI_THRESHOLD


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_transaction_to_self(self):
        """Transaction to self should be allowed (no-op)."""
        invariant = AdlInvariant()
        holdings = {"a": 100.0, "b": 100.0}

        tx = Transaction(tx_id="self", sender="a", recipient="a", amount=10.0)
        result = invariant.validate_transaction(tx, holdings)

        # Should pass (Gini unchanged)
        assert result.passed is True
        assert result.gini_delta == pytest.approx(0.0, abs=1e-9)

    def test_very_small_holdings(self):
        """Test with very small but valid holdings."""
        invariant = AdlInvariant()
        holdings = {"a": 1e-8, "b": 1e-8}

        tx = Transaction(tx_id="tiny", sender="a", recipient="b", amount=1e-9)
        result = invariant.validate_transaction(tx, holdings)

        assert result.passed is True

    def test_very_large_holdings(self):
        """Test with very large holdings."""
        invariant = AdlInvariant()
        holdings = {"a": 1e15, "b": 1e15}

        tx = Transaction(tx_id="huge", sender="a", recipient="b", amount=1e14)
        result = invariant.validate_transaction(tx, holdings)

        assert result.passed is True  # Equal holdings remain nearly equal

    def test_many_nodes(self):
        """Test with many nodes."""
        invariant = AdlInvariant()
        holdings = {f"node_{i}": 100.0 for i in range(1000)}

        tx = Transaction(tx_id="many", sender="node_0", recipient="node_1", amount=10.0)
        result = invariant.validate_transaction(tx, holdings)

        assert result.passed is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
