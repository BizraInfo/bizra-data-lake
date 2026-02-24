"""
Tests for Logistic Emission Decay — Gini-Gated Token Minting
"""

import math

import pytest

from core.token.emission_decay import (
    DEFAULT_EMISSION_MAX,
    DEFAULT_GINI_TARGET,
    DEFAULT_STEEPNESS,
    LogisticEmissionGate,
    compute_logistic_emission,
)


class TestLogisticEmission:
    """Test the logistic emission decay function."""

    def test_at_target_gini(self):
        """At G_target, emission should be E_max / 2."""
        rate = compute_logistic_emission(
            gini=DEFAULT_GINI_TARGET,
            e_max=DEFAULT_EMISSION_MAX,
            g_target=DEFAULT_GINI_TARGET,
            steepness=DEFAULT_STEEPNESS,
        )
        expected = DEFAULT_EMISSION_MAX / 2.0
        assert abs(rate - expected) < 0.01

    def test_low_gini_high_emission(self):
        """When Gini is well below target, emission should be near E_max."""
        rate = compute_logistic_emission(gini=0.10, e_max=1000.0)
        assert rate > 900.0

    def test_high_gini_low_emission(self):
        """When Gini exceeds target, emission should drop sharply."""
        rate = compute_logistic_emission(gini=0.50, e_max=1000.0)
        assert rate < 100.0

    def test_monotonic_decrease(self):
        """Emission should monotonically decrease as Gini increases."""
        gini_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
        rates = [compute_logistic_emission(g) for g in gini_values]

        for i in range(len(rates) - 1):
            assert rates[i] >= rates[i + 1], (
                f"Not monotonic: emission({gini_values[i]})={rates[i]} "
                f"< emission({gini_values[i+1]})={rates[i+1]}"
            )

    def test_emission_always_positive(self):
        """Emission should always be positive (logistic never reaches 0)."""
        for gini in [0.0, 0.35, 0.50, 0.80, 1.0]:
            rate = compute_logistic_emission(gini)
            assert rate > 0

    def test_emission_bounded_by_e_max(self):
        """Emission should never exceed E_max."""
        for gini in [0.0, 0.01, 0.05, 0.10]:
            rate = compute_logistic_emission(gini, e_max=500.0)
            assert rate <= 500.0

    def test_emission_curve_against_analytical(self):
        """Verify emission matches the analytical logistic function."""
        e_max = 1000.0
        g_target = 0.35
        k = 20.0

        for gini in [0.10, 0.20, 0.30, 0.35, 0.40, 0.50]:
            expected = e_max / (1.0 + math.exp(k * (gini - g_target)))
            actual = compute_logistic_emission(
                gini=gini, e_max=e_max, g_target=g_target, steepness=k
            )
            assert (
                abs(actual - expected) < 0.001
            ), f"At gini={gini}: expected={expected}, actual={actual}"


class TestLogisticEmissionGate:
    """Test the emission gate that applies to token minting."""

    def test_gate_with_low_gini(self):
        gate = LogisticEmissionGate()
        result = gate.compute_gated_emission(
            requested_amount=1000.0,
            current_holdings=[10.0, 10.0, 10.0, 10.0],
        )

        assert result["gated_amount"] > 0
        assert result["gini"] < DEFAULT_GINI_TARGET
        assert result["gate_open"]

    def test_gate_with_high_gini(self):
        gate = LogisticEmissionGate()
        # Extremely unequal distribution
        result = gate.compute_gated_emission(
            requested_amount=1000.0,
            current_holdings=[1000.0, 1.0, 1.0, 1.0],
        )

        assert result["gated_amount"] < 1000.0
        assert result["gini"] > DEFAULT_GINI_TARGET

    def test_gate_with_empty_holdings(self):
        gate = LogisticEmissionGate()
        result = gate.compute_gated_emission(
            requested_amount=100.0,
            current_holdings=[],
        )
        # No holdings = Gini 0, gate fully open
        assert result["gated_amount"] == 100.0
        assert result["gate_open"]

    def test_gini_rising_causes_emission_drop(self):
        """SAPE spec: simulate Gini rising from 0.20 to 0.40, verify emission drops."""
        gate = LogisticEmissionGate(e_max=1000.0)

        rates = []
        for gini in [0.20, 0.25, 0.30, 0.35, 0.40]:
            rate = compute_logistic_emission(gini, e_max=1000.0)
            rates.append(rate)

        # Each subsequent rate should be lower
        for i in range(len(rates) - 1):
            assert (
                rates[i] > rates[i + 1]
            ), f"Emission should decrease: {rates[i]} -> {rates[i+1]}"

        # The drop from 0.20 to 0.40 should be substantial
        drop_pct = (rates[0] - rates[-1]) / rates[0]
        assert drop_pct > 0.5, f"Emission drop {drop_pct:.1%} should be > 50%"
