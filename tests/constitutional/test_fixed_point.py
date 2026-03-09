"""
Tests for the Fixed-Point Arithmetic Kernel
════════════════════════════════════════════

TDD anchors from Phase 67.01 specification.
Every test here MUST pass before any algorithm implementation proceeds.

Standing on Giants:
- Beck (2002): Test-Driven Development by Example
- Al-Khwarizmi (780-850): Deterministic procedures
"""

from __future__ import annotations

import pytest

from core.constitutional.fixed_point import (
    FP_MAX,
    FP_ONE,
    FP_PRECISION,
    FP_ZERO,
    fp,
    fp_add,
    fp_clamp,
    fp_div,
    fp_float,
    fp_mul,
    fp_percentage,
    fp_sub,
    fp_weighted_avg,
)

# ═══════════════════════════════════════════════════════════════════
# Conversions
# ═══════════════════════════════════════════════════════════════════


class TestFpConversions:
    """Float ↔ fixed-point roundtrip tests."""

    def test_fp_zero(self) -> None:
        assert fp(0.0) == 0
        assert fp_float(0) == 0.0

    def test_fp_one(self) -> None:
        assert fp(1.0) == FP_PRECISION
        assert fp(1.0) == FP_ONE
        assert fp_float(FP_ONE) == 1.0

    def test_fp_fractional_roundtrip(self) -> None:
        """6-decimal precision preserved."""
        assert fp_float(fp(3.141592)) == 3.141592
        assert fp_float(fp(0.950000)) == 0.95
        assert fp_float(fp(0.025000)) == 0.025

    def test_fp_ihsan_threshold(self) -> None:
        """0.95 converts correctly."""
        assert fp(0.95) == 950_000

    def test_fp_gini_threshold(self) -> None:
        """0.35 converts correctly."""
        assert fp(0.35) == 350_000

    def test_fp_equity_factor(self) -> None:
        """3.27 converts correctly (newcomer advantage)."""
        assert fp(3.27) == 3_270_000

    def test_fp_constants(self) -> None:
        assert FP_PRECISION == 1_000_000
        assert FP_ZERO == 0
        assert FP_ONE == 1_000_000
        assert FP_MAX == (2**63) - 1


# ═══════════════════════════════════════════════════════════════════
# Arithmetic
# ═══════════════════════════════════════════════════════════════════


class TestFpArithmetic:
    """Core arithmetic operations."""

    def test_fp_add_basic(self) -> None:
        assert fp_add(fp(1.5), fp(2.5)) == fp(4.0)

    def test_fp_add_zero_identity(self) -> None:
        assert fp_add(fp(42.0), FP_ZERO) == fp(42.0)

    def test_fp_add_overflow_guard(self) -> None:
        with pytest.raises(OverflowError):
            fp_add(FP_MAX, 1)

    def test_fp_sub_basic(self) -> None:
        assert fp_sub(fp(5.0), fp(3.0)) == fp(2.0)

    def test_fp_sub_underflow_returns_zero(self) -> None:
        """Constitutional rule: balances cannot go negative."""
        assert fp_sub(fp(1.0), fp(5.0)) == 0

    def test_fp_sub_zero_identity(self) -> None:
        assert fp_sub(fp(42.0), FP_ZERO) == fp(42.0)

    def test_fp_mul_half_times_half(self) -> None:
        """0.5 * 0.5 = 0.25"""
        assert fp_mul(fp(0.5), fp(0.5)) == fp(0.25)

    def test_fp_mul_identity(self) -> None:
        """x * 1.0 = x"""
        assert fp_mul(fp(3.27), FP_ONE) == fp(3.27)

    def test_fp_mul_by_zero(self) -> None:
        assert fp_mul(fp(999.0), FP_ZERO) == 0

    def test_fp_mul_commutative(self) -> None:
        a, b = fp(1.23), fp(4.56)
        assert fp_mul(a, b) == fp_mul(b, a)

    def test_fp_div_basic(self) -> None:
        """1.0 / 2.0 = 0.5"""
        assert fp_div(fp(1.0), fp(2.0)) == fp(0.5)

    def test_fp_div_by_zero_returns_zero(self) -> None:
        """Constitutional rule: div-by-zero → 0, not exception."""
        assert fp_div(fp(100.0), 0) == 0

    def test_fp_div_precision(self) -> None:
        """1.0 / 3.0 ≈ 0.333333"""
        result = fp_div(fp(1.0), fp(3.0))
        assert abs(fp_float(result) - 0.333333) < 0.000002

    def test_fp_div_identity(self) -> None:
        """x / 1.0 = x"""
        assert fp_div(fp(42.0), FP_ONE) == fp(42.0)


# ═══════════════════════════════════════════════════════════════════
# Clamping
# ═══════════════════════════════════════════════════════════════════


class TestFpClamp:
    """Boundary enforcement."""

    def test_clamp_below(self) -> None:
        assert fp_clamp(fp(-1.0), FP_ZERO, FP_ONE) == FP_ZERO

    def test_clamp_above(self) -> None:
        assert fp_clamp(fp(5.0), FP_ZERO, FP_ONE) == FP_ONE

    def test_clamp_within(self) -> None:
        assert fp_clamp(fp(0.5), FP_ZERO, FP_ONE) == fp(0.5)

    def test_clamp_at_boundary(self) -> None:
        assert fp_clamp(FP_ZERO, FP_ZERO, FP_ONE) == FP_ZERO
        assert fp_clamp(FP_ONE, FP_ZERO, FP_ONE) == FP_ONE


# ═══════════════════════════════════════════════════════════════════
# Derived Operations
# ═══════════════════════════════════════════════════════════════════


class TestFpDerived:
    """Weighted average, percentage."""

    def test_weighted_avg_uniform(self) -> None:
        """Equal weights → arithmetic mean."""
        values = [fp(1.0), fp(2.0), fp(3.0)]
        weights = [FP_ONE, FP_ONE, FP_ONE]
        result = fp_weighted_avg(values, weights)
        assert abs(result - fp(2.0)) <= 1  # ±1 LSB tolerance

    def test_weighted_avg_skewed(self) -> None:
        """Weighted toward first value."""
        values = [fp(1.0), fp(2.0), fp(3.0)]
        weights = [fp(0.5), fp(0.3), fp(0.2)]
        result = fp_weighted_avg(values, weights)
        # 0.5*1 + 0.3*2 + 0.2*3 = 0.5 + 0.6 + 0.6 = 1.7
        assert abs(result - fp(1.7)) <= 1

    def test_weighted_avg_zero_weights(self) -> None:
        """All-zero weights → zero."""
        values = [fp(1.0), fp(2.0)]
        weights = [0, 0]
        assert fp_weighted_avg(values, weights) == FP_ZERO

    def test_weighted_avg_length_mismatch(self) -> None:
        with pytest.raises(ValueError):
            fp_weighted_avg([fp(1.0)], [fp(1.0), fp(2.0)])

    def test_percentage_zakat(self) -> None:
        """2.5% of 1000 = 25."""
        result = fp_percentage(fp(1000.0), fp(2.5))
        assert abs(fp_float(result) - 25.0) < 0.01


# ═══════════════════════════════════════════════════════════════════
# Determinism
# ═══════════════════════════════════════════════════════════════════


class TestFpDeterminism:
    """The whole point: same inputs → same outputs, always."""

    def test_1000_iterations_identical(self) -> None:
        """Run fp_mul 1000 times — must produce identical result each time."""
        a, b = fp(0.123456), fp(0.654321)
        expected = fp_mul(a, b)
        for _ in range(1000):
            assert fp_mul(a, b) == expected

    def test_order_of_operations_deterministic(self) -> None:
        """Multi-step computation produces same result regardless of run."""
        # Simulate a mini ihsan score computation
        intent = fp(0.95)
        efficiency = fp(0.90)
        impact = fp(0.92)
        repro = fp(0.88)

        w_i, w_e, w_im, w_r = fp(0.25), fp(0.25), fp(0.30), fp(0.20)

        score = (
            fp_mul(w_i, intent)
            + fp_mul(w_e, efficiency)
            + fp_mul(w_im, impact)
            + fp_mul(w_r, repro)
        )

        # Run it 100 times
        for _ in range(100):
            check = (
                fp_mul(w_i, intent)
                + fp_mul(w_e, efficiency)
                + fp_mul(w_im, impact)
                + fp_mul(w_r, repro)
            )
            assert check == score

    def test_associativity_integer_addition(self) -> None:
        """Integer addition is associative — verify no float contamination."""
        a, b, c = fp(1.1), fp(2.2), fp(3.3)
        assert fp_add(fp_add(a, b), c) == fp_add(a, fp_add(b, c))


# ═══════════════════════════════════════════════════════════════════
# Constitutional Threshold Accessors
# ═══════════════════════════════════════════════════════════════════


class TestFpThresholds:
    """Verify thresholds are sourced from constants.py."""

    def test_ihsan_floor_from_constants(self) -> None:
        from core.constitutional.fixed_point import fp_ihsan_floor

        assert fp_ihsan_floor() == fp(0.95)

    def test_gini_threshold_from_constants(self) -> None:
        from core.constitutional.fixed_point import fp_gini_threshold

        assert fp_gini_threshold() == fp(0.35)
