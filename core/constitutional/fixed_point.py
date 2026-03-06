"""
Fixed-Point Arithmetic Kernel
═════════════════════════════

Deterministic integer arithmetic producing byte-identical results
on ARM Cortex-M, x86_64, and RISC-V. No floating-point. No IEEE 754
rounding. No platform drift.

Layer 0 of the constitutional kernel: if the math drifts, consensus
breaks, and sovereignty is lost.

Standing on Giants:
- Al-Khwarizmi (780-850): "Every truth is a procedure"
- Knuth (1997): Fixed-point arithmetic in TAoCP Vol.2
"""

from __future__ import annotations

from core.integration.constants import (
    ADL_GINI_THRESHOLD,
    UNIFIED_IHSAN_THRESHOLD,
)

# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════

FP_PRECISION: int = 1_000_000  # 6 decimal places
FP_ONE: int = FP_PRECISION  # 1.0 in fixed-point
FP_ZERO: int = 0  # 0.0 in fixed-point
FP_MAX: int = (2**63) - 1  # Maximum representable value


# ═══════════════════════════════════════════════════════════════════
# Core Conversions
# ═══════════════════════════════════════════════════════════════════


def fp(value: float) -> int:
    """Convert float to fixed-point integer.

    >>> fp(1.0)
    1000000
    >>> fp(0.95)
    950000
    >>> fp(3.27)
    3270000
    """
    return round(value * FP_PRECISION)


def fp_float(fp_value: int) -> float:
    """Convert fixed-point back to float.

    WARNING: Use for display/logging only, NEVER for computation.

    >>> fp_float(950000)
    0.95
    """
    return fp_value / FP_PRECISION


# ═══════════════════════════════════════════════════════════════════
# Arithmetic Operations (all integer, all deterministic)
# ═══════════════════════════════════════════════════════════════════


def fp_add(a: int, b: int) -> int:
    """Fixed-point addition with overflow guard.

    >>> fp_add(fp(1.5), fp(2.5))
    4000000
    """
    result = a + b
    if result > FP_MAX:
        raise OverflowError(f"Fixed-point overflow: {a} + {b} = {result} > FP_MAX")
    return result


def fp_sub(a: int, b: int) -> int:
    """Fixed-point subtraction. Returns 0 if result would be negative.

    Constitutional rule: balances cannot go negative.

    >>> fp_sub(fp(5.0), fp(3.0))
    2000000
    >>> fp_sub(fp(1.0), fp(5.0))
    0
    """
    result = a - b
    if result < 0:
        return 0
    return result


def fp_mul(a: int, b: int) -> int:
    """Fixed-point multiplication with precision correction.

    (a * b) / PRECISION avoids double-scaling.
    Integer division — deterministic truncation, not rounding.

    >>> fp_mul(fp(0.5), fp(0.5))
    250000
    >>> fp_mul(fp(3.27), fp(1.0))
    3270000
    """
    return (a * b) // FP_PRECISION


def fp_div(a: int, b: int) -> int:
    """Fixed-point division with precision correction.

    (a * PRECISION) / b scales before dividing to preserve precision.
    Constitutional rule: division by zero returns 0, not exception.

    >>> fp_div(fp(1.0), fp(2.0))
    500000
    >>> fp_div(fp(100.0), 0)
    0
    """
    if b == 0:
        return 0
    return (a * FP_PRECISION) // b


def fp_clamp(value: int, min_val: int, max_val: int) -> int:
    """Clamp value within [min_val, max_val].

    >>> fp_clamp(fp(5.0), fp(0.0), fp(1.0))
    1000000
    >>> fp_clamp(fp(0.5), fp(0.0), fp(1.0))
    500000
    """
    if value < min_val:
        return min_val
    if value > max_val:
        return max_val
    return value


# ═══════════════════════════════════════════════════════════════════
# Derived Operations
# ═══════════════════════════════════════════════════════════════════


def fp_weighted_avg(values: list[int], weights: list[int]) -> int:
    """Weighted average in fixed-point.

    sum(v_i * w_i) / sum(w_i), all in integer arithmetic.

    >>> vals = [fp(1.0), fp(2.0), fp(3.0)]
    >>> wts = [fp(0.5), fp(0.3), fp(0.2)]
    >>> abs(fp_weighted_avg(vals, wts) - fp(1.7)) <= 1
    True
    """
    if len(values) != len(weights):
        raise ValueError(
            f"values ({len(values)}) and weights ({len(weights)}) must match"
        )

    numerator = 0
    denominator = 0
    for v, w in zip(values, weights):
        numerator += v * w
        denominator += w

    if denominator == 0:
        return FP_ZERO
    return numerator // denominator


def fp_percentage(value: int, percent: int) -> int:
    """Compute value * percent / 100 in fixed-point.

    percent is in fixed-point (e.g., fp(2.5) for 2.5%).

    >>> fp_percentage(fp(1000.0), fp(2.5))  # 2.5% of 1000 = 25
    25000000
    """
    return fp_mul(value, fp_div(percent, fp(100.0)))


# ═══════════════════════════════════════════════════════════════════
# Constitutional threshold accessors (from constants.py)
# ═══════════════════════════════════════════════════════════════════


def fp_ihsan_floor() -> int:
    """Return IHSAN floor as fixed-point (from constants.py)."""
    return fp(UNIFIED_IHSAN_THRESHOLD)


def fp_gini_threshold() -> int:
    """Return ADL Gini threshold as fixed-point (from constants.py)."""
    return fp(ADL_GINI_THRESHOLD)
