"""
Tests for core.rdve.stability — Warmup Scheduling, Convergence Detection,
and Stability Protocol.

Covers:
    - WarmupStrategy enum membership
    - WarmupSchedule.get_exploration_factor for all 4 strategies
    - Boundary conditions: cycle 0, mid-warmup, at/beyond warmup_cycles
    - WarmupSchedule.get_snr_threshold warmup premium and decay
    - ConvergenceDetector.record, is_converged, get_trend
    - Convergence via plateau and via low-improvement window
    - StabilityProtocol exponential backoff on failures
    - StabilityProtocol.should_stop on convergence/divergence/backoff
"""

import math

import pytest

from core.integration.constants import UNIFIED_SNR_THRESHOLD
from core.rdve.stability import (
    ConvergenceDetector,
    StabilityProtocol,
    WarmupSchedule,
    WarmupStrategy,
)


# ============================================================================
# WarmupStrategy Enum
# ============================================================================


class TestWarmupStrategy:
    def test_enum_has_four_members(self):
        assert len(WarmupStrategy) == 4

    def test_enum_values(self):
        assert WarmupStrategy.LINEAR.value == "linear"
        assert WarmupStrategy.COSINE.value == "cosine"
        assert WarmupStrategy.EXPONENTIAL.value == "exponential"
        assert WarmupStrategy.ADAPTIVE.value == "adaptive"


# ============================================================================
# WarmupSchedule — LINEAR strategy
# ============================================================================


class TestWarmupLinear:
    """Linear warmup: factor = min + (max - min) * (cycle / warmup_cycles)."""

    def setup_method(self):
        self.schedule = WarmupSchedule(
            warmup_cycles=5,
            strategy=WarmupStrategy.LINEAR,
            min_exploration_factor=0.3,
            max_exploration_factor=1.0,
        )

    def test_cycle_zero_returns_min_factor(self):
        factor = self.schedule.get_exploration_factor(0)
        assert factor == pytest.approx(0.3, abs=1e-9)

    def test_mid_warmup_returns_interpolated(self):
        # cycle=2, progress=0.4, factor = 0.3 + 0.7*0.4 = 0.58
        factor = self.schedule.get_exploration_factor(2)
        assert factor == pytest.approx(0.3 + 0.7 * (2 / 5), abs=1e-6)

    def test_at_warmup_cycles_returns_max_factor(self):
        factor = self.schedule.get_exploration_factor(5)
        assert factor == pytest.approx(1.0, abs=1e-9)

    def test_beyond_warmup_returns_max_factor(self):
        factor = self.schedule.get_exploration_factor(100)
        assert factor == pytest.approx(1.0, abs=1e-9)


# ============================================================================
# WarmupSchedule — COSINE strategy
# ============================================================================


class TestWarmupCosine:
    """Cosine annealing: factor = min + (max-min) * (1 - cos(pi*progress)) / 2."""

    def setup_method(self):
        self.schedule = WarmupSchedule(
            warmup_cycles=5,
            strategy=WarmupStrategy.COSINE,
            min_exploration_factor=0.3,
            max_exploration_factor=1.0,
        )

    def test_cycle_zero_returns_min_factor(self):
        # cos(0) = 1 => (1 - 1)/2 = 0 => factor = 0.3
        factor = self.schedule.get_exploration_factor(0)
        assert factor == pytest.approx(0.3, abs=1e-9)

    def test_mid_warmup_cosine_value(self):
        # cycle=2, progress=0.4
        progress = 2 / 5
        expected = 0.3 + 0.7 * (1 - math.cos(math.pi * progress)) / 2
        factor = self.schedule.get_exploration_factor(2)
        assert factor == pytest.approx(expected, abs=1e-6)

    def test_at_warmup_cycles_returns_max(self):
        factor = self.schedule.get_exploration_factor(5)
        assert factor == pytest.approx(1.0, abs=1e-9)


# ============================================================================
# WarmupSchedule — EXPONENTIAL strategy
# ============================================================================


class TestWarmupExponential:
    """Exponential: factor = min + (max-min) * (1 - exp(-3*progress))."""

    def setup_method(self):
        self.schedule = WarmupSchedule(
            warmup_cycles=5,
            strategy=WarmupStrategy.EXPONENTIAL,
            min_exploration_factor=0.3,
            max_exploration_factor=1.0,
        )

    def test_cycle_zero_returns_min_factor(self):
        # exp(0) = 1 => 1-1 = 0 => factor = 0.3
        factor = self.schedule.get_exploration_factor(0)
        assert factor == pytest.approx(0.3, abs=1e-9)

    def test_mid_warmup_exponential_value(self):
        progress = 3 / 5
        expected = 0.3 + 0.7 * (1 - math.exp(-3 * progress))
        factor = self.schedule.get_exploration_factor(3)
        assert factor == pytest.approx(expected, abs=1e-6)

    def test_at_warmup_cycles_returns_max(self):
        factor = self.schedule.get_exploration_factor(5)
        assert factor == pytest.approx(1.0, abs=1e-9)


# ============================================================================
# WarmupSchedule — ADAPTIVE strategy
# ============================================================================


class TestWarmupAdaptive:
    """Adaptive warmup adjusts based on success_rate."""

    def setup_method(self):
        self.schedule = WarmupSchedule(
            warmup_cycles=5,
            strategy=WarmupStrategy.ADAPTIVE,
            min_exploration_factor=0.3,
            max_exploration_factor=1.0,
            early_success_threshold=0.5,
            early_failure_threshold=0.8,
        )

    def test_high_success_accelerates_warmup(self):
        # success_rate=0.9 > 0.5 => accelerated: min + progress*2
        factor = self.schedule.get_exploration_factor(2, success_rate=0.9)
        progress = 2 / 5
        expected = min(1.0, 0.3 + progress * 2)
        assert factor == pytest.approx(expected, abs=1e-6)

    def test_low_success_slows_warmup(self):
        # success_rate=0.1 < (1 - 0.8) = 0.2 => slow: min + progress*0.5
        factor = self.schedule.get_exploration_factor(2, success_rate=0.1)
        progress = 2 / 5
        expected = 0.3 + progress * 0.5
        assert factor == pytest.approx(expected, abs=1e-6)

    def test_normal_success_uses_linear(self):
        # success_rate=0.4, not > 0.5 and not < 0.2 => linear
        factor = self.schedule.get_exploration_factor(2, success_rate=0.4)
        progress = 2 / 5
        expected = 0.3 + 0.7 * progress
        assert factor == pytest.approx(expected, abs=1e-6)

    def test_cycle_zero_always_returns_min(self):
        factor = self.schedule.get_exploration_factor(0, success_rate=0.9)
        assert factor == pytest.approx(0.3, abs=1e-9)


# ============================================================================
# WarmupSchedule — SNR threshold during warmup
# ============================================================================


class TestWarmupSNRThreshold:
    def setup_method(self):
        self.schedule = WarmupSchedule(warmup_cycles=5)

    def test_cycle_zero_has_highest_snr_premium(self):
        threshold = self.schedule.get_snr_threshold(0)
        # Start 10% above base
        expected = min(1.0, UNIFIED_SNR_THRESHOLD + 0.10)
        assert threshold == pytest.approx(expected, abs=1e-6)

    def test_at_warmup_cycles_returns_base_threshold(self):
        threshold = self.schedule.get_snr_threshold(5)
        assert threshold == pytest.approx(UNIFIED_SNR_THRESHOLD, abs=1e-9)

    def test_mid_warmup_decays_linearly(self):
        threshold = self.schedule.get_snr_threshold(2)
        progress = 2 / 5
        warmup_premium = 0.10 * (1 - progress)
        expected = min(1.0, UNIFIED_SNR_THRESHOLD + warmup_premium)
        assert threshold == pytest.approx(expected, abs=1e-6)

    def test_is_warmup_complete(self):
        assert not self.schedule.is_warmup_complete(0)
        assert not self.schedule.is_warmup_complete(4)
        assert self.schedule.is_warmup_complete(5)
        assert self.schedule.is_warmup_complete(10)


# ============================================================================
# ConvergenceDetector
# ============================================================================


class TestConvergenceDetector:
    def test_not_converged_with_insufficient_data(self):
        cd = ConvergenceDetector(window_size=5)
        cd.record(0.5)
        cd.record(0.6)
        assert cd.is_converged() is False

    def test_converged_when_recent_improvements_below_threshold(self):
        cd = ConvergenceDetector(window_size=3, min_improvement=0.05)
        # Record scores where all recent ones are below min_improvement
        cd.record(0.01)
        cd.record(0.02)
        cd.record(0.01)
        assert cd.is_converged() is True

    def test_not_converged_with_strong_improvements(self):
        cd = ConvergenceDetector(window_size=3, min_improvement=0.01)
        cd.record(0.5)
        cd.record(0.6)
        cd.record(0.7)
        assert cd.is_converged() is False

    def test_converged_via_extended_plateau(self):
        cd = ConvergenceDetector(
            window_size=3,
            min_improvement=0.001,
            max_plateau_cycles=5,
        )
        # Record initial best, then many non-improvements
        cd.record(1.0)
        for _ in range(6):
            cd.record(0.5)  # Never beats 1.0, plateau_count rises
        assert cd.is_converged() is True

    def test_divergence_detection(self):
        cd = ConvergenceDetector(window_size=5)
        cd.record(1.0)
        # Each drops more than 5% from previous => divergence_count increments
        cd.record(0.90)
        cd.record(0.80)
        cd.record(0.70)
        assert cd.is_diverging() is True

    def test_get_trend_improving(self):
        cd = ConvergenceDetector(window_size=3, max_plateau_cycles=100)
        cd.record(0.3)
        cd.record(0.5)
        cd.record(0.7)
        assert cd.get_trend() == "improving"

    def test_get_trend_insufficient_data(self):
        cd = ConvergenceDetector(window_size=5)
        cd.record(0.5)
        assert cd.get_trend() == "insufficient_data"

    def test_get_status_returns_dict(self):
        cd = ConvergenceDetector()
        cd.record(0.5)
        cd.record(0.6)
        status = cd.get_status()
        assert "total_cycles" in status
        assert "best_score" in status
        assert status["total_cycles"] == 2
        assert status["best_score"] == pytest.approx(0.6)


# ============================================================================
# StabilityProtocol — Exponential Backoff
# ============================================================================


class TestStabilityProtocol:
    def test_backoff_starts_at_zero(self):
        sp = StabilityProtocol()
        assert sp._current_backoff == 0.0

    def test_first_failure_sets_backoff_to_one(self):
        sp = StabilityProtocol()
        sp.post_cycle(score=0.1, success=False)
        assert sp._current_backoff == 1.0

    def test_consecutive_failures_double_backoff(self):
        sp = StabilityProtocol(backoff_factor=2.0)
        sp.post_cycle(score=0.1, success=False)
        assert sp._current_backoff == 1.0
        sp.post_cycle(score=0.1, success=False)
        assert sp._current_backoff == 2.0
        sp.post_cycle(score=0.1, success=False)
        assert sp._current_backoff == 4.0

    def test_backoff_capped_at_max(self):
        sp = StabilityProtocol(backoff_factor=2.0, max_backoff_seconds=10.0)
        for _ in range(20):
            sp.post_cycle(score=0.1, success=False)
        assert sp._current_backoff == 10.0

    def test_success_resets_backoff(self):
        sp = StabilityProtocol()
        sp.post_cycle(score=0.1, success=False)
        sp.post_cycle(score=0.1, success=False)
        assert sp._current_backoff > 0
        sp.post_cycle(score=0.5, success=True)
        assert sp._current_backoff == 0.0

    def test_should_stop_on_convergence(self):
        sp = StabilityProtocol(
            convergence=ConvergenceDetector(
                window_size=3,
                min_improvement=0.05,
            )
        )
        # Feed low scores to trigger convergence
        sp.post_cycle(score=0.01, success=True)
        sp.post_cycle(score=0.01, success=True)
        sp.post_cycle(score=0.01, success=True)
        should_stop, reason = sp.should_stop()
        assert should_stop is True
        assert reason == "converged"

    def test_should_stop_on_max_backoff(self):
        sp = StabilityProtocol(
            backoff_factor=2.0,
            max_backoff_seconds=8.0,
        )
        # Cause enough failures to hit max backoff
        for _ in range(10):
            sp.post_cycle(score=0.1, success=False)
        should_stop, reason = sp.should_stop()
        assert should_stop is True
        assert reason == "max_backoff_reached"

    def test_should_continue_when_healthy(self):
        sp = StabilityProtocol()
        sp.post_cycle(score=0.5, success=True)
        should_stop, reason = sp.should_stop()
        assert should_stop is False
        assert reason == "continue"

    def test_get_status_contains_expected_keys(self):
        sp = StabilityProtocol()
        status = sp.get_status()
        assert "warmup" in status
        assert "convergence" in status
        assert "rate_limiting" in status
        assert "cycle_count" in status
        assert "success_rate" in status
