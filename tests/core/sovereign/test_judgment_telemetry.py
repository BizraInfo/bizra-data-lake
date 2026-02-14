"""
Tests for the Self-Evolving Judgment Engine (SJE) — Observation Telemetry.

Covers: verdict observation, entropy computation, stability check,
dominant verdict detection, and epoch distribution simulation.
"""

import math

import pytest

from core.sovereign.judgment_telemetry import (
    JudgmentTelemetry,
    JudgmentVerdict,
    simulate_epoch_distribution,
)


# ═══════════════════════════════════════════════════════════════════════════════
# JudgmentTelemetry Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestJudgmentTelemetry:
    """Test observation-mode telemetry."""

    def test_empty_telemetry(self):
        t = JudgmentTelemetry()
        assert t.total_observations == 0
        assert t.entropy() == 0.0
        assert t.dominant_verdict() is None
        assert t.is_stable()

    def test_single_observation(self):
        t = JudgmentTelemetry()
        t.observe(JudgmentVerdict.PROMOTE)
        assert t.total_observations == 1
        assert t.entropy() == 0.0  # Single obs = 0 entropy

    def test_observe_increments(self):
        t = JudgmentTelemetry()
        t.observe(JudgmentVerdict.PROMOTE)
        t.observe(JudgmentVerdict.PROMOTE)
        t.observe(JudgmentVerdict.DEMOTE)
        assert t.total_observations == 3
        assert t.verdict_counts[JudgmentVerdict.PROMOTE] == 2
        assert t.verdict_counts[JudgmentVerdict.DEMOTE] == 1

    def test_entropy_uniform(self):
        """Uniform distribution: H = log2(4) = 2.0."""
        t = JudgmentTelemetry()
        for _ in range(25):
            t.observe(JudgmentVerdict.PROMOTE)
            t.observe(JudgmentVerdict.NEUTRAL)
            t.observe(JudgmentVerdict.DEMOTE)
            t.observe(JudgmentVerdict.FORBID)
        assert abs(t.entropy() - 2.0) < 0.01

    def test_entropy_dominant(self):
        """Heavily skewed: low entropy."""
        t = JudgmentTelemetry()
        for _ in range(95):
            t.observe(JudgmentVerdict.PROMOTE)
        for _ in range(5):
            t.observe(JudgmentVerdict.NEUTRAL)
        assert t.entropy() < 0.5
        assert t.is_stable()

    def test_entropy_two_equal(self):
        """Two equal verdicts: H = log2(2) = 1.0."""
        t = JudgmentTelemetry()
        for _ in range(50):
            t.observe(JudgmentVerdict.PROMOTE)
            t.observe(JudgmentVerdict.DEMOTE)
        assert abs(t.entropy() - 1.0) < 0.01

    def test_dominant_verdict(self):
        t = JudgmentTelemetry()
        t.observe(JudgmentVerdict.PROMOTE)
        t.observe(JudgmentVerdict.PROMOTE)
        t.observe(JudgmentVerdict.DEMOTE)
        assert t.dominant_verdict() == JudgmentVerdict.PROMOTE

    def test_is_stable_high_entropy(self):
        """Uniform distribution should not be stable."""
        t = JudgmentTelemetry()
        for _ in range(25):
            t.observe(JudgmentVerdict.PROMOTE)
            t.observe(JudgmentVerdict.NEUTRAL)
            t.observe(JudgmentVerdict.DEMOTE)
            t.observe(JudgmentVerdict.FORBID)
        assert not t.is_stable()

    def test_distribution(self):
        t = JudgmentTelemetry()
        for _ in range(80):
            t.observe(JudgmentVerdict.PROMOTE)
        for _ in range(20):
            t.observe(JudgmentVerdict.NEUTRAL)

        dist = t.distribution()
        assert abs(dist["promote"] - 80.0) < 0.01
        assert abs(dist["neutral"] - 20.0) < 0.01
        assert abs(dist["demote"] - 0.0) < 0.01
        assert abs(dist["forbid"] - 0.0) < 0.01

    def test_to_dict(self):
        t = JudgmentTelemetry()
        t.observe(JudgmentVerdict.PROMOTE)
        d = t.to_dict()
        assert d["total_observations"] == 1
        assert "entropy" in d
        assert d["dominant_verdict"] == "promote"
        assert d["is_stable"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# Epoch Distribution Simulator Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestEpochDistribution:
    """Test genesis economy simulation (no mint)."""

    def test_empty_impacts(self):
        assert simulate_epoch_distribution([], 1000) == []

    def test_zero_total(self):
        assert simulate_epoch_distribution([0, 0, 0], 1000) == [0, 0, 0]

    def test_single_node(self):
        assert simulate_epoch_distribution([100], 1000) == [1000]

    def test_proportional(self):
        result = simulate_epoch_distribution([100, 200, 300], 600)
        assert result == [100, 200, 300]

    def test_rounding_floor(self):
        """1/3 of 100 = 33 (floor division)."""
        result = simulate_epoch_distribution([1, 1, 1], 100)
        assert result == [33, 33, 33]
        assert sum(result) <= 100  # Dust remains

    def test_two_nodes_equal(self):
        result = simulate_epoch_distribution([500, 500], 1000)
        assert result == [500, 500]

    def test_large_epoch_cap(self):
        result = simulate_epoch_distribution([1, 2, 3], 1_000_000)
        assert result[0] < result[1] < result[2]
        assert sum(result) <= 1_000_000

    def test_single_dominant_node(self):
        """One node with all impact should get everything."""
        result = simulate_epoch_distribution([0, 0, 1000, 0], 500)
        assert result == [0, 0, 500, 0]

    def test_zero_epoch_cap(self):
        result = simulate_epoch_distribution([100, 200], 0)
        assert result == [0, 0]
