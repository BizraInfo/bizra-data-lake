"""Tests for core.rollout.metrics — Phase 46 observability metrics.

Standing on Giants: Shannon (information measurement, 1948)
"""

from __future__ import annotations

import math

import pytest

from core.rollout.metrics import Phase46Metrics

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def metrics() -> Phase46Metrics:
    return Phase46Metrics()


# ------------------------------------------------------------------
# Counter tests
# ------------------------------------------------------------------


class TestCounters:
    """Counter increment and retrieval."""

    def test_inc_default_value(self, metrics: Phase46Metrics) -> None:
        metrics.inc("search_requests")
        assert metrics.get_counter("search_requests") == 1

    def test_inc_custom_value(self, metrics: Phase46Metrics) -> None:
        metrics.inc("search_requests", 5)
        metrics.inc("search_requests", 3)
        assert metrics.get_counter("search_requests") == 8

    def test_get_counter_missing_key(self, metrics: Phase46Metrics) -> None:
        assert metrics.get_counter("nonexistent") == 0


# ------------------------------------------------------------------
# Latency percentile tests
# ------------------------------------------------------------------


class TestLatencyPercentiles:
    """Latency recording and percentile computation."""

    def test_latency_percentile_sorted(self, metrics: Phase46Metrics) -> None:
        for v in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            metrics.record_latency("search", float(v))
        p50 = metrics.percentile(metrics._latencies["search"], 50)
        assert 50 <= p50 <= 60  # Midpoint of sorted list

    def test_latency_p95(self, metrics: Phase46Metrics) -> None:
        values = list(range(1, 101))  # 1..100
        for v in values:
            metrics.record_latency("search", float(v))
        p95 = metrics.percentile(metrics._latencies["search"], 95)
        assert 94 <= p95 <= 96

    def test_percentile_empty_returns_zero(self) -> None:
        assert Phase46Metrics.percentile([], 50) == 0.0

    def test_percentile_single_value(self) -> None:
        assert Phase46Metrics.percentile([42.0], 50) == 42.0
        assert Phase46Metrics.percentile([42.0], 0) == 42.0
        assert Phase46Metrics.percentile([42.0], 100) == 42.0


# ------------------------------------------------------------------
# SNR tests
# ------------------------------------------------------------------


class TestSNR:
    """SNR recording and percentile."""

    def test_record_snr_and_percentile(self, metrics: Phase46Metrics) -> None:
        for v in [0.90, 0.92, 0.95, 0.97, 0.99]:
            metrics.record_snr(v)
        p50 = metrics.percentile(metrics._snr_values, 50)
        assert 0.94 <= p50 <= 0.96


# ------------------------------------------------------------------
# HMM confidence tests
# ------------------------------------------------------------------


class TestHMMConfidence:
    """HMM confidence recording and percentile."""

    def test_record_hmm_confidence(self, metrics: Phase46Metrics) -> None:
        for v in [0.80, 0.85, 0.90, 0.92, 0.95]:
            metrics.record_hmm_confidence(v)
        p50 = metrics.percentile(metrics._hmm_confidences, 50)
        assert 0.89 <= p50 <= 0.91


# ------------------------------------------------------------------
# Observation entropy tests
# ------------------------------------------------------------------


class TestObservationEntropy:
    """Shannon entropy of observation symbol distribution."""

    def test_uniform_distribution_log2_n(self, metrics: Phase46Metrics) -> None:
        """Uniform distribution over N symbols => entropy = log2(N)."""
        n = 8
        for i in range(n):
            for _ in range(100):  # Equal counts
                metrics.record_hmm_observation(f"symbol_{i}")
        entropy = metrics.observation_entropy()
        expected = math.log2(n)
        assert abs(entropy - expected) < 0.01

    def test_single_symbol_entropy_zero(self, metrics: Phase46Metrics) -> None:
        """A single symbol => entropy = 0."""
        for _ in range(100):
            metrics.record_hmm_observation("only_symbol")
        assert metrics.observation_entropy() == 0.0

    def test_empty_observations_entropy_zero(self, metrics: Phase46Metrics) -> None:
        assert metrics.observation_entropy() == 0.0


# ------------------------------------------------------------------
# Rate computation tests
# ------------------------------------------------------------------


class TestRateComputation:
    """Rate computation from counter pairs."""

    def test_compute_rate_normal(self, metrics: Phase46Metrics) -> None:
        metrics.inc("search_hits", 75)
        metrics.inc("search_requests", 100)
        rate = metrics.compute_rate("search_hits", "search_requests")
        assert rate == 0.75

    def test_compute_rate_zero_denominator(self, metrics: Phase46Metrics) -> None:
        assert metrics.compute_rate("hits", "requests") == 0.0

    def test_compute_hit_rate(self, metrics: Phase46Metrics) -> None:
        metrics.inc("search_hits", 80)
        metrics.inc("search_requests", 100)
        assert metrics.compute_hit_rate() == 0.80


# ------------------------------------------------------------------
# Snapshot structure tests
# ------------------------------------------------------------------


class TestSnapshot:
    """Snapshot has all required keys."""

    def test_snapshot_structure(self, metrics: Phase46Metrics) -> None:
        snap = metrics.snapshot()
        assert "counters" in snap
        assert "uptime_seconds" in snap
        assert "search" in snap
        assert "got_bridge" in snap
        assert "hmm" in snap
        assert "resonance" in snap

        # Sub-keys
        assert "latency_p50_ms" in snap["search"]
        assert "latency_p95_ms" in snap["search"]
        assert "hit_rate" in snap["search"]
        assert "convergence_pass_rate" in snap["got_bridge"]
        assert "fallback_rate" in snap["got_bridge"]
        assert "confidence_p50" in snap["hmm"]
        assert "confidence_p95" in snap["hmm"]
        assert "observation_entropy" in snap["hmm"]
        assert "combined_snr_p50" in snap["resonance"]
        assert "combined_snr_p95" in snap["resonance"]

    def test_snapshot_uptime_positive(self, metrics: Phase46Metrics) -> None:
        snap = metrics.snapshot()
        assert snap["uptime_seconds"] >= 0


# ------------------------------------------------------------------
# Buffer trimming tests
# ------------------------------------------------------------------


class TestBufferTrimming:
    """Buffers trim at MAX_OBSERVATIONS."""

    def test_latency_buffer_trims(self, metrics: Phase46Metrics) -> None:
        """Adding MAX_OBSERVATIONS+1 items triggers trim to TRIM_TO."""
        max_obs = Phase46Metrics._MAX_OBSERVATIONS
        trim_to = Phase46Metrics._TRIM_TO
        for i in range(max_obs + 1):
            metrics.record_latency("search", float(i))
        assert len(metrics._latencies["search"]) == trim_to

    def test_snr_buffer_trims(self, metrics: Phase46Metrics) -> None:
        max_obs = Phase46Metrics._MAX_OBSERVATIONS
        trim_to = Phase46Metrics._TRIM_TO
        for i in range(max_obs + 1):
            metrics.record_snr(float(i))
        assert len(metrics._snr_values) == trim_to

    def test_hmm_confidence_buffer_trims(self, metrics: Phase46Metrics) -> None:
        max_obs = Phase46Metrics._MAX_OBSERVATIONS
        trim_to = Phase46Metrics._TRIM_TO
        for i in range(max_obs + 1):
            metrics.record_hmm_confidence(float(i))
        assert len(metrics._hmm_confidences) == trim_to

    def test_buffer_keeps_most_recent(self, metrics: Phase46Metrics) -> None:
        """After trim, the buffer contains the most recent values."""
        max_obs = Phase46Metrics._MAX_OBSERVATIONS
        for i in range(max_obs + 1):
            metrics.record_snr(float(i))
        # Last value should be max_obs (0-indexed: 0..max_obs)
        assert metrics._snr_values[-1] == float(max_obs)
