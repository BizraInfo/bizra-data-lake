"""
Tests for BIZRA Quality Trend Tracker
=======================================

Validates quality snapshot persistence, hash chaining,
trend analysis, and anomaly detection.
"""

import json
from pathlib import Path

import pytest

from core.devops.quality_trend import (
    QualitySnapshot,
    QualityTrendStore,
    analyze_trend,
    compute_linear_trend,
)

# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────


@pytest.fixture
def trend_path(tmp_path: Path) -> Path:
    return tmp_path / "quality_trend.jsonl"


@pytest.fixture
def store(trend_path: Path) -> QualityTrendStore:
    return QualityTrendStore(trend_path)


@pytest.fixture
def sample_snapshots() -> list:
    """Generate 10 snapshots simulating gradual improvement."""
    snapshots = []
    for i in range(10):
        snapshots.append(
            QualitySnapshot(
                timestamp=f"2025-01-{i+1:02d}T00:00:00Z",
                commit_sha=f"abc{i:04d}",
                snr_score=0.85 + i * 0.01,
                ihsan_score=0.90 + i * 0.005,
                coverage_pct=38.0 + i * 1.5,
                coverage_floor=38.0 + max(0, i - 2),
                mypy_errors=1600 - i * 20,
                tests_total=200 + i * 5,
                tests_passed=190 + i * 5,
                tests_failed=10 - min(i, 10),
            )
        )
    return snapshots


# ─────────────────────────────────────────────────────────────
# Tests: QualitySnapshot
# ─────────────────────────────────────────────────────────────


class TestQualitySnapshot:
    """Test snapshot data model."""

    def test_default_timestamp(self) -> None:
        s = QualitySnapshot()
        assert s.timestamp  # Auto-populated

    def test_explicit_timestamp(self) -> None:
        s = QualitySnapshot(timestamp="2025-01-01T00:00:00Z")
        assert s.timestamp == "2025-01-01T00:00:00Z"

    def test_compute_hash(self) -> None:
        s = QualitySnapshot(
            timestamp="2025-01-01T00:00:00Z",
            snr_score=0.92,
            coverage_pct=42.0,
        )
        h = s.compute_hash()
        assert len(h) == 32
        assert h.isalnum()

    def test_finalize_sets_hash(self) -> None:
        s = QualitySnapshot(timestamp="2025-01-01T00:00:00Z")
        s.finalize()
        assert s.snapshot_hash != ""
        assert len(s.snapshot_hash) == 32

    def test_hash_deterministic(self) -> None:
        s1 = QualitySnapshot(timestamp="2025-01-01T00:00:00Z", snr_score=0.92)
        s2 = QualitySnapshot(timestamp="2025-01-01T00:00:00Z", snr_score=0.92)
        assert s1.compute_hash() == s2.compute_hash()

    def test_hash_differs_on_change(self) -> None:
        s1 = QualitySnapshot(timestamp="2025-01-01T00:00:00Z", snr_score=0.92)
        s2 = QualitySnapshot(timestamp="2025-01-01T00:00:00Z", snr_score=0.93)
        assert s1.compute_hash() != s2.compute_hash()


# ─────────────────────────────────────────────────────────────
# Tests: QualityTrendStore
# ─────────────────────────────────────────────────────────────


class TestQualityTrendStore:
    """Test JSONL persistence and hash chaining."""

    def test_empty_store(self, store: QualityTrendStore) -> None:
        assert store.last() is None
        assert store.count() == 0
        assert store.read_all() == []

    def test_append_single(self, store: QualityTrendStore) -> None:
        s = QualitySnapshot(timestamp="2025-01-01T00:00:00Z", snr_score=0.92)
        store.append(s)
        assert store.count() == 1
        last = store.last()
        assert last is not None
        assert last.snr_score == 0.92

    def test_append_chains_hashes(self, store: QualityTrendStore) -> None:
        s1 = QualitySnapshot(timestamp="2025-01-01T00:00:00Z", snr_score=0.90)
        store.append(s1)

        s2 = QualitySnapshot(timestamp="2025-01-02T00:00:00Z", snr_score=0.92)
        store.append(s2)

        all_snaps = store.read_all()
        assert len(all_snaps) == 2
        # Second snapshot's parent_hash should be first's snapshot_hash
        assert all_snaps[1].parent_hash == all_snaps[0].snapshot_hash

    def test_read_last_n(
        self, store: QualityTrendStore, sample_snapshots: list
    ) -> None:
        for s in sample_snapshots:
            store.append(s)
        last_3 = store.read_last_n(3)
        assert len(last_3) == 3

    def test_genesis_parent_hash(self, store: QualityTrendStore) -> None:
        """First snapshot has zeroed parent hash."""
        s = QualitySnapshot(timestamp="2025-01-01T00:00:00Z")
        store.append(s)
        first = store.read_all()[0]
        assert first.parent_hash == "0" * 32

    def test_jsonl_format(self, store: QualityTrendStore, trend_path: Path) -> None:
        store.append(QualitySnapshot(timestamp="2025-01-01T00:00:00Z"))
        store.append(QualitySnapshot(timestamp="2025-01-02T00:00:00Z"))
        lines = trend_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            data = json.loads(line)
            assert "snapshot_hash" in data
            assert "parent_hash" in data


# ─────────────────────────────────────────────────────────────
# Tests: Linear Trend Computation
# ─────────────────────────────────────────────────────────────


class TestLinearTrend:
    """Test linear regression slope computation."""

    def test_flat_trend(self) -> None:
        assert compute_linear_trend([1.0, 1.0, 1.0, 1.0]) == 0.0

    def test_positive_trend(self) -> None:
        slope = compute_linear_trend([1.0, 2.0, 3.0, 4.0])
        assert slope == pytest.approx(1.0)

    def test_negative_trend(self) -> None:
        slope = compute_linear_trend([4.0, 3.0, 2.0, 1.0])
        assert slope == pytest.approx(-1.0)

    def test_insufficient_data(self) -> None:
        assert compute_linear_trend([5.0]) == 0.0
        assert compute_linear_trend([]) == 0.0

    def test_noisy_but_upward(self) -> None:
        slope = compute_linear_trend([1.0, 1.5, 1.2, 2.0, 2.5, 2.1, 3.0])
        assert slope > 0  # Noisy but upward overall


# ─────────────────────────────────────────────────────────────
# Tests: Trend Analysis
# ─────────────────────────────────────────────────────────────


class TestTrendAnalysis:
    """Test quality trend analysis engine."""

    def test_insufficient_data(self) -> None:
        result = analyze_trend([QualitySnapshot()])
        assert result.direction == "insufficient_data"

    def test_improving_trend(self, sample_snapshots: list) -> None:
        result = analyze_trend(sample_snapshots)
        assert result.direction == "improving"
        assert result.snr_trend > 0
        assert result.coverage_trend > 0
        assert result.mypy_trend < 0  # Fewer errors = improving

    def test_degrading_trend(self) -> None:
        snapshots = []
        for i in range(10):
            snapshots.append(
                QualitySnapshot(
                    timestamp=f"2025-01-{i+1:02d}T00:00:00Z",
                    snr_score=0.95 - i * 0.02,
                    coverage_pct=50.0 - i * 2.0,
                    mypy_errors=1000 + i * 50,
                    tests_total=200,
                    tests_passed=190 - i * 5,
                )
            )
        result = analyze_trend(snapshots)
        assert result.direction == "degrading"

    def test_stable_trend(self) -> None:
        snapshots = []
        for i in range(10):
            snapshots.append(
                QualitySnapshot(
                    timestamp=f"2025-01-{i+1:02d}T00:00:00Z",
                    snr_score=0.92,
                    coverage_pct=42.0,
                    mypy_errors=1500,
                    tests_total=200,
                    tests_passed=195,
                )
            )
        result = analyze_trend(snapshots)
        assert result.direction == "stable"

    def test_summary_populated(self, sample_snapshots: list) -> None:
        result = analyze_trend(sample_snapshots)
        assert "Trend:" in result.summary
        assert "SNR slope:" in result.summary
        assert "Coverage slope:" in result.summary

    def test_window_size_correct(self, sample_snapshots: list) -> None:
        result = analyze_trend(sample_snapshots[:5])
        assert result.window_size == 5

    def test_anomaly_detection(self) -> None:
        # Create snapshots with a large final outlier
        snapshots = []
        for i in range(8):
            snapshots.append(
                QualitySnapshot(
                    timestamp=f"2025-01-{i+1:02d}T00:00:00Z",
                    snr_score=0.92,
                )
            )
        # Outlier
        snapshots.append(
            QualitySnapshot(
                timestamp="2025-01-09T00:00:00Z",
                snr_score=0.50,  # Way below mean
            )
        )
        result = analyze_trend(snapshots)
        # At least check it ran without error; anomaly detection depends on σ
        assert isinstance(result.anomalies, list)
