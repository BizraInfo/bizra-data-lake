"""Tests for RecursiveGainTracker — verifies improvement detection is honest."""

from core.benchmark.recursive_gain_tracker import (
    GainObservation,
    GainReport,
    RecursiveGainTracker,
)


class TestRecursiveGainTracker:
    """Core tracker functionality."""

    def test_empty_tracker_report(self):
        tracker = RecursiveGainTracker()
        report = tracker.report()
        assert report.observation_count == 0
        assert report.slope == 0.0
        assert not report.improvement_real
        assert not report.plateau_detected
        assert not report.pathology_detected

    def test_single_observation(self):
        tracker = RecursiveGainTracker()
        obs = tracker.record(score=0.85, cost_usd=0.01)
        assert obs.score == 0.85
        assert obs.iteration == 1
        report = tracker.report()
        assert report.observation_count == 1
        assert report.mean_score == 0.85

    def test_real_improvement_detected(self):
        tracker = RecursiveGainTracker()
        for i in range(10):
            tracker.record(score=0.70 + i * 0.02, cost_usd=0.01)
        report = tracker.report()
        assert report.improvement_real
        assert report.slope > 0
        assert report.r_squared > 0.9
        assert report.total_improvement > 0.15
        assert not report.plateau_detected
        assert not report.pathology_detected

    def test_plateau_detected(self):
        tracker = RecursiveGainTracker(plateau_patience=3)
        for i in range(8):
            tracker.record(score=0.85, cost_usd=0.01)
        report = tracker.report()
        assert report.plateau_detected
        assert report.plateau_length >= 3
        assert not report.improvement_real

    def test_regression_detected(self):
        tracker = RecursiveGainTracker()
        for i in range(10):
            tracker.record(score=0.90 - i * 0.02, cost_usd=0.01)
        report = tracker.report()
        assert report.regression_detected
        assert report.slope < 0
        assert not report.improvement_real

    def test_pathology_zero_cost(self):
        tracker = RecursiveGainTracker()
        for i in range(5):
            tracker.record(score=0.50 + i * 0.10, cost_usd=0.0)
        report = tracker.report()
        assert report.pathology_detected
        assert not report.improvement_real

    def test_pathology_constant_scores(self):
        tracker = RecursiveGainTracker()
        for _ in range(5):
            tracker.record(score=0.85, cost_usd=0.01)
        report = tracker.report()
        assert report.pathology_detected

    def test_pathology_out_of_range(self):
        tracker = RecursiveGainTracker()
        tracker.record(score=0.5, cost_usd=0.01)
        tracker.record(score=1.5, cost_usd=0.01)  # > 1.0
        tracker.record(score=0.9, cost_usd=0.01)
        report = tracker.report()
        assert report.pathology_detected

    def test_cost_per_point(self):
        tracker = RecursiveGainTracker()
        for i in range(5):
            tracker.record(score=0.80 + i * 0.01, cost_usd=0.10)
        report = tracker.report()
        assert report.total_cost_usd == 0.50
        assert report.cost_per_point > 0

    def test_reset(self):
        tracker = RecursiveGainTracker()
        tracker.record(score=0.5)
        tracker.record(score=0.6)
        assert len(tracker.observations) == 2
        tracker.reset()
        assert len(tracker.observations) == 0
        report = tracker.report()
        assert report.observation_count == 0

    def test_max_history_trimming(self):
        tracker = RecursiveGainTracker(max_history=5)
        for i in range(10):
            tracker.record(score=0.50 + i * 0.05, cost_usd=0.01)
        assert len(tracker.observations) <= 5

    def test_to_dict(self):
        tracker = RecursiveGainTracker()
        tracker.record(score=0.80, cost_usd=0.01, label="first")
        tracker.record(score=0.85, cost_usd=0.02, label="second")
        d = tracker.to_dict()
        assert "observations" in d
        assert "report" in d
        assert len(d["observations"]) == 2
        assert d["observations"][0]["label"] == "first"
        assert d["report"]["observation_count"] == 2

    def test_report_summary_string(self):
        tracker = RecursiveGainTracker()
        for i in range(5):
            tracker.record(score=0.70 + i * 0.05, cost_usd=0.01)
        report = tracker.report()
        summary = report.summary()
        assert "Recursive Gain Report" in summary
        assert "Slope" in summary
        assert "Total cost" in summary


class TestLinearRegression:
    """Verify the regression math."""

    def test_perfect_positive_trend(self):
        tracker = RecursiveGainTracker()
        for i in range(10):
            tracker.record(score=0.0 + i * 0.1)
        report = tracker.report()
        assert abs(report.slope - 0.1) < 0.01
        assert report.r_squared > 0.99

    def test_flat_line(self):
        tracker = RecursiveGainTracker()
        for _ in range(10):
            tracker.record(score=0.5)
        report = tracker.report()
        assert abs(report.slope) < 0.001

    def test_noisy_improvement(self):
        """Noisy but overall improving — should detect improvement if strong."""
        tracker = RecursiveGainTracker()
        scores = [0.70, 0.72, 0.71, 0.74, 0.73, 0.76, 0.75, 0.78, 0.77, 0.80]
        for s in scores:
            tracker.record(score=s, cost_usd=0.01)
        report = tracker.report()
        assert report.slope > 0
        assert report.total_improvement > 0.05


class TestConsecutiveRegressions:
    """Verify regression counting at tail."""

    def test_no_regressions(self):
        tracker = RecursiveGainTracker()
        for i in range(5):
            tracker.record(score=0.70 + i * 0.05)
        report = tracker.report()
        assert report.consecutive_regressions == 0

    def test_tail_regressions(self):
        tracker = RecursiveGainTracker()
        tracker.record(score=0.90)
        tracker.record(score=0.88)
        tracker.record(score=0.86)
        tracker.record(score=0.84)
        report = tracker.report()
        assert report.consecutive_regressions >= 2
