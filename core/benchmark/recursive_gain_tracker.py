"""
Recursive Gain Tracker — Measures Whether Improvement Is Real
═══════════════════════════════════════════════════════════════

Tracks improvement slopes, cost-per-improvement, plateau detection,
and pathology detection for the recursive optimization loop.

The core question: "Is the system getting better, or just claiming it is?"

Standing on Giants:
  Deming (1950): PDCA — measure, don't assume
  Boyd (1995): OODA — tight feedback loops with observed reality
  Goldratt (1984): Theory of Constraints — find the real bottleneck
  Al-Ghazali (1095): Maqasid — preserve through verified means

Usage:
    tracker = RecursiveGainTracker()
    tracker.record(score=0.72, cost_usd=0.01)
    tracker.record(score=0.74, cost_usd=0.02)
    tracker.record(score=0.75, cost_usd=0.01)
    report = tracker.report()
    # report.slope > 0 → real improvement
    # report.plateau_detected → improvement stalled
    # report.pathology_detected → fabricated metrics
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class GainObservation:
    """A single measurement point."""

    score: float
    cost_usd: float = 0.0
    timestamp: float = field(default_factory=time.time)
    iteration: int = 0
    label: str = ""


@dataclass
class GainReport:
    """Report on recursive improvement status."""

    # Core metrics
    slope: float = 0.0  # Linear regression slope of score vs iteration
    r_squared: float = 0.0  # Fit quality (0 = random, 1 = perfect trend)
    total_improvement: float = 0.0  # Last score - first score
    mean_score: float = 0.0  # Average score across all observations
    best_score: float = 0.0  # Peak score observed
    worst_score: float = 0.0  # Minimum score observed

    # Cost analysis
    total_cost_usd: float = 0.0
    cost_per_point: float = 0.0  # Cost per 0.01 improvement
    efficiency: float = 0.0  # improvement / cost (higher = better)

    # Detection flags
    plateau_detected: bool = False  # Improvement stalled
    regression_detected: bool = False  # Getting worse
    pathology_detected: bool = False  # Fabricated metrics suspected
    improvement_real: bool = False  # Genuine measured improvement

    # Metadata
    observation_count: int = 0
    time_span_seconds: float = 0.0
    plateau_length: int = 0
    consecutive_regressions: int = 0

    def summary(self) -> str:
        """Human-readable summary."""
        status = (
            "IMPROVING"
            if self.improvement_real
            else (
                "PLATEAU"
                if self.plateau_detected
                else (
                    "REGRESSING"
                    if self.regression_detected
                    else (
                        "PATHOLOGICAL"
                        if self.pathology_detected
                        else "INSUFFICIENT DATA"
                    )
                )
            )
        )
        lines = [
            f"Recursive Gain Report ({self.observation_count} observations)",
            f"  Status:          {status}",
            f"  Slope:           {self.slope:+.6f} per iteration",
            f"  R-squared:       {self.r_squared:.4f}",
            f"  Total gain:      {self.total_improvement:+.4f}",
            f"  Best/Worst/Mean: {self.best_score:.4f} / {self.worst_score:.4f} / {self.mean_score:.4f}",
            f"  Total cost:      ${self.total_cost_usd:.4f}",
            f"  Cost/0.01:       ${self.cost_per_point:.4f}",
        ]
        if self.plateau_detected:
            lines.append(f"  Plateau length:  {self.plateau_length} iterations")
        if self.consecutive_regressions > 0:
            lines.append(
                f"  Regressions:     {self.consecutive_regressions} consecutive"
            )
        return "\n".join(lines)


class RecursiveGainTracker:
    """
    Tracks improvement over time and detects real vs fabricated gains.

    Plateau: >= `plateau_patience` consecutive observations with
             delta < `improvement_threshold`.
    Regression: slope < 0 with R^2 > 0.3 (statistically meaningful decline).
    Pathology: improvement claimed but variance near zero (constant output),
               or score > 1.0, or cost = 0 with large improvement.
    """

    def __init__(
        self,
        improvement_threshold: float = 0.005,
        plateau_patience: int = 5,
        max_history: int = 1000,
    ):
        self.improvement_threshold = improvement_threshold
        self.plateau_patience = plateau_patience
        self.max_history = max_history
        self.observations: List[GainObservation] = []
        self._iteration_counter = 0

    def record(
        self,
        score: float,
        cost_usd: float = 0.0,
        label: str = "",
    ) -> GainObservation:
        """Record a new measurement."""
        self._iteration_counter += 1
        obs = GainObservation(
            score=score,
            cost_usd=cost_usd,
            timestamp=time.time(),
            iteration=self._iteration_counter,
            label=label,
        )
        self.observations.append(obs)

        # Trim history
        if len(self.observations) > self.max_history:
            self.observations = self.observations[-self.max_history :]

        return obs

    def report(self) -> GainReport:
        """Generate a gain report from all observations."""
        report = GainReport()
        n = len(self.observations)
        report.observation_count = n

        if n == 0:
            return report

        scores = [o.score for o in self.observations]
        costs = [o.cost_usd for o in self.observations]
        iterations = [o.iteration for o in self.observations]

        report.mean_score = statistics.mean(scores)
        report.best_score = max(scores)
        report.worst_score = min(scores)
        report.total_cost_usd = sum(costs)
        report.total_improvement = scores[-1] - scores[0]

        if n >= 2:
            report.time_span_seconds = (
                self.observations[-1].timestamp - self.observations[0].timestamp
            )

        # Linear regression: score = slope * iteration + intercept
        if n >= 3:
            slope, r_sq = self._linear_regression(iterations, scores)
            report.slope = slope
            report.r_squared = r_sq

        # Cost per 0.01 improvement
        if report.total_improvement > 0 and report.total_cost_usd > 0:
            improvement_in_points = report.total_improvement / 0.01
            report.cost_per_point = report.total_cost_usd / max(
                improvement_in_points, 0.01
            )
            report.efficiency = report.total_improvement / max(
                report.total_cost_usd, 1e-9
            )

        # Plateau detection
        report.plateau_length, report.plateau_detected = self._detect_plateau(scores)

        # Regression detection
        report.consecutive_regressions = self._count_consecutive_regressions(scores)
        if n >= 3:
            report.regression_detected = (
                report.slope < -self.improvement_threshold and report.r_squared > 0.3
            )

        # Pathology detection
        report.pathology_detected = self._detect_pathology(scores, costs)

        # Real improvement: positive slope, good fit, no pathology
        report.improvement_real = (
            n >= 3
            and report.slope > self.improvement_threshold
            and report.r_squared > 0.3
            and not report.pathology_detected
            and not report.plateau_detected
        )

        return report

    def _linear_regression(self, x: List[float], y: List[float]) -> Tuple[float, float]:
        """Simple linear regression returning (slope, r_squared)."""
        n = len(x)
        if n < 2:
            return 0.0, 0.0

        x_mean = statistics.mean(x)
        y_mean = statistics.mean(y)

        ss_xy = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        ss_xx = sum((xi - x_mean) ** 2 for xi in x)
        ss_yy = sum((yi - y_mean) ** 2 for yi in y)

        if ss_xx == 0 or ss_yy == 0:
            return 0.0, 0.0

        slope = ss_xy / ss_xx
        r_squared = (ss_xy**2) / (ss_xx * ss_yy)

        return slope, r_squared

    def _detect_plateau(self, scores: List[float]) -> Tuple[int, bool]:
        """Count consecutive small-delta observations at the tail."""
        if len(scores) < 2:
            return 0, False

        plateau_len = 0
        for i in range(len(scores) - 1, 0, -1):
            delta = abs(scores[i] - scores[i - 1])
            if delta < self.improvement_threshold:
                plateau_len += 1
            else:
                break

        return plateau_len, plateau_len >= self.plateau_patience

    def _count_consecutive_regressions(self, scores: List[float]) -> int:
        """Count consecutive decreasing observations at the tail."""
        count = 0
        for i in range(len(scores) - 1, 0, -1):
            if scores[i] < scores[i - 1] - self.improvement_threshold:
                count += 1
            else:
                break
        return count

    def _detect_pathology(self, scores: List[float], costs: List[float]) -> bool:
        """Detect fabricated or meaningless metrics."""
        if len(scores) < 3:
            return False

        # Pathology 1: All scores identical (no real measurement)
        if len(set(round(s, 6) for s in scores)) == 1:
            return True

        # Pathology 2: Score > 1.0 (impossible for normalized metrics)
        if any(s > 1.0 for s in scores):
            return True

        # Pathology 3: Score < 0.0 (impossible for normalized metrics)
        if any(s < 0.0 for s in scores):
            return True

        # Pathology 4: Variance is suspiciously low but not zero
        # (looks like small random noise around a constant)
        if len(scores) >= 5:
            variance = statistics.variance(scores)
            if variance < 1e-8 and max(scores) != min(scores):
                return True

        # Pathology 5: Large improvement with zero cost
        total_improvement = scores[-1] - scores[0]
        total_cost = sum(costs)
        if total_improvement > 0.1 and total_cost == 0.0:
            return True

        return False

    def reset(self) -> None:
        """Clear all observations."""
        self.observations.clear()
        self._iteration_counter = 0

    def to_dict(self) -> dict:
        """Serialize tracker state."""
        report = self.report()
        return {
            "observations": [
                {
                    "score": o.score,
                    "cost_usd": o.cost_usd,
                    "iteration": o.iteration,
                    "label": o.label,
                    "timestamp": o.timestamp,
                }
                for o in self.observations
            ],
            "report": {
                "slope": report.slope,
                "r_squared": report.r_squared,
                "total_improvement": report.total_improvement,
                "mean_score": report.mean_score,
                "best_score": report.best_score,
                "total_cost_usd": report.total_cost_usd,
                "cost_per_point": report.cost_per_point,
                "plateau_detected": report.plateau_detected,
                "regression_detected": report.regression_detected,
                "pathology_detected": report.pathology_detected,
                "improvement_real": report.improvement_real,
                "observation_count": report.observation_count,
            },
        }
