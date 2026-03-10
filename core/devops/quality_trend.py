#!/usr/bin/env python3
"""
BIZRA Quality Trend Tracker
============================

Persists quality metrics over time and computes trend direction,
enabling data-driven quality decisions and regression surfacing.
This is the "Statistical Process Control" layer that Deming
would demand of any quality-managed system.

Standing on Giants:
- Deming (SPC charts, 1950)
- Shewhart (control charts, 1924)
- Shannon (signal-over-time, 1948)

Architecture:
    COLLECT → PERSIST → ANALYZE → ALERT

Storage Format: JSONL in 04_GOLD/quality_trend.jsonl
Each line is a QualitySnapshot — append-only, hash-chained.

Usage:
    # Record a snapshot
    python -m core.devops.quality_trend record --snr 0.92 --coverage 42 --mypy-errors 1580

    # Analyze trend (last N snapshots)
    python -m core.devops.quality_trend analyze --last 30

    # Export for dashboard
    python -m core.devops.quality_trend export --format json
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# ─────────────────────────────────────────────────────────────
# Data Model
# ─────────────────────────────────────────────────────────────


@dataclass
class QualitySnapshot:
    """A point-in-time quality measurement."""

    timestamp: str = ""
    commit_sha: str = ""

    # Core quality metrics
    snr_score: float = 0.0
    ihsan_score: float = 0.0
    coverage_pct: float = 0.0
    coverage_floor: float = 0.0
    mypy_errors: int = 0
    mypy_baseline: int = 1600

    # Test metrics
    tests_total: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    tests_skipped: int = 0

    # Performance metrics
    p95_latency_ms: float = 0.0
    memory_peak_mb: float = 0.0
    startup_ms: float = 0.0

    # Security metrics
    vulnerabilities_critical: int = 0
    vulnerabilities_high: int = 0

    # Rust metrics
    rust_tests_passed: int = 0
    rust_clippy_warnings: int = 0

    # Frontend metrics
    frontend_bundle_kb: int = 0
    frontend_tests_passed: int = 0

    # Metadata
    ci_run_id: str = ""
    branch: str = ""

    # Chain
    parent_hash: str = ""
    snapshot_hash: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def compute_hash(self) -> str:
        """SHA-256 of snapshot content (excluding snapshot_hash itself)."""
        d = asdict(self)
        d.pop("snapshot_hash", None)
        content = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def finalize(self) -> None:
        """Compute and set the snapshot hash."""
        self.snapshot_hash = self.compute_hash()


@dataclass
class TrendAnalysis:
    """Result of analyzing quality trend over time."""

    window_size: int = 0
    direction: str = "stable"  # improving | degrading | stable
    snr_trend: float = 0.0  # positive = improving
    coverage_trend: float = 0.0
    mypy_trend: float = 0.0  # negative = improving (fewer errors)
    test_pass_rate_trend: float = 0.0
    anomalies: List[str] = field(default_factory=list)
    summary: str = ""


# ─────────────────────────────────────────────────────────────
# Persistence Layer
# ─────────────────────────────────────────────────────────────

DEFAULT_TREND_PATH = Path("04_GOLD/quality_trend.jsonl")


class QualityTrendStore:
    """Append-only, hash-chained quality snapshot store."""

    def __init__(self, path: Path = DEFAULT_TREND_PATH) -> None:
        self._path = path

    def append(self, snapshot: QualitySnapshot) -> None:
        """Append a snapshot, chaining to the previous entry."""
        last = self.last()
        snapshot.parent_hash = last.snapshot_hash if last else "0" * 32
        snapshot.finalize()

        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(snapshot), default=str) + "\n")

    def last(self) -> Optional[QualitySnapshot]:
        """Read the most recent snapshot."""
        if not self._path.exists():
            return None
        with open(self._path, encoding="utf-8") as f:
            lines = f.readlines()
        if not lines:
            return None
        data = json.loads(lines[-1])
        return QualitySnapshot(
            **{
                k: v
                for k, v in data.items()
                if k in QualitySnapshot.__dataclass_fields__
            }
        )

    def read_last_n(self, n: int) -> List[QualitySnapshot]:
        """Read the last N snapshots."""
        if not self._path.exists():
            return []
        with open(self._path, encoding="utf-8") as f:
            lines = f.readlines()
        snapshots = []
        for line in lines[-n:]:
            data = json.loads(line)
            snapshots.append(
                QualitySnapshot(
                    **{
                        k: v
                        for k, v in data.items()
                        if k in QualitySnapshot.__dataclass_fields__
                    }
                )
            )
        return snapshots

    def read_all(self) -> List[QualitySnapshot]:
        """Read all snapshots."""
        if not self._path.exists():
            return []
        with open(self._path, encoding="utf-8") as f:
            lines = f.readlines()
        snapshots = []
        for line in lines:
            if line.strip():
                data = json.loads(line)
                snapshots.append(
                    QualitySnapshot(
                        **{
                            k: v
                            for k, v in data.items()
                            if k in QualitySnapshot.__dataclass_fields__
                        }
                    )
                )
        return snapshots

    def count(self) -> int:
        """Number of snapshots in the store."""
        if not self._path.exists():
            return 0
        with open(self._path, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())


# ─────────────────────────────────────────────────────────────
# Trend Analysis Engine
# ─────────────────────────────────────────────────────────────


def compute_linear_trend(values: List[float]) -> float:
    """Compute the slope of a simple linear regression through values.

    Returns positive slope for improving trends, negative for degrading.
    Uses least-squares over indices 0..n-1.
    """
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = statistics.mean(values)
    numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    if denominator == 0:
        return 0.0
    return numerator / denominator


def analyze_trend(snapshots: List[QualitySnapshot]) -> TrendAnalysis:
    """Analyze quality trend from a series of snapshots."""
    if len(snapshots) < 2:
        return TrendAnalysis(
            window_size=len(snapshots),
            direction="insufficient_data",
            summary=f"Need at least 2 snapshots, have {len(snapshots)}",
        )

    analysis = TrendAnalysis(window_size=len(snapshots))
    anomalies = []

    # SNR trend
    snr_vals = [s.snr_score for s in snapshots if s.snr_score > 0]
    if snr_vals:
        analysis.snr_trend = compute_linear_trend(snr_vals)

    # Coverage trend
    cov_vals = [s.coverage_pct for s in snapshots if s.coverage_pct > 0]
    if cov_vals:
        analysis.coverage_trend = compute_linear_trend(cov_vals)

    # MyPy trend (negative slope = improving)
    mypy_vals = [float(s.mypy_errors) for s in snapshots if s.mypy_errors > 0]
    if mypy_vals:
        analysis.mypy_trend = compute_linear_trend(mypy_vals)

    # Test pass rate
    pass_rates = []
    for s in snapshots:
        if s.tests_total > 0:
            pass_rates.append(s.tests_passed / s.tests_total)
    if pass_rates:
        analysis.test_pass_rate_trend = compute_linear_trend(pass_rates)

    # Anomaly detection (simple: value deviates > 2σ from mean)
    if len(snr_vals) >= 5:
        mean_snr = statistics.mean(snr_vals)
        std_snr = statistics.stdev(snr_vals) if len(snr_vals) > 1 else 0
        if std_snr > 0:
            latest = snr_vals[-1]
            z_score = (latest - mean_snr) / std_snr
            if abs(z_score) > 2.0:
                anomalies.append(
                    f"SNR anomaly: {latest:.3f} is {z_score:+.1f}σ from mean {mean_snr:.3f}"
                )

    if len(cov_vals) >= 5:
        mean_cov = statistics.mean(cov_vals)
        std_cov = statistics.stdev(cov_vals) if len(cov_vals) > 1 else 0
        if std_cov > 0:
            latest = cov_vals[-1]
            z_score = (latest - mean_cov) / std_cov
            if z_score < -2.0:
                anomalies.append(
                    f"Coverage anomaly: {latest:.1f}% is {z_score:+.1f}σ below mean {mean_cov:.1f}%"
                )

    analysis.anomalies = anomalies

    # Overall direction
    positive_signals = 0
    negative_signals = 0

    if analysis.snr_trend > 0.001:
        positive_signals += 1
    elif analysis.snr_trend < -0.001:
        negative_signals += 1

    if analysis.coverage_trend > 0.1:
        positive_signals += 1
    elif analysis.coverage_trend < -0.1:
        negative_signals += 1

    if analysis.mypy_trend < -1.0:  # Fewer errors = good
        positive_signals += 1
    elif analysis.mypy_trend > 1.0:
        negative_signals += 1

    if positive_signals > negative_signals:
        analysis.direction = "improving"
    elif negative_signals > positive_signals:
        analysis.direction = "degrading"
    else:
        analysis.direction = "stable"

    # Summary
    analysis.summary = (
        f"Trend: {analysis.direction} over {len(snapshots)} snapshots | "
        f"SNR slope: {analysis.snr_trend:+.4f} | "
        f"Coverage slope: {analysis.coverage_trend:+.2f}%/snapshot | "
        f"MyPy slope: {analysis.mypy_trend:+.1f} errors/snapshot"
    )

    return analysis


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────


def _cli_record(args) -> int:
    """Record a new quality snapshot."""
    store = QualityTrendStore(args.store)
    snapshot = QualitySnapshot(
        commit_sha=args.commit_sha or "",
        branch=args.branch or "",
        snr_score=args.snr,
        ihsan_score=args.ihsan,
        coverage_pct=args.coverage,
        coverage_floor=args.coverage_floor,
        mypy_errors=args.mypy_errors,
        tests_total=args.tests_total,
        tests_passed=args.tests_passed,
        tests_failed=args.tests_failed,
        ci_run_id=args.ci_run_id or "",
    )
    store.append(snapshot)
    print(
        f"Snapshot recorded: {snapshot.snapshot_hash} (chain: {snapshot.parent_hash[:8]}...)"
    )
    return 0


def _cli_analyze(args) -> int:
    """Analyze quality trend."""
    store = QualityTrendStore(args.store)
    snapshots = store.read_last_n(args.last)
    analysis = analyze_trend(snapshots)

    print("=" * 60)
    print("BIZRA Quality Trend Analysis")
    print("=" * 60)
    print(f"  Window:      {analysis.window_size} snapshots")
    print(f"  Direction:   {analysis.direction.upper()}")
    print(f"  SNR Trend:   {analysis.snr_trend:+.4f}/snapshot")
    print(f"  Coverage:    {analysis.coverage_trend:+.2f}%/snapshot")
    print(f"  MyPy:        {analysis.mypy_trend:+.1f} errors/snapshot")
    print(f"  Pass Rate:   {analysis.test_pass_rate_trend:+.4f}/snapshot")
    if analysis.anomalies:
        print("\n  Anomalies:")
        for a in analysis.anomalies:
            print(f"    - {a}")
    print(f"\n  {analysis.summary}")
    return 0


def _cli_export(args) -> int:
    """Export quality data for dashboard."""
    store = QualityTrendStore(args.store)
    snapshots = store.read_all()
    data = [asdict(s) for s in snapshots]

    if args.format == "json":
        output = json.dumps(data, indent=2, default=str)
    else:
        output = "\n".join(json.dumps(d, default=str) for d in data)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Exported {len(data)} snapshots to {args.output}")
    else:
        print(output)
    return 0


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="BIZRA Quality Trend Tracker")
    parser.add_argument(
        "--store", type=Path, default=DEFAULT_TREND_PATH, help="Trend store path"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Record
    rec = subparsers.add_parser("record", help="Record a quality snapshot")
    rec.add_argument("--commit-sha", default="")
    rec.add_argument("--branch", default="")
    rec.add_argument("--snr", type=float, default=0.0)
    rec.add_argument("--ihsan", type=float, default=0.0)
    rec.add_argument("--coverage", type=float, default=0.0)
    rec.add_argument("--coverage-floor", type=float, default=0.0)
    rec.add_argument("--mypy-errors", type=int, default=0)
    rec.add_argument("--tests-total", type=int, default=0)
    rec.add_argument("--tests-passed", type=int, default=0)
    rec.add_argument("--tests-failed", type=int, default=0)
    rec.add_argument("--ci-run-id", default="")

    # Analyze
    ana = subparsers.add_parser("analyze", help="Analyze quality trend")
    ana.add_argument("--last", type=int, default=30, help="Last N snapshots to analyze")

    # Export
    exp = subparsers.add_parser("export", help="Export quality data")
    exp.add_argument("--format", choices=["json", "jsonl"], default="json")
    exp.add_argument("--output", default=None, help="Output file path")

    args = parser.parse_args()

    if args.command == "record":
        return _cli_record(args)
    elif args.command == "analyze":
        return _cli_analyze(args)
    elif args.command == "export":
        return _cli_export(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
