#!/usr/bin/env python3
"""
BIZRA Benchmark Dashboard - Historical Trend Analysis and Visualization

Generates comprehensive benchmark reports including:
- Model performance trends over time
- BIZRA vs Direct/Routed comparisons
- Accuracy, latency, and SNR metrics
- HTML dashboard with interactive charts
- SQLite historical tracking

Usage:
    python scripts/benchmark_dashboard.py --html
    python scripts/benchmark_dashboard.py --json
    python scripts/benchmark_dashboard.py --summary
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Optional imports
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ============================================================================
# CONFIGURATION
# ============================================================================

WORKSPACE = Path(__file__).parent.parent
BENCHMARK_PATH = WORKSPACE / "docs" / "evidence" / "benchmarks"
HISTORY_DB = WORKSPACE / "docs" / "evidence" / "quality_history.db"
DASHBOARD_OUTPUT = WORKSPACE / "docs" / "evidence" / "benchmark_dashboard.html"

# Color palette aligned with quality_radar_elite.py
COLORS = {
    "bizra": "#1FB8CD",      # Teal - BIZRA mode
    "direct": "#5D878F",     # Grey-teal - Direct mode
    "routed": "#D2BA4C",     # Gold - Routed mode
    "excellent": "#1FB8CD",
    "good": "#5D878F",
    "warning": "#D2BA4C",
    "needs": "#DB4545",
}

# SNR tier thresholds
SNR_TIERS = {
    "T6": (9.0, float("inf"), "Elite", "#1FB8CD"),
    "T5": (8.6, 9.0, "Expert", "#5D878F"),
    "T4": (8.2, 8.6, "Strong", "#7BA695"),
    "T3": (7.8, 8.2, "Target", "#D2BA4C"),
    "T2": (7.4, 7.8, "Acceptable", "#E89B3C"),
    "T1": (0.0, 7.4, "Baseline", "#DB4545"),
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ModelResult:
    """Single model benchmark result."""
    model_name: str
    provider: str
    accuracy: float
    latency_p50_ms: float
    latency_p95_ms: float
    tokens_per_second: float
    total_questions: int
    correct: int
    errors: int
    timestamp: str = ""


@dataclass
class ModeResult:
    """Single execution mode result."""
    mode: str
    accuracy: float
    latency_p50_ms: float
    latency_p95_ms: float
    ihsan_score: Optional[float] = None
    snr_score: Optional[float] = None
    sat_consensus_rate: Optional[float] = None


@dataclass
class BenchmarkSummary:
    """Aggregated benchmark summary."""
    total_runs: int = 0
    latest_timestamp: str = ""

    # Model baselines
    models_tested: list[str] = field(default_factory=list)
    best_model: str = ""
    best_model_accuracy: float = 0.0
    avg_model_accuracy: float = 0.0

    # Mode comparisons
    bizra_accuracy: float = 0.0
    direct_accuracy: float = 0.0
    routed_accuracy: float = 0.0
    bizra_improvement: float = 0.0

    # BIZRA-specific
    avg_ihsan: float = 0.0
    avg_snr: float = 0.0
    snr_tier: str = "T1"

    # Trends
    accuracy_trend: str = "unknown"
    accuracy_delta: float = 0.0


# ============================================================================
# DATA LOADING
# ============================================================================

def load_benchmark_files() -> tuple[list[dict], list[dict], list[dict]]:
    """Load all benchmark files from evidence directory."""
    baselines = []
    comparisons = []
    benchmarks = []

    if not BENCHMARK_PATH.exists():
        return baselines, comparisons, benchmarks

    for filepath in sorted(BENCHMARK_PATH.glob("*.json")):
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
            name = filepath.stem

            if name.startswith("model_baselines_"):
                baselines.append(data)
            elif name.startswith("comparison_"):
                comparisons.append(data)
            elif name.startswith("benchmark_"):
                benchmarks.append(data)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: Failed to load {filepath}: {e}")

    return baselines, comparisons, benchmarks


def compute_summary(baselines: list[dict], comparisons: list[dict]) -> BenchmarkSummary:
    """Compute aggregate summary from benchmark data."""
    summary = BenchmarkSummary()

    # Process baselines
    all_models: dict[str, list[float]] = {}

    for baseline in baselines:
        summary.total_runs += 1
        ts = baseline.get("timestamp", "")
        if ts > summary.latest_timestamp:
            summary.latest_timestamp = ts

        for result in baseline.get("results", []):
            model = result.get("model_name", "unknown")
            metrics = result.get("metrics", {})
            accuracy = metrics.get("accuracy", 0)

            if model not in all_models:
                all_models[model] = []
            all_models[model].append(accuracy)

    # Model statistics
    if all_models:
        summary.models_tested = list(all_models.keys())
        model_avgs = {m: sum(scores) / len(scores) for m, scores in all_models.items()}
        summary.best_model = max(model_avgs, key=model_avgs.get)
        summary.best_model_accuracy = model_avgs[summary.best_model]
        summary.avg_model_accuracy = sum(model_avgs.values()) / len(model_avgs)

    # Process comparisons
    bizra_accuracies = []
    direct_accuracies = []
    routed_accuracies = []
    ihsan_scores = []
    snr_scores = []

    for comparison in comparisons:
        summary.total_runs += 1
        ts = comparison.get("timestamp", "")
        if ts > summary.latest_timestamp:
            summary.latest_timestamp = ts

        comp_summary = comparison.get("comparison_summary", {})
        modes = comp_summary.get("modes", {})

        if "bizra" in modes:
            bizra_accuracies.append(modes["bizra"].get("avg_accuracy", 0))
            ihsan = modes["bizra"].get("avg_ihsan")
            snr = modes["bizra"].get("avg_snr")
            if ihsan:
                ihsan_scores.append(ihsan)
            if snr:
                snr_scores.append(snr)

        if "direct" in modes:
            direct_accuracies.append(modes["direct"].get("avg_accuracy", 0))

        if "routed" in modes:
            routed_accuracies.append(modes["routed"].get("avg_accuracy", 0))

    # Mode averages
    if bizra_accuracies:
        summary.bizra_accuracy = sum(bizra_accuracies) / len(bizra_accuracies)
    if direct_accuracies:
        summary.direct_accuracy = sum(direct_accuracies) / len(direct_accuracies)
    if routed_accuracies:
        summary.routed_accuracy = sum(routed_accuracies) / len(routed_accuracies)

    # BIZRA improvement
    if summary.direct_accuracy > 0:
        summary.bizra_improvement = summary.bizra_accuracy - summary.direct_accuracy

    # BIZRA-specific metrics
    if ihsan_scores:
        summary.avg_ihsan = sum(ihsan_scores) / len(ihsan_scores)
    if snr_scores:
        summary.avg_snr = sum(snr_scores) / len(snr_scores)
        # Determine SNR tier
        for tier, (low, high, _, _) in SNR_TIERS.items():
            if low <= summary.avg_snr < high:
                summary.snr_tier = tier
                break

    # Compute trend from comparison history
    if len(bizra_accuracies) >= 2:
        recent = bizra_accuracies[:len(bizra_accuracies)//2]
        older = bizra_accuracies[len(bizra_accuracies)//2:]
        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        summary.accuracy_delta = recent_avg - older_avg

        if summary.accuracy_delta > 0.02:
            summary.accuracy_trend = "improving"
        elif summary.accuracy_delta < -0.02:
            summary.accuracy_trend = "declining"
        else:
            summary.accuracy_trend = "stable"

    return summary


# ============================================================================
# DASHBOARD GENERATION
# ============================================================================

def generate_html_dashboard(
    baselines: list[dict],
    comparisons: list[dict],
    summary: BenchmarkSummary,
    output_path: Path,
) -> bool:
    """Generate interactive HTML dashboard."""
    if not PLOTLY_AVAILABLE:
        print("Warning: Plotly not installed, skipping HTML generation")
        return False

    print("Generating HTML dashboard...")

    # Create figure with subplots
    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "pie"}],
            [{"type": "indicator", "colspan": 2}, None],
        ],
        subplot_titles=(
            "Model Baseline Accuracy",
            "Mode Comparison",
            "Accuracy Trend Over Time",
            "Test Set Distribution",
            "",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # 1. Model Baseline Chart (top left)
    if baselines:
        latest_baseline = baselines[-1]
        model_names = []
        accuracies = []

        for result in latest_baseline.get("results", []):
            model_names.append(result.get("model_name", "")[:15])
            accuracies.append(result.get("metrics", {}).get("accuracy", 0) * 100)

        bar_colors = [
            COLORS["excellent"] if a >= 80 else COLORS["good"] if a >= 70 else COLORS["warning"]
            for a in accuracies
        ]

        fig.add_trace(go.Bar(
            x=model_names,
            y=accuracies,
            marker_color=bar_colors,
            text=[f"{a:.1f}%" for a in accuracies],
            textposition="outside",
            name="Model Accuracy",
            showlegend=False,
        ), row=1, col=1)

        fig.update_yaxes(range=[0, 100], title="Accuracy %", row=1, col=1)

    # 2. Mode Comparison Chart (top right)
    modes = ["Direct", "Routed", "BIZRA"]
    mode_accuracies = [
        summary.direct_accuracy * 100,
        summary.routed_accuracy * 100,
        summary.bizra_accuracy * 100,
    ]
    mode_colors = [COLORS["direct"], COLORS["routed"], COLORS["bizra"]]

    fig.add_trace(go.Bar(
        x=modes,
        y=mode_accuracies,
        marker_color=mode_colors,
        text=[f"{a:.1f}%" for a in mode_accuracies],
        textposition="outside",
        name="Mode Accuracy",
        showlegend=False,
    ), row=1, col=2)

    fig.update_yaxes(range=[0, 100], title="Accuracy %", row=1, col=2)

    # 3. Accuracy Trend (middle left)
    if comparisons:
        timestamps = []
        bizra_trend = []
        direct_trend = []

        for comp in sorted(comparisons, key=lambda x: x.get("timestamp", "")):
            ts = comp.get("timestamp", "")[:10]
            timestamps.append(ts)

            modes_data = comp.get("comparison_summary", {}).get("modes", {})
            bizra_trend.append(modes_data.get("bizra", {}).get("avg_accuracy", 0) * 100)
            direct_trend.append(modes_data.get("direct", {}).get("avg_accuracy", 0) * 100)

        fig.add_trace(go.Scatter(
            x=timestamps,
            y=bizra_trend,
            mode="lines+markers",
            name="BIZRA",
            line=dict(color=COLORS["bizra"], width=3),
            marker=dict(size=8),
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=timestamps,
            y=direct_trend,
            mode="lines+markers",
            name="Direct",
            line=dict(color=COLORS["direct"], width=2, dash="dash"),
            marker=dict(size=6),
        ), row=2, col=1)

        fig.update_yaxes(title="Accuracy %", row=2, col=1)
        fig.update_xaxes(title="Date", row=2, col=1)

    # 4. Test Set Distribution (middle right)
    test_sets = {"mmlu_mini": 0, "hellaswag_mini": 0, "bizra_qa": 0}
    for baseline in baselines:
        ts = baseline.get("test_set", "unknown")
        if ts in test_sets:
            test_sets[ts] += 1
    for comp in comparisons:
        ts = comp.get("test_set", "unknown")
        if ts in test_sets:
            test_sets[ts] += 1

    if any(test_sets.values()):
        fig.add_trace(go.Pie(
            labels=list(test_sets.keys()),
            values=list(test_sets.values()),
            hole=0.4,
            marker_colors=[COLORS["bizra"], COLORS["routed"], COLORS["direct"]],
        ), row=2, col=2)

    # 5. Overall Score Indicator (bottom)
    overall_score = summary.bizra_accuracy * 10 if summary.bizra_accuracy else 5.0

    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=overall_score,
        title={"text": "BIZRA Benchmark Score"},
        delta={"reference": summary.direct_accuracy * 10, "relative": False, "valueformat": ".2f"},
        gauge={
            "axis": {"range": [0, 10], "tickwidth": 1},
            "bar": {"color": COLORS["bizra"]},
            "bgcolor": "white",
            "borderwidth": 2,
            "steps": [
                {"range": [0, 6], "color": "rgba(219,69,69,0.2)"},
                {"range": [6, 7], "color": "rgba(210,186,76,0.2)"},
                {"range": [7, 8], "color": "rgba(93,135,143,0.2)"},
                {"range": [8, 10], "color": "rgba(31,184,205,0.2)"},
            ],
            "threshold": {
                "line": {"color": COLORS["direct"], "width": 4},
                "thickness": 0.75,
                "value": summary.direct_accuracy * 10,
            },
        },
    ), row=3, col=1)

    # Layout
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    improvement_text = f"+{summary.bizra_improvement*100:.1f}%" if summary.bizra_improvement > 0 else f"{summary.bizra_improvement*100:.1f}%"

    fig.update_layout(
        title={
            "text": (
                f"<b>BIZRA Benchmark Dashboard</b><br>"
                f"<span style='font-size:12px;color:gray'>"
                f"Generated: {timestamp} | "
                f"Runs: {summary.total_runs} | "
                f"BIZRA vs Direct: {improvement_text} | "
                f"SNR Tier: {summary.snr_tier}"
                f"</span>"
            ),
            "x": 0.5,
        },
        height=1000,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=0.52, xanchor="center", x=0.25),
    )

    # Save
    try:
        fig.write_html(str(output_path))
        print(f"Dashboard saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving dashboard: {e}")
        return False


def print_summary(summary: BenchmarkSummary) -> None:
    """Print text summary to console."""
    print("\n" + "=" * 60)
    print("BIZRA BENCHMARK SUMMARY")
    print("=" * 60)

    print(f"\nTotal benchmark runs: {summary.total_runs}")
    print(f"Latest timestamp: {summary.latest_timestamp}")

    if summary.models_tested:
        print(f"\nModels tested: {len(summary.models_tested)}")
        print(f"Best model: {summary.best_model} ({summary.best_model_accuracy*100:.1f}%)")
        print(f"Avg model accuracy: {summary.avg_model_accuracy*100:.1f}%")

    print("\n--- Mode Comparison ---")
    print(f"Direct accuracy:  {summary.direct_accuracy*100:.1f}%")
    print(f"Routed accuracy:  {summary.routed_accuracy*100:.1f}%")
    print(f"BIZRA accuracy:   {summary.bizra_accuracy*100:.1f}%")

    delta_str = f"+{summary.bizra_improvement*100:.1f}%" if summary.bizra_improvement > 0 else f"{summary.bizra_improvement*100:.1f}%"
    print(f"BIZRA improvement: {delta_str}")

    print("\n--- BIZRA Metrics ---")
    print(f"Avg Ihsan score: {summary.avg_ihsan:.4f}")
    print(f"Avg SNR score:   {summary.avg_snr:.2f}")
    print(f"SNR Tier:        {summary.snr_tier}")

    print("\n--- Trend ---")
    trend_emoji = {"improving": "rising", "declining": "falling", "stable": "neutral", "unknown": "?"}
    print(f"Accuracy trend: {summary.accuracy_trend} ({summary.accuracy_delta:+.2%})")

    print("\n" + "=" * 60)


# ============================================================================
# HISTORY TRACKING
# ============================================================================

def init_benchmark_table() -> None:
    """Initialize benchmark history table in SQLite."""
    HISTORY_DB.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(HISTORY_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                run_type TEXT NOT NULL,
                test_set TEXT,
                mode TEXT,
                model TEXT,
                accuracy REAL,
                latency_p50 REAL,
                latency_p95 REAL,
                ihsan_score REAL,
                snr_score REAL,
                raw_json TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_bench_timestamp
            ON benchmark_history(timestamp)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_bench_type
            ON benchmark_history(run_type)
        """)


def save_to_history(
    run_type: str,
    test_set: str,
    results: list[dict],
    timestamp: str,
) -> None:
    """Save benchmark results to history database."""
    try:
        init_benchmark_table()

        with sqlite3.connect(HISTORY_DB) as conn:
            for result in results:
                mode = result.get("mode", "")
                model = result.get("model_name", result.get("model", ""))
                metrics = result.get("metrics", {})
                bizra_metrics = result.get("bizra_metrics", {})

                conn.execute("""
                    INSERT INTO benchmark_history
                    (timestamp, run_type, test_set, mode, model, accuracy,
                     latency_p50, latency_p95, ihsan_score, snr_score, raw_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    timestamp,
                    run_type,
                    test_set,
                    mode,
                    model,
                    metrics.get("accuracy"),
                    metrics.get("latency_p50_ms"),
                    metrics.get("latency_p95_ms"),
                    bizra_metrics.get("ihsan_score"),
                    bizra_metrics.get("snr_score"),
                    json.dumps(result),
                ))
    except Exception as e:
        print(f"Warning: Failed to save to history: {e}")


def get_historical_trend(days: int = 30) -> list[dict]:
    """Get historical benchmark trend."""
    try:
        if not HISTORY_DB.exists():
            return []

        with sqlite3.connect(HISTORY_DB) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT
                    date(timestamp) as date,
                    mode,
                    AVG(accuracy) as avg_accuracy,
                    AVG(ihsan_score) as avg_ihsan,
                    AVG(snr_score) as avg_snr
                FROM benchmark_history
                WHERE timestamp >= date('now', ?)
                GROUP BY date(timestamp), mode
                ORDER BY date(timestamp)
            """, (f"-{days} days",))

            return [dict(row) for row in cursor.fetchall()]
    except Exception:
        return []


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="BIZRA Benchmark Dashboard Generator"
    )
    parser.add_argument(
        "--html", action="store_true",
        help="Generate HTML dashboard"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output JSON summary"
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print text summary"
    )
    parser.add_argument(
        "--output", "-o",
        default=str(DASHBOARD_OUTPUT),
        help="Output path for HTML dashboard"
    )

    args = parser.parse_args()

    # Default to summary if no options specified
    if not (args.html or args.json or args.summary):
        args.summary = True

    # Load data
    print("Loading benchmark data...")
    baselines, comparisons, benchmarks = load_benchmark_files()

    print(f"Found: {len(baselines)} baseline files, {len(comparisons)} comparison files")

    # Compute summary
    summary = compute_summary(baselines, comparisons)

    # Output based on flags
    if args.summary:
        print_summary(summary)

    if args.json:
        output = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_runs": summary.total_runs,
            "models_tested": summary.models_tested,
            "best_model": {"name": summary.best_model, "accuracy": summary.best_model_accuracy},
            "mode_comparison": {
                "direct": summary.direct_accuracy,
                "routed": summary.routed_accuracy,
                "bizra": summary.bizra_accuracy,
                "bizra_improvement": summary.bizra_improvement,
            },
            "bizra_metrics": {
                "avg_ihsan": summary.avg_ihsan,
                "avg_snr": summary.avg_snr,
                "snr_tier": summary.snr_tier,
            },
            "trend": {
                "direction": summary.accuracy_trend,
                "delta": summary.accuracy_delta,
            },
        }
        print(json.dumps(output, indent=2))

    if args.html:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        generate_html_dashboard(baselines, comparisons, summary, output_path)


if __name__ == "__main__":
    main()
