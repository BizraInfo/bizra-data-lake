"""
BIZRA Benchmark Suite - Main Entry Point

Usage:
    python -m benchmark                    # Run all suites
    python -m benchmark inference          # Run inference suite
    python -m benchmark security           # Run security suite
    python -m benchmark quality            # Run quality suite
    python -m benchmark --iterations 50    # Custom iteration count
    python -m benchmark --json output.json # JSON output

Output:
    - Console table with min/avg/max/p95 metrics
    - Overall score (weighted)
    - Bottleneck identification
"""

import argparse
import json
import sys
from pathlib import Path

from benchmark.runner import BenchmarkRunner
from benchmark.suites.inference import InferenceBenchmark
from benchmark.suites.security import SecurityBenchmark
from benchmark.suites.quality import QualityBenchmark


def compute_overall_score(
    all_results: dict,
    reference: dict = None,
) -> float:
    """
    Compute overall benchmark score (0-100).

    Weights suites by importance and normalizes against reference.
    """
    if reference is None:
        reference = {
            "inference": 50.0,  # Throughput is important
            "security": 0.5,    # Latency should be tiny
            "quality": 2.0,     # Validation should be fast
        }

    weights = {
        "inference": 0.5,
        "security": 0.2,
        "quality": 0.3,
    }

    scores = {}
    for suite_name, suite_results in all_results.items():
        if suite_name not in weights:
            continue

        suite_latencies = []
        for bench_name, bench_result in suite_results.items():
            if "metrics" in bench_result:
                for metric_name, metric in bench_result["metrics"].items():
                    if "ms" in metric["unit"]:
                        suite_latencies.append(metric["avg"])
                    elif "qps" in metric["unit"]:
                        # Inverse: higher is better
                        suite_latencies.append(100.0 / max(metric["avg"], 1.0))

        if suite_latencies:
            avg_latency = sum(suite_latencies) / len(suite_latencies)
            ref_latency = reference.get(suite_name, 10.0)
            score = min(100.0, (ref_latency / max(avg_latency, 0.001)) * 100)
            scores[suite_name] = score

    # Weighted average
    if scores:
        overall = sum(
            scores.get(name, 50.0) * weight
            for name, weight in weights.items()
        )
        return min(100.0, overall)

    return 50.0


def print_summary_table(all_results: dict, overall_score: float) -> None:
    """Print human-readable summary table."""
    print("\n" + "=" * 70)
    print("BIZRA BENCHMARK SUITE RESULTS")
    print("=" * 70)

    for suite_name, suite_results in all_results.items():
        print(f"\n{suite_name.upper()}")
        print("-" * 70)

        for bench_name, bench_result in suite_results.items():
            if "metrics" not in bench_result:
                continue

            for metric_name, metric in bench_result["metrics"].items():
                unit = metric.get("unit", "")
                avg = metric.get("avg", 0.0)
                p95 = metric.get("p95", 0.0)

                print(
                    f"  {bench_name:40s}: "
                    f"{avg:8.2f} {unit:6s} (p95: {p95:8.2f} {unit})"
                )

    print("\n" + "=" * 70)
    print(f"OVERALL SCORE: {overall_score:6.1f}/100")
    print("=" * 70)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="BIZRA Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m benchmark                    # Run all suites
  python -m benchmark inference          # Run inference suite only
  python -m benchmark --iterations 50    # 50 iterations per benchmark
  python -m benchmark --json results.json # Save JSON output
        """,
    )

    parser.add_argument(
        "suite",
        nargs="?",
        choices=["all", "inference", "security", "quality"],
        default="all",
        help="Which suite to run (default: all)",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Iterations per benchmark (default: 20)",
    )

    parser.add_argument(
        "--json",
        dest="json_output",
        help="Write JSON output to file",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output during benchmarks",
    )

    args = parser.parse_args()

    # Create runner
    runner = BenchmarkRunner(warmup=2, verbose=args.verbose)

    print(f"\nStarting BIZRA Benchmark Suite")
    print(f"Iterations: {args.iterations} | Suite: {args.suite}")
    print("-" * 70)

    all_results = {}

    # Run selected suites
    if args.suite in ("all", "inference"):
        inference = InferenceBenchmark(runner)
        all_results["inference"] = inference.run_all(args.iterations)

    if args.suite in ("all", "security"):
        security = SecurityBenchmark(runner)
        all_results["security"] = security.run_all(args.iterations)

    if args.suite in ("all", "quality"):
        quality = QualityBenchmark(runner)
        all_results["quality"] = quality.run_all(args.iterations)

    # Compute overall score
    overall_score = compute_overall_score(all_results)

    # Print summary
    print_summary_table(all_results, overall_score)

    # JSON output if requested
    if args.json_output:
        output = {
            "overall_score": overall_score,
            "iterations": args.iterations,
            "suites": all_results,
        }
        output_path = Path(args.json_output)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"JSON output written to: {output_path}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
