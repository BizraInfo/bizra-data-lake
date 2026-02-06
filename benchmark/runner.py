"""
BIZRA Benchmark Runner
Minimal, offline benchmarking framework measuring real metrics.

Metrics: latency_ms, throughput_qps, memory_mb
Reports: min/avg/max/p95 per benchmark

Standing on Giants: Knuth (measure, don't guess), Amdahl (identify bottlenecks)
"""

import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Any
import statistics


@dataclass
class MetricResult:
    """Result of a single metric measurement."""
    name: str
    unit: str
    values: List[float] = field(default_factory=list)

    @property
    def min_val(self) -> float:
        return min(self.values) if self.values else 0.0

    @property
    def max_val(self) -> float:
        return max(self.values) if self.values else 0.0

    @property
    def avg_val(self) -> float:
        return statistics.mean(self.values) if self.values else 0.0

    @property
    def p95_val(self) -> float:
        if len(self.values) < 2:
            return self.avg_val
        sorted_vals = sorted(self.values)
        idx = int(len(sorted_vals) * 0.95)
        return sorted_vals[idx]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "unit": self.unit,
            "min": self.min_val,
            "avg": self.avg_val,
            "max": self.max_val,
            "p95": self.p95_val,
            "count": len(self.values),
        }


@dataclass
class BenchmarkResult:
    """Result of a complete benchmark run."""
    name: str
    iterations: int
    metrics: Dict[str, MetricResult] = field(default_factory=dict)

    def add_metric(self, name: str, unit: str, value: float) -> None:
        """Add a measurement to a metric."""
        if name not in self.metrics:
            self.metrics[name] = MetricResult(name, unit)
        self.metrics[name].values.append(value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
        }


class BenchmarkRunner:
    """
    Runs benchmarks with automatic metric collection.

    Measures: latency_ms, throughput_qps, memory_mb
    Supports: Warmup iterations, automatic GC, per-iteration tracking
    """

    def __init__(self, warmup: int = 2, verbose: bool = False):
        self.warmup = warmup
        self.verbose = verbose
        self.results: Dict[str, BenchmarkResult] = {}

    def run(
        self,
        name: str,
        func: Callable[[], Any],
        iterations: int = 10,
        track_memory: bool = False,
    ) -> BenchmarkResult:
        """
        Run a benchmark function multiple times.

        Args:
            name: Benchmark name
            func: Function to benchmark (no args)
            iterations: Number of iterations
            track_memory: Enable memory tracking (slower)

        Returns:
            BenchmarkResult with metrics
        """
        result = BenchmarkResult(name, iterations)

        # Warmup
        if self.verbose:
            print(f"  Warming up {name}...")
        for _ in range(self.warmup):
            try:
                func()
            except Exception as e:
                if self.verbose:
                    print(f"    Warmup iteration failed: {e}")

        # Benchmark iterations
        if self.verbose:
            print(f"  Running {iterations} iterations...")

        for i in range(iterations):
            try:
                # Latency measurement
                start_time = time.perf_counter()
                if track_memory:
                    tracemalloc.start()

                func()

                elapsed_ms = (time.perf_counter() - start_time) * 1000
                result.add_metric("latency_ms", "ms", elapsed_ms)

                # Memory measurement
                if track_memory:
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    result.add_metric("memory_mb", "MB", peak / 1024 / 1024)

                if self.verbose and (i + 1) % max(1, iterations // 5) == 0:
                    print(f"    Completed {i + 1}/{iterations}")

            except Exception as e:
                if self.verbose:
                    print(f"    Iteration {i} failed: {e}")

        self.results[name] = result
        return result

    def run_throughput(
        self,
        name: str,
        func: Callable[[], int],
        iterations: int = 10,
    ) -> BenchmarkResult:
        """
        Benchmark throughput-oriented operations.

        Args:
            name: Benchmark name
            func: Function returning operation count
            iterations: Number of iterations

        Returns:
            BenchmarkResult with qps metric
        """
        result = BenchmarkResult(name, iterations)

        if self.verbose:
            print(f"  Warming up {name}...")
        for _ in range(self.warmup):
            try:
                func()
            except Exception as e:
                if self.verbose:
                    print(f"    Warmup failed: {e}")

        if self.verbose:
            print(f"  Running {iterations} iterations...")

        for i in range(iterations):
            try:
                start_time = time.perf_counter()
                ops = func()
                elapsed_s = time.perf_counter() - start_time

                if elapsed_s > 0:
                    qps = ops / elapsed_s
                    result.add_metric("throughput_qps", "qps", qps)

                if self.verbose and (i + 1) % max(1, iterations // 5) == 0:
                    print(f"    Completed {i + 1}/{iterations}")

            except Exception as e:
                if self.verbose:
                    print(f"    Iteration {i} failed: {e}")

        self.results[name] = result
        return result

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all benchmark results."""
        return {
            name: result.to_dict()
            for name, result in self.results.items()
        }

    def print_summary(self) -> None:
        """Print human-readable summary."""
        if not self.results:
            print("No benchmarks run.")
            return

        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)

        for name, result in self.results.items():
            print(f"\n{name}:")
            for metric_name, metric in result.metrics.items():
                print(
                    f"  {metric_name:30s}: "
                    f"{metric.avg_val:8.2f} {metric.unit} "
                    f"(min: {metric.min_val:8.2f}, "
                    f"max: {metric.max_val:8.2f}, "
                    f"p95: {metric.p95_val:8.2f})"
                )

        print("\n" + "=" * 60)
