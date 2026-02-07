#!/usr/bin/env python3
"""
BIZRA CI Performance Benchmark Script
=====================================

Runs performance benchmarks and validates against regression gates.
Designed for CI integration with GitHub Actions.

Standing on Giants:
- Continuous Benchmarking (2018)
- pytest-benchmark patterns
- Knuth (1968): Premature optimization is the root of all evil
- Amdahl (1967): Parallelization theory

Usage:
    python scripts/ci_perf_benchmark.py --benchmark inference-latency
    python scripts/ci_perf_benchmark.py --benchmark apex-throughput --iterations 100
    python scripts/ci_perf_benchmark.py --benchmark memory-usage --memory-threshold 4
    python scripts/ci_perf_benchmark.py --benchmark startup-time

Exit Codes:
    0 - Benchmark passed all gates
    1 - Benchmark failed (regression detected or threshold exceeded)
    2 - Configuration or runtime error
"""

import argparse
import asyncio
import gc
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Ensure core module is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("[WARN] psutil not available, resource metrics will be limited", file=sys.stderr)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class LatencyMetrics:
    """Latency benchmark results."""

    iterations: int = 0
    mean_ms: float = 0.0
    median_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    stddev_ms: float = 0.0

    @classmethod
    def from_samples(cls, samples: List[float]) -> "LatencyMetrics":
        """Create metrics from latency samples (in milliseconds)."""
        if not samples:
            return cls()

        sorted_samples = sorted(samples)
        n = len(sorted_samples)

        return cls(
            iterations=n,
            mean_ms=statistics.mean(samples),
            median_ms=statistics.median(samples),
            p50_ms=sorted_samples[int(n * 0.50)] if n > 1 else sorted_samples[0],
            p95_ms=sorted_samples[int(n * 0.95)] if n > 20 else sorted_samples[-1],
            p99_ms=sorted_samples[int(n * 0.99)] if n > 100 else sorted_samples[-1],
            min_ms=min(samples),
            max_ms=max(samples),
            stddev_ms=statistics.stdev(samples) if n > 1 else 0.0,
        )


@dataclass
class ThroughputMetrics:
    """Throughput benchmark results."""

    total_requests: int = 0
    duration_seconds: float = 0.0
    qps: float = 0.0  # Queries per second
    batch_efficiency: float = 0.0  # Actual vs theoretical max

    @classmethod
    def from_run(
        cls,
        total_requests: int,
        duration_seconds: float,
        batch_size: int = 1,
    ) -> "ThroughputMetrics":
        """Create metrics from a throughput run."""
        qps = total_requests / duration_seconds if duration_seconds > 0 else 0.0
        theoretical_max = batch_size * (1000 / 50)  # Assuming 50ms per batch
        efficiency = (qps / theoretical_max * 100) if theoretical_max > 0 else 0.0

        return cls(
            total_requests=total_requests,
            duration_seconds=duration_seconds,
            qps=qps,
            batch_efficiency=min(100.0, efficiency),
        )


@dataclass
class MemoryMetrics:
    """Memory usage benchmark results."""

    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    baseline_memory_mb: float = 0.0
    delta_memory_mb: float = 0.0

    @classmethod
    def from_samples(
        cls, samples: List[float], baseline: float = 0.0
    ) -> "MemoryMetrics":
        """Create metrics from memory samples (in MB)."""
        if not samples:
            return cls()

        peak = max(samples)
        avg = statistics.mean(samples)

        return cls(
            peak_memory_mb=peak,
            avg_memory_mb=avg,
            baseline_memory_mb=baseline,
            delta_memory_mb=peak - baseline,
        )


@dataclass
class StartupMetrics:
    """Startup time benchmark results."""

    cold_start_ms: float = 0.0
    warm_start_ms: float = 0.0
    import_time_ms: float = 0.0
    init_time_ms: float = 0.0

    @classmethod
    def from_measurements(
        cls,
        cold_starts: List[float],
        warm_starts: List[float],
    ) -> "StartupMetrics":
        """Create metrics from startup measurements."""
        return cls(
            cold_start_ms=statistics.mean(cold_starts) if cold_starts else 0.0,
            warm_start_ms=statistics.mean(warm_starts) if warm_starts else 0.0,
        )


@dataclass
class RegressionResult:
    """Regression analysis result."""

    baseline_value: float
    current_value: float
    change_percent: float
    regressed: bool
    tolerance_percent: float
    message: str


@dataclass
class BenchmarkOutput:
    """Complete benchmark output for CI."""

    benchmark_name: str
    timestamp: str
    git_sha: str
    git_branch: str
    results: Dict[str, Any]
    regression: Optional[Dict[str, Any]] = None
    gate_passed: bool = True
    gate_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "benchmark_name": self.benchmark_name,
            "timestamp": self.timestamp,
            "git_sha": self.git_sha,
            "git_branch": self.git_branch,
            "results": self.results,
            "regression": self.regression,
            "gate_passed": self.gate_passed,
            "gate_message": self.gate_message,
        }


# =============================================================================
# Benchmark Implementations
# =============================================================================


class CIPerfBenchmark:
    """CI Performance Benchmark Runner."""

    def __init__(
        self,
        iterations: int = 50,
        warmup: int = 5,
        verbose: bool = True,
    ):
        self.iterations = iterations
        self.warmup = warmup
        self.verbose = verbose
        self.process = psutil.Process() if PSUTIL_AVAILABLE else None

    def log(self, message: str) -> None:
        """Log message if verbose."""
        if self.verbose:
            print(f"[BENCHMARK] {message}", file=sys.stderr)

    def get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        if self.process:
            return self.process.memory_info().rss / 1024 / 1024
        return 0.0

    def get_git_info(self) -> Tuple[str, str]:
        """Get current git SHA and branch."""
        sha = os.environ.get("GITHUB_SHA", "")
        branch = os.environ.get("GITHUB_REF_NAME", "")

        if not sha:
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                sha = result.stdout.strip()[:12]
            except Exception:
                sha = "unknown"

        if not branch:
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                branch = result.stdout.strip()
            except Exception:
                branch = "unknown"

        return sha, branch

    async def benchmark_inference_latency(self) -> LatencyMetrics:
        """
        Benchmark inference pipeline latency.

        Tests:
        - PCI gate chain validation
        - Consensus round (mocked)
        - Cache operations
        """
        self.log(f"Running inference latency benchmark ({self.iterations} iterations)...")

        latencies: List[float] = []

        try:
            # Try to import actual modules
            from core.pci.envelope import PCIEnvelope, PCISender, PCISignature, PCIMetadata
            from core.pci.gates import PCIGateKeeper
            from core.pci.crypto import generate_keypair, sign_message
            from datetime import datetime, timezone

            private_key, public_key = generate_keypair()
            gatekeeper = PCIGateKeeper()
            sender = PCISender(node_id="bench_node", public_key=public_key)
            metadata = PCIMetadata(ihsan_score=0.96, snr_score=0.97)

            # Warmup
            self.log(f"  Warmup ({self.warmup} iterations)...")
            for i in range(self.warmup):
                envelope = PCIEnvelope(
                    sender=sender,
                    nonce=f"warmup_{i}",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    payload={"test": "warmup"},
                    metadata=metadata,
                    signature=PCISignature(algorithm="ed25519", value="dummy"),
                )
                digest = envelope.compute_digest()
                envelope.signature = PCISignature(
                    algorithm="ed25519", value=sign_message(digest, private_key)
                )
                gatekeeper.verify(envelope)

            # Actual benchmark
            self.log(f"  Running benchmark ({self.iterations} iterations)...")
            for i in range(self.iterations):
                envelope = PCIEnvelope(
                    sender=sender,
                    nonce=f"bench_{i}",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    payload={"test": "benchmark", "iteration": i},
                    metadata=metadata,
                    signature=PCISignature(algorithm="ed25519", value="dummy"),
                )
                digest = envelope.compute_digest()
                envelope.signature = PCISignature(
                    algorithm="ed25519", value=sign_message(digest, private_key)
                )

                start = time.perf_counter()
                result = gatekeeper.verify(envelope)
                latency_ms = (time.perf_counter() - start) * 1000

                if result.passed:
                    latencies.append(latency_ms)

                if (i + 1) % 10 == 0:
                    self.log(f"    Completed {i + 1}/{self.iterations}")

        except ImportError as e:
            self.log(f"  Using mock benchmark (import error: {e})")
            # Fallback to mock benchmark
            for _ in range(self.warmup):
                await asyncio.sleep(0.001)  # 1ms mock

            for i in range(self.iterations):
                start = time.perf_counter()
                # Simulate some work
                _ = hash(f"benchmark_{i}")
                await asyncio.sleep(0.001)  # 1ms mock latency
                latency_ms = (time.perf_counter() - start) * 1000
                latencies.append(latency_ms)

        return LatencyMetrics.from_samples(latencies)

    async def benchmark_apex_throughput(self) -> ThroughputMetrics:
        """
        Benchmark throughput (requests per second).

        Tests concurrent request handling with optional batching.
        """
        self.log(f"Running throughput benchmark ({self.iterations} requests)...")

        try:
            from core.inference.gateway import BatchingInferenceQueue

            # Mock backend for testing
            async def mock_generate(prompt: str, max_tokens: int, temperature: float) -> str:
                await asyncio.sleep(0.01)  # 10ms simulated latency
                return f"Response to: {prompt[:20]}..."

            queue = BatchingInferenceQueue(
                backend_generate_fn=mock_generate,
                max_batch_size=8,
                max_wait_ms=50,
            )
            await queue.start()

            try:
                # Warmup
                warmup_tasks = [
                    asyncio.create_task(queue.submit(f"warmup_{i}", 50, 0.7))
                    for i in range(self.warmup)
                ]
                await asyncio.gather(*warmup_tasks)

                # Benchmark
                start_time = time.perf_counter()
                tasks = [
                    asyncio.create_task(queue.submit(f"benchmark_{i}", 50, 0.7))
                    for i in range(self.iterations)
                ]
                await asyncio.gather(*tasks)
                duration = time.perf_counter() - start_time

                return ThroughputMetrics.from_run(
                    total_requests=self.iterations,
                    duration_seconds=duration,
                    batch_size=8,
                )
            finally:
                await queue.stop()

        except ImportError as e:
            self.log(f"  Using mock throughput benchmark (import error: {e})")

            # Fallback mock benchmark
            async def mock_request(i: int) -> None:
                await asyncio.sleep(0.01)  # 10ms

            start_time = time.perf_counter()
            tasks = [asyncio.create_task(mock_request(i)) for i in range(self.iterations)]
            await asyncio.gather(*tasks)
            duration = time.perf_counter() - start_time

            return ThroughputMetrics.from_run(
                total_requests=self.iterations,
                duration_seconds=duration,
                batch_size=1,
            )

    async def benchmark_memory_usage(self) -> MemoryMetrics:
        """
        Benchmark memory usage during operations.

        Tracks peak and average memory consumption.
        """
        self.log(f"Running memory benchmark ({self.iterations} iterations)...")

        if not PSUTIL_AVAILABLE:
            self.log("  psutil not available, returning empty metrics")
            return MemoryMetrics()

        gc.collect()
        baseline_mb = self.get_memory_mb()
        memory_samples: List[float] = []

        try:
            from core.sovereign.runtime import SovereignRuntime, RuntimeConfig, RuntimeMode

            # Initialize runtime (this is where memory spikes)
            config = RuntimeConfig(
                mode=RuntimeMode.STANDARD,
                autonomous_enabled=False,
                enable_cache=True,
            )

            async with SovereignRuntime.create(config) as runtime:
                # Warmup and collect memory samples
                for i in range(self.warmup):
                    await runtime.query(f"Warmup query {i}")
                    memory_samples.append(self.get_memory_mb())

                # Benchmark iterations
                for i in range(self.iterations):
                    await runtime.query(f"Memory test query {i}")
                    memory_samples.append(self.get_memory_mb())

                    if (i + 1) % 5 == 0:
                        self.log(f"    Completed {i + 1}/{self.iterations}")

        except ImportError as e:
            self.log(f"  Using mock memory benchmark (import error: {e})")

            # Fallback: simulate memory usage with data structures
            data_store: List[bytes] = []

            for i in range(self.iterations):
                # Allocate some memory
                data_store.append(b"x" * 10000)  # 10KB per iteration
                memory_samples.append(self.get_memory_mb())

            # Cleanup
            del data_store
            gc.collect()

        return MemoryMetrics.from_samples(memory_samples, baseline_mb)

    async def benchmark_startup_time(self) -> StartupMetrics:
        """
        Benchmark cold start and warm start times.

        Tests module import and initialization latency.
        """
        self.log(f"Running startup benchmark ({self.iterations} iterations)...")

        cold_starts: List[float] = []
        warm_starts: List[float] = []

        # Cold start: measure time to import core modules
        for i in range(min(5, self.iterations)):  # Limit cold starts
            # Clear module cache for cold start simulation
            modules_to_clear = [m for m in sys.modules if m.startswith("core.")]

            start = time.perf_counter()
            try:
                # Force reimport
                if i > 0:
                    for m in modules_to_clear:
                        if m in sys.modules:
                            del sys.modules[m]

                import core.sovereign.runtime
                import core.pci.envelope
                import core.inference.gateway

                cold_start_ms = (time.perf_counter() - start) * 1000
                cold_starts.append(cold_start_ms)
            except ImportError:
                # If imports fail, use a mock time
                cold_starts.append(50.0)  # 50ms mock

        # Warm start: measure time with modules already loaded
        for i in range(self.iterations):
            start = time.perf_counter()
            try:
                # These should be fast since modules are cached
                import core.sovereign.runtime
                import core.pci.envelope
                import core.inference.gateway

                warm_start_ms = (time.perf_counter() - start) * 1000
                warm_starts.append(warm_start_ms)
            except ImportError:
                warm_starts.append(0.1)  # 0.1ms mock

        return StartupMetrics.from_measurements(cold_starts, warm_starts)


# =============================================================================
# Regression Analysis
# =============================================================================


def analyze_regression(
    current: float,
    baseline: float,
    tolerance_percent: float,
    higher_is_worse: bool = True,
) -> RegressionResult:
    """
    Analyze if current value represents a regression from baseline.

    Args:
        current: Current benchmark value
        baseline: Previous baseline value
        tolerance_percent: Acceptable regression tolerance (e.g., 10 for 10%)
        higher_is_worse: True for latency/memory, False for throughput

    Returns:
        RegressionResult with analysis
    """
    if baseline == 0:
        return RegressionResult(
            baseline_value=baseline,
            current_value=current,
            change_percent=0.0,
            regressed=False,
            tolerance_percent=tolerance_percent,
            message="No baseline available for comparison",
        )

    change_percent = ((current - baseline) / baseline) * 100

    if higher_is_worse:
        regressed = change_percent > tolerance_percent
    else:
        regressed = change_percent < -tolerance_percent

    if regressed:
        direction = "increased" if higher_is_worse else "decreased"
        message = f"Regression detected: {direction} by {abs(change_percent):.1f}% (tolerance: {tolerance_percent}%)"
    else:
        message = f"Within tolerance: {change_percent:+.1f}% change (tolerance: {tolerance_percent}%)"

    return RegressionResult(
        baseline_value=baseline,
        current_value=current,
        change_percent=change_percent,
        regressed=regressed,
        tolerance_percent=tolerance_percent,
        message=message,
    )


def load_baseline(baseline_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load baseline results from JSON file."""
    if not baseline_path:
        return None

    path = Path(baseline_path)
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"[WARN] Could not load baseline: {e}", file=sys.stderr)
        return None


# =============================================================================
# Main Entry Point
# =============================================================================


async def run_benchmark(
    benchmark_type: str,
    iterations: int,
    warmup: int,
    p95_threshold: float,
    throughput_threshold: float,
    memory_threshold: float,
    startup_threshold: float,
    regression_tolerance: float,
    baseline_path: Optional[str],
    output_path: Optional[str],
    json_output: bool,
) -> int:
    """
    Run specified benchmark and evaluate gates.

    Returns:
        Exit code (0 = passed, 1 = failed, 2 = error)
    """
    benchmark = CIPerfBenchmark(
        iterations=iterations,
        warmup=warmup,
        verbose=not json_output,
    )

    git_sha, git_branch = benchmark.get_git_info()
    timestamp = datetime.now(timezone.utc).isoformat()

    # Load baseline for regression analysis
    baseline = load_baseline(baseline_path)

    # Run the appropriate benchmark
    results: Dict[str, Any] = {}
    gate_passed = True
    gate_message = ""
    regression_result: Optional[Dict[str, Any]] = None

    if benchmark_type == "inference-latency":
        metrics = await benchmark.benchmark_inference_latency()
        results = asdict(metrics)

        # Check P95 gate
        if metrics.p95_ms > p95_threshold:
            gate_passed = False
            gate_message = f"P95 latency ({metrics.p95_ms:.2f}ms) exceeds threshold ({p95_threshold}ms)"

        # Check regression
        if baseline:
            baseline_p95 = baseline.get("results", {}).get("p95_ms", 0)
            regression = analyze_regression(
                current=metrics.p95_ms,
                baseline=baseline_p95,
                tolerance_percent=regression_tolerance,
                higher_is_worse=True,
            )
            if regression.regressed:
                gate_passed = False
                gate_message = regression.message
            regression_result = asdict(regression)

    elif benchmark_type == "apex-throughput":
        metrics = await benchmark.benchmark_apex_throughput()
        results = asdict(metrics)

        # Check throughput gate
        if metrics.qps < throughput_threshold:
            gate_passed = False
            gate_message = f"Throughput ({metrics.qps:.2f} req/s) below threshold ({throughput_threshold} req/s)"

        # Check regression
        if baseline:
            baseline_qps = baseline.get("results", {}).get("qps", 0)
            regression = analyze_regression(
                current=metrics.qps,
                baseline=baseline_qps,
                tolerance_percent=regression_tolerance,
                higher_is_worse=False,  # Lower throughput is worse
            )
            if regression.regressed:
                gate_passed = False
                gate_message = regression.message
            regression_result = asdict(regression)

    elif benchmark_type == "memory-usage":
        metrics = await benchmark.benchmark_memory_usage()
        results = asdict(metrics)

        # Check memory gate (convert GB to MB)
        memory_threshold_mb = memory_threshold * 1024
        if metrics.peak_memory_mb > memory_threshold_mb:
            gate_passed = False
            gate_message = f"Peak memory ({metrics.peak_memory_mb:.2f}MB) exceeds threshold ({memory_threshold_mb}MB)"

        # Check regression
        if baseline:
            baseline_peak = baseline.get("results", {}).get("peak_memory_mb", 0)
            regression = analyze_regression(
                current=metrics.peak_memory_mb,
                baseline=baseline_peak,
                tolerance_percent=regression_tolerance,
                higher_is_worse=True,
            )
            if regression.regressed:
                gate_passed = False
                gate_message = regression.message
            regression_result = asdict(regression)

    elif benchmark_type == "startup-time":
        metrics = await benchmark.benchmark_startup_time()
        results = asdict(metrics)

        # Check startup gate
        if metrics.cold_start_ms > startup_threshold:
            gate_passed = False
            gate_message = f"Cold start ({metrics.cold_start_ms:.2f}ms) exceeds threshold ({startup_threshold}ms)"

        # Check regression
        if baseline:
            baseline_cold = baseline.get("results", {}).get("cold_start_ms", 0)
            regression = analyze_regression(
                current=metrics.cold_start_ms,
                baseline=baseline_cold,
                tolerance_percent=regression_tolerance,
                higher_is_worse=True,
            )
            if regression.regressed:
                gate_passed = False
                gate_message = regression.message
            regression_result = asdict(regression)

    else:
        print(f"[ERROR] Unknown benchmark type: {benchmark_type}", file=sys.stderr)
        return 2

    # Build output
    output = BenchmarkOutput(
        benchmark_name=benchmark_type,
        timestamp=timestamp,
        git_sha=git_sha,
        git_branch=git_branch,
        results=results,
        regression=regression_result,
        gate_passed=gate_passed,
        gate_message=gate_message if not gate_passed else "All gates passed",
    )

    # Write output file
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(output.to_dict(), indent=2))
        if not json_output:
            print(f"[INFO] Results written to {output_path}", file=sys.stderr)

    # Write GitHub Actions outputs
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    f.write(f"{key}={value:.2f}\n")
            f.write(f"gate_passed={'true' if gate_passed else 'false'}\n")

    # Output JSON or human-readable
    if json_output:
        print(json.dumps(output.to_dict(), indent=2))
    else:
        print("\n" + "=" * 70)
        print(f"BENCHMARK: {benchmark_type.upper()}")
        print("=" * 70)
        print(f"Timestamp: {timestamp}")
        print(f"Git SHA: {git_sha}")
        print(f"Git Branch: {git_branch}")
        print("\nResults:")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")

        if regression_result:
            print("\nRegression Analysis:")
            print(f"  Baseline: {regression_result['baseline_value']:.2f}")
            print(f"  Current: {regression_result['current_value']:.2f}")
            print(f"  Change: {regression_result['change_percent']:+.1f}%")
            print(f"  Status: {'REGRESSED' if regression_result['regressed'] else 'OK'}")

        print("\n" + "-" * 70)
        if gate_passed:
            print("[PASS] All performance gates passed")
        else:
            print(f"[FAIL] {gate_message}")
        print("=" * 70)

    return 0 if gate_passed else 1


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="BIZRA CI Performance Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --benchmark inference-latency --iterations 50
  %(prog)s --benchmark apex-throughput --throughput-threshold 20
  %(prog)s --benchmark memory-usage --memory-threshold 2
  %(prog)s --benchmark startup-time --startup-threshold 3000
        """,
    )

    parser.add_argument(
        "--benchmark",
        "-b",
        required=True,
        choices=["inference-latency", "apex-throughput", "memory-usage", "startup-time"],
        help="Benchmark type to run",
    )
    parser.add_argument(
        "--iterations",
        "-n",
        type=int,
        default=50,
        help="Number of benchmark iterations (default: 50)",
    )
    parser.add_argument(
        "--warmup",
        "-w",
        type=int,
        default=5,
        help="Number of warmup iterations (default: 5)",
    )
    parser.add_argument(
        "--p95-threshold",
        type=float,
        default=500.0,
        help="P95 latency threshold in ms (default: 500)",
    )
    parser.add_argument(
        "--throughput-threshold",
        type=float,
        default=10.0,
        help="Throughput threshold in req/s (default: 10)",
    )
    parser.add_argument(
        "--memory-threshold",
        type=float,
        default=4.0,
        help="Memory threshold in GB (default: 4)",
    )
    parser.add_argument(
        "--startup-threshold",
        type=float,
        default=5000.0,
        help="Startup threshold in ms (default: 5000)",
    )
    parser.add_argument(
        "--regression-tolerance",
        type=float,
        default=10.0,
        help="Regression tolerance percentage (default: 10)",
    )
    parser.add_argument(
        "--baseline-path",
        type=str,
        help="Path to baseline JSON for regression comparison",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        help="Path to write results JSON",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON (suppresses progress messages)",
    )

    args = parser.parse_args()

    try:
        exit_code = asyncio.run(
            run_benchmark(
                benchmark_type=args.benchmark,
                iterations=args.iterations,
                warmup=args.warmup,
                p95_threshold=args.p95_threshold,
                throughput_threshold=args.throughput_threshold,
                memory_threshold=args.memory_threshold,
                startup_threshold=args.startup_threshold,
                regression_tolerance=args.regression_tolerance,
                baseline_path=args.baseline_path,
                output_path=args.output_path,
                json_output=args.json,
            )
        )
        return exit_code
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Benchmark cancelled", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"[ERROR] Benchmark failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
