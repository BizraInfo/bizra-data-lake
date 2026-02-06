"""
Bench-as-Receipt Harness — Performance Claims as Sealed Evidence

Performance benchmarks are not just metrics; they are cryptographically
signed receipts that prove what happened during execution.

Key capabilities:
- Capture execution metrics (p99, allocations, throughput)
- Seal metrics into receipt format
- Verify benchmark claims against sealed evidence
- Detect performance regressions via receipt comparison
"""

import gc
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, TypeVar

from core.proof_engine.canonical import (
    CanonEnvironment,
    blake3_digest,
    canonical_bytes,
)
from core.proof_engine.receipt import (
    Metrics,
    SovereignSigner,
)

T = TypeVar("T")


@dataclass
class BenchSample:
    """Single benchmark sample."""

    iteration: int
    duration_ns: int
    allocs: int
    peak_memory_bytes: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "duration_ns": self.duration_ns,
            "allocs": self.allocs,
            "peak_memory_bytes": self.peak_memory_bytes,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class BenchResult:
    """
    Aggregated benchmark result.

    Contains percentile statistics for timing and memory.
    """

    name: str
    iterations: int
    samples: List[BenchSample]

    # Timing stats (nanoseconds)
    min_ns: int = 0
    max_ns: int = 0
    mean_ns: float = 0.0
    p50_ns: int = 0
    p95_ns: int = 0
    p99_ns: int = 0

    # Memory stats
    total_allocs: int = 0
    peak_memory_bytes: int = 0

    # Environment
    environment: Optional[CanonEnvironment] = None

    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        """Compute statistics from samples."""
        if not self.samples:
            return

        durations = sorted(s.duration_ns for s in self.samples)
        self.min_ns = durations[0]
        self.max_ns = durations[-1]
        self.mean_ns = sum(durations) / len(durations)
        self.p50_ns = self._percentile(durations, 50)
        self.p95_ns = self._percentile(durations, 95)
        self.p99_ns = self._percentile(durations, 99)

        self.total_allocs = sum(s.allocs for s in self.samples)
        self.peak_memory_bytes = max(s.peak_memory_bytes for s in self.samples)

    def _percentile(self, data: List[int], p: int) -> int:
        """Compute percentile of sorted data."""
        if not data:
            return 0
        k = (len(data) - 1) * p / 100
        f = int(k)
        c = f + 1 if f < len(data) - 1 else f
        return int(data[f] + (data[c] - data[f]) * (k - f))

    @property
    def p99_us(self) -> int:
        """P99 latency in microseconds."""
        return self.p99_ns // 1000

    @property
    def throughput_ops(self) -> float:
        """Operations per second."""
        if self.mean_ns <= 0:
            return 0.0
        return 1_000_000_000 / self.mean_ns

    def to_metrics(self) -> Metrics:
        """Convert to receipt Metrics."""
        return Metrics(
            p99_us=self.p99_us,
            allocs=self.total_allocs,
            memory_bytes=self.peak_memory_bytes,
            duration_ms=self.mean_ns / 1_000_000,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "iterations": self.iterations,
            "min_ns": self.min_ns,
            "max_ns": self.max_ns,
            "mean_ns": self.mean_ns,
            "p50_ns": self.p50_ns,
            "p95_ns": self.p95_ns,
            "p99_ns": self.p99_ns,
            "p99_us": self.p99_us,
            "throughput_ops": self.throughput_ops,
            "total_allocs": self.total_allocs,
            "peak_memory_bytes": self.peak_memory_bytes,
            "environment": (
                self.environment.canonical_bytes().hex() if self.environment else None
            ),
            "timestamp": self.timestamp.isoformat(),
        }

    def canonical_bytes(self) -> bytes:
        """Get deterministic byte representation."""
        return canonical_bytes(self.to_dict())

    def digest(self) -> bytes:
        """Compute BLAKE3 digest."""
        return blake3_digest(self.canonical_bytes())

    def hex_digest(self) -> str:
        """Compute hex-encoded digest."""
        return self.digest().hex()


@dataclass
class BenchReceipt:
    """
    Sealed benchmark receipt.

    The proof of performance claims.
    """

    receipt_id: str
    bench_result: BenchResult
    claims_verified: bool

    # Claimed vs actual
    claimed_p99_us: int
    actual_p99_us: int
    claimed_throughput: float
    actual_throughput: float
    claimed_allocs: int
    actual_allocs: int

    # Verdict
    verdict: str  # "PASS", "FAIL", "DEGRADED"
    degradation_pct: float = 0.0

    # Signature
    signature: bytes = field(default_factory=bytes)
    signer_pubkey: bytes = field(default_factory=bytes)

    # Timestamp
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def body_bytes(self) -> bytes:
        """Get receipt body for signing."""
        data = {
            "receipt_id": self.receipt_id,
            "bench_digest": self.bench_result.hex_digest(),
            "claims_verified": self.claims_verified,
            "claimed_p99_us": self.claimed_p99_us,
            "actual_p99_us": self.actual_p99_us,
            "claimed_throughput": self.claimed_throughput,
            "actual_throughput": self.actual_throughput,
            "claimed_allocs": self.claimed_allocs,
            "actual_allocs": self.actual_allocs,
            "verdict": self.verdict,
            "degradation_pct": self.degradation_pct,
            "timestamp": self.timestamp.isoformat(),
        }
        return canonical_bytes(data)

    def sign_with(self, signer: SovereignSigner) -> "BenchReceipt":
        """Sign the receipt."""
        body = self.body_bytes()
        self.signature = signer.sign(body)
        self.signer_pubkey = signer.public_key_bytes()
        return self

    def verify_signature(self, signer: SovereignSigner) -> bool:
        """Verify signature."""
        body = self.body_bytes()
        return signer.verify(body, self.signature)

    def digest(self) -> bytes:
        """Compute receipt digest."""
        return blake3_digest(self.body_bytes() + self.signature)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "receipt_id": self.receipt_id,
            "bench_result": self.bench_result.to_dict(),
            "claims_verified": self.claims_verified,
            "claimed_p99_us": self.claimed_p99_us,
            "actual_p99_us": self.actual_p99_us,
            "claimed_throughput": self.claimed_throughput,
            "actual_throughput": self.actual_throughput,
            "claimed_allocs": self.claimed_allocs,
            "actual_allocs": self.actual_allocs,
            "verdict": self.verdict,
            "degradation_pct": self.degradation_pct,
            "signature": self.signature.hex(),
            "signer_pubkey": self.signer_pubkey.hex(),
            "timestamp": self.timestamp.isoformat(),
            "receipt_digest": self.digest().hex(),
        }


class BenchHarness:
    """
    Benchmark harness with receipt generation.

    Runs benchmarks and seals results into verifiable receipts.
    """

    def __init__(
        self,
        signer: SovereignSigner,
        warmup_iterations: int = 3,
        bench_iterations: int = 100,
    ):
        self.signer = signer
        self.warmup_iterations = warmup_iterations
        self.bench_iterations = bench_iterations
        self._receipt_counter = 0
        self._results: List[BenchResult] = []
        self._receipts: List[BenchReceipt] = []

    def _next_receipt_id(self) -> str:
        """Generate next receipt ID."""
        self._receipt_counter += 1
        return f"bench_{self._receipt_counter:08d}_{int(time.time() * 1000)}"

    @contextmanager
    def _track_memory(self) -> Generator[Callable[[], Tuple[int, int]], None, None]:
        """Context manager to track memory allocations."""
        gc.collect()
        tracemalloc.start()
        start_snapshot = tracemalloc.take_snapshot()

        def get_stats() -> Tuple[int, int]:
            """Get (allocation_count, peak_bytes)."""
            current = tracemalloc.take_snapshot()
            stats = current.compare_to(start_snapshot, "lineno")
            allocs = sum(stat.count for stat in stats)
            peak = tracemalloc.get_traced_memory()[1]
            return allocs, peak

        try:
            yield get_stats
        finally:
            tracemalloc.stop()

    def bench_fn(
        self,
        name: str,
        fn: Callable[[], T],
        iterations: Optional[int] = None,
    ) -> BenchResult:
        """
        Benchmark a function.

        Args:
            name: Benchmark name
            fn: Function to benchmark (called repeatedly)
            iterations: Number of iterations (default: self.bench_iterations)

        Returns:
            BenchResult with statistics
        """
        iterations = iterations or self.bench_iterations
        environment = CanonEnvironment.capture()
        samples: List[BenchSample] = []

        # Warmup
        for _ in range(self.warmup_iterations):
            fn()

        # Benchmark
        for i in range(iterations):
            gc.collect()

            with self._track_memory() as get_stats:
                start = time.perf_counter_ns()
                fn()
                end = time.perf_counter_ns()
                allocs, peak = get_stats()

            samples.append(
                BenchSample(
                    iteration=i,
                    duration_ns=end - start,
                    allocs=allocs,
                    peak_memory_bytes=peak,
                )
            )

        result = BenchResult(
            name=name,
            iterations=iterations,
            samples=samples,
            environment=environment,
        )

        self._results.append(result)
        return result

    def bench_to_receipt(
        self,
        result: BenchResult,
        claimed_p99_us: int,
        claimed_throughput: float,
        claimed_allocs: int,
        tolerance_pct: float = 10.0,
    ) -> BenchReceipt:
        """
        Convert benchmark result to sealed receipt.

        Args:
            result: Benchmark result
            claimed_p99_us: Claimed p99 latency in microseconds
            claimed_throughput: Claimed throughput in ops/sec
            claimed_allocs: Claimed allocation count
            tolerance_pct: Acceptable degradation percentage

        Returns:
            Signed BenchReceipt
        """
        actual_p99 = result.p99_us
        actual_throughput = result.throughput_ops
        actual_allocs = result.total_allocs

        # Check claims
        p99_ok = actual_p99 <= claimed_p99_us * (1 + tolerance_pct / 100)
        throughput_ok = actual_throughput >= claimed_throughput * (
            1 - tolerance_pct / 100
        )
        allocs_ok = actual_allocs <= claimed_allocs * (1 + tolerance_pct / 100)

        claims_verified = p99_ok and throughput_ok and allocs_ok

        # Calculate degradation
        degradation = 0.0
        if claimed_p99_us > 0:
            degradation = max(
                degradation, (actual_p99 - claimed_p99_us) / claimed_p99_us * 100
            )
        if claimed_throughput > 0:
            degradation = max(
                degradation,
                (claimed_throughput - actual_throughput) / claimed_throughput * 100,
            )

        # Determine verdict
        if claims_verified:
            verdict = "PASS"
        elif degradation <= tolerance_pct:
            verdict = "DEGRADED"
        else:
            verdict = "FAIL"

        receipt = BenchReceipt(
            receipt_id=self._next_receipt_id(),
            bench_result=result,
            claims_verified=claims_verified,
            claimed_p99_us=claimed_p99_us,
            actual_p99_us=actual_p99,
            claimed_throughput=claimed_throughput,
            actual_throughput=actual_throughput,
            claimed_allocs=claimed_allocs,
            actual_allocs=actual_allocs,
            verdict=verdict,
            degradation_pct=degradation,
        )

        receipt.sign_with(self.signer)
        self._receipts.append(receipt)

        return receipt

    def verify_receipt(self, receipt: BenchReceipt) -> Tuple[bool, Optional[str]]:
        """
        Verify a benchmark receipt.

        Returns (valid, error_message).
        """
        # Verify signature
        if not receipt.verify_signature(self.signer):
            return False, "Invalid signature"

        # Verify signer matches
        if receipt.signer_pubkey != self.signer.public_key_bytes():
            return False, "Signer mismatch"

        # Verify bench digest
        expected_digest = receipt.bench_result.hex_digest()
        if expected_digest != receipt.bench_result.hex_digest():
            return False, "Bench result tampered"

        return True, None

    def compare_receipts(
        self,
        baseline: BenchReceipt,
        current: BenchReceipt,
        regression_threshold_pct: float = 5.0,
    ) -> Dict[str, Any]:
        """
        Compare two receipts for regression detection.

        Args:
            baseline: Baseline (previous) receipt
            current: Current receipt
            regression_threshold_pct: Threshold for regression detection

        Returns:
            Comparison result with regression details
        """
        # P99 regression
        p99_delta_pct = 0.0
        if baseline.actual_p99_us > 0:
            p99_delta_pct = (
                (current.actual_p99_us - baseline.actual_p99_us)
                / baseline.actual_p99_us
                * 100
            )

        # Throughput regression
        throughput_delta_pct = 0.0
        if baseline.actual_throughput > 0:
            throughput_delta_pct = (
                (baseline.actual_throughput - current.actual_throughput)
                / baseline.actual_throughput
                * 100
            )

        # Allocs regression
        allocs_delta_pct = 0.0
        if baseline.actual_allocs > 0:
            allocs_delta_pct = (
                (current.actual_allocs - baseline.actual_allocs)
                / baseline.actual_allocs
                * 100
            )

        # Determine if regressed
        p99_regressed = p99_delta_pct > regression_threshold_pct
        throughput_regressed = throughput_delta_pct > regression_threshold_pct
        allocs_regressed = allocs_delta_pct > regression_threshold_pct

        has_regression = p99_regressed or throughput_regressed or allocs_regressed

        return {
            "baseline_receipt_id": baseline.receipt_id,
            "current_receipt_id": current.receipt_id,
            "has_regression": has_regression,
            "p99": {
                "baseline_us": baseline.actual_p99_us,
                "current_us": current.actual_p99_us,
                "delta_pct": p99_delta_pct,
                "regressed": p99_regressed,
            },
            "throughput": {
                "baseline_ops": baseline.actual_throughput,
                "current_ops": current.actual_throughput,
                "delta_pct": throughput_delta_pct,
                "regressed": throughput_regressed,
            },
            "allocs": {
                "baseline": baseline.actual_allocs,
                "current": current.actual_allocs,
                "delta_pct": allocs_delta_pct,
                "regressed": allocs_regressed,
            },
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get harness statistics."""
        if not self._receipts:
            return {
                "total_benchmarks": 0,
                "total_receipts": 0,
            }

        pass_count = sum(1 for r in self._receipts if r.verdict == "PASS")
        fail_count = sum(1 for r in self._receipts if r.verdict == "FAIL")
        degraded_count = sum(1 for r in self._receipts if r.verdict == "DEGRADED")

        return {
            "total_benchmarks": len(self._results),
            "total_receipts": len(self._receipts),
            "pass_count": pass_count,
            "fail_count": fail_count,
            "degraded_count": degraded_count,
            "pass_rate": pass_count / len(self._receipts) if self._receipts else 0.0,
        }

    def get_recent_receipts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent benchmark receipts."""
        return [r.to_dict() for r in self._receipts[-limit:]]


def bench_to_receipt(
    result: BenchResult,
    signer: SovereignSigner,
    claimed_p99_us: int,
    claimed_throughput: float,
    claimed_allocs: int,
    tolerance_pct: float = 10.0,
) -> BenchReceipt:
    """
    Convenience function to convert benchmark result to receipt.

    Args:
        result: Benchmark result
        signer: Signer for receipt
        claimed_p99_us: Claimed p99 latency in microseconds
        claimed_throughput: Claimed throughput in ops/sec
        claimed_allocs: Claimed allocation count
        tolerance_pct: Acceptable degradation percentage

    Returns:
        Signed BenchReceipt
    """
    harness = BenchHarness(signer)
    return harness.bench_to_receipt(
        result=result,
        claimed_p99_us=claimed_p99_us,
        claimed_throughput=claimed_throughput,
        claimed_allocs=claimed_allocs,
        tolerance_pct=tolerance_pct,
    )
