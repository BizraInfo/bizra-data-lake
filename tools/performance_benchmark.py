#!/usr/bin/env python3
"""
BIZRA Performance Benchmark Suite
Standing on Giants: Knuth (1968), Amdahl (1967), Shannon (1948)

Usage:
    python tools/performance_benchmark.py --all
    python tools/performance_benchmark.py --inference --cache --consensus
    python tools/performance_benchmark.py --baseline > baseline.json
"""

import asyncio
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import psutil


@dataclass
class BenchmarkResult:
    """Performance benchmark result."""

    name: str
    iterations: int

    # Latency metrics (milliseconds)
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    stddev_ms: float

    # Throughput metrics
    qps: float  # Queries per second

    # Resource metrics
    cpu_percent: float
    memory_mb: float

    def improvement_over(self, baseline: "BenchmarkResult") -> float:
        """Calculate percentage improvement over baseline."""
        if baseline.mean_ms == 0:
            return 0.0
        return ((baseline.mean_ms - self.mean_ms) / baseline.mean_ms) * 100

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class PerformanceBenchmark:
    """Performance benchmarking framework with resource monitoring."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: List[BenchmarkResult] = []
        self.process = psutil.Process()

    def log(self, message: str):
        """Log message if verbose."""
        if self.verbose:
            print(f"[BENCHMARK] {message}", file=sys.stderr)

    async def benchmark_query_pipeline(
        self, iterations: int = 50, warmup: int = 5
    ) -> BenchmarkResult:
        """
        Benchmark end-to-end query pipeline.

        Tests:
        - Query parsing
        - Graph-of-Thoughts reasoning
        - LLM inference
        - SNR optimization
        - Guardian validation
        """
        self.log(f"Benchmarking query pipeline ({iterations} iterations, {warmup} warmup)...")

        try:
            from core.sovereign.runtime import SovereignRuntime, RuntimeConfig, RuntimeMode

            config = RuntimeConfig(
                mode=RuntimeMode.PRODUCTION,
                autonomous_enabled=False,  # Disable for clean benchmarks
                enable_cache=False,  # Disable cache for pure measurement
            )

            latencies = []
            cpu_samples = []
            memory_samples = []

            async with SovereignRuntime.create(config) as runtime:
                # Warmup
                for i in range(warmup):
                    await runtime.query(f"Warmup query {i}: What is sovereignty?")

                # Actual benchmark
                for i in range(iterations):
                    query = f"Test query {i}: Explain the concept of distributed consensus in Byzantine fault-tolerant systems."

                    # Measure resources before
                    cpu_before = self.process.cpu_percent(interval=None)
                    mem_before = self.process.memory_info().rss / 1024 / 1024

                    start = time.perf_counter()
                    result = await runtime.query(query)
                    latency_ms = (time.perf_counter() - start) * 1000

                    # Measure resources after
                    cpu_after = self.process.cpu_percent(interval=None)
                    mem_after = self.process.memory_info().rss / 1024 / 1024

                    if result.success:
                        latencies.append(latency_ms)
                        cpu_samples.append((cpu_before + cpu_after) / 2)
                        memory_samples.append(mem_after)

                    if (i + 1) % 10 == 0:
                        self.log(f"  Completed {i + 1}/{iterations} iterations")

            return self._create_result(
                "query_pipeline", len(latencies), latencies, cpu_samples, memory_samples
            )

        except ImportError as e:
            self.log(f"Skipping query_pipeline benchmark: {e}")
            return self._empty_result("query_pipeline")

    async def benchmark_inference_only(
        self, iterations: int = 100, warmup: int = 10
    ) -> BenchmarkResult:
        """
        Benchmark pure LLM inference (no reasoning/validation).

        Tests:
        - Model loading
        - Token generation speed
        - Backend switching
        """
        self.log(f"Benchmarking inference only ({iterations} iterations, {warmup} warmup)...")

        try:
            from core.inference.gateway import InferenceGateway, InferenceConfig

            config = InferenceConfig(require_local=False)
            gateway = InferenceGateway(config)

            if not await gateway.initialize():
                self.log("Failed to initialize gateway, skipping inference benchmark")
                return self._empty_result("inference_only")

            latencies = []
            cpu_samples = []
            memory_samples = []

            # Warmup
            for i in range(warmup):
                await gateway.infer("Warmup", max_tokens=10)

            # Actual benchmark
            for i in range(iterations):
                prompt = f"Explain concept {i} in one sentence."

                cpu_before = self.process.cpu_percent(interval=None)
                mem_before = self.process.memory_info().rss / 1024 / 1024

                start = time.perf_counter()
                result = await gateway.infer(prompt, max_tokens=50)
                latency_ms = (time.perf_counter() - start) * 1000

                cpu_after = self.process.cpu_percent(interval=None)
                mem_after = self.process.memory_info().rss / 1024 / 1024

                latencies.append(latency_ms)
                cpu_samples.append((cpu_before + cpu_after) / 2)
                memory_samples.append(mem_after)

                if (i + 1) % 20 == 0:
                    self.log(f"  Completed {i + 1}/{iterations} iterations")

            return self._create_result(
                "inference_only", len(latencies), latencies, cpu_samples, memory_samples
            )

        except ImportError as e:
            self.log(f"Skipping inference benchmark: {e}")
            return self._empty_result("inference_only")

    async def benchmark_consensus_round(
        self, iterations: int = 100, warmup: int = 10
    ) -> BenchmarkResult:
        """
        Benchmark PBFT consensus round (8 validators).

        Tests:
        - Signature creation
        - Signature verification
        - Vote aggregation
        - Quorum detection
        """
        self.log(f"Benchmarking consensus ({iterations} iterations, {warmup} warmup)...")

        try:
            from core.federation.consensus import ConsensusEngine
            from core.pci.crypto import generate_keypair

            # Create 8 validators
            node_count = 8
            validators = []

            for i in range(node_count):
                private_key, public_key = generate_keypair()
                validator = ConsensusEngine(f"node_{i}", private_key, public_key)
                validators.append(validator)

            # Register all peers with each other
            for v in validators:
                for peer in validators:
                    if v.node_id != peer.node_id:
                        v.register_peer(peer.node_id, peer.public_key)

            # Set leader
            validators[0].set_leader(validators[0].node_id)

            latencies = []
            cpu_samples = []
            memory_samples = []

            # Warmup
            for i in range(warmup):
                proposal = validators[0].propose_pattern({"test": i, "data": "warmup"})
                for v in validators:
                    vote = v.cast_vote(proposal, ihsan_score=0.95)
                    if vote:
                        v.receive_vote(vote, node_count)

            # Actual benchmark
            for i in range(iterations):
                cpu_before = self.process.cpu_percent(interval=None)
                mem_before = self.process.memory_info().rss / 1024 / 1024

                start = time.perf_counter()

                # Propose pattern
                proposal = validators[0].propose_pattern({"test": i, "data": "benchmark"})

                # Each validator votes
                for v in validators:
                    vote = v.cast_vote(proposal, ihsan_score=0.95)
                    if vote:
                        # Broadcast vote to all validators
                        for receiver in validators:
                            receiver.receive_vote(vote, node_count)

                latency_ms = (time.perf_counter() - start) * 1000

                cpu_after = self.process.cpu_percent(interval=None)
                mem_after = self.process.memory_info().rss / 1024 / 1024

                latencies.append(latency_ms)
                cpu_samples.append((cpu_before + cpu_after) / 2)
                memory_samples.append(mem_after)

                if (i + 1) % 20 == 0:
                    self.log(f"  Completed {i + 1}/{iterations} iterations")

            return self._create_result(
                "consensus_round", len(latencies), latencies, cpu_samples, memory_samples
            )

        except ImportError as e:
            self.log(f"Skipping consensus benchmark: {e}")
            return self._empty_result("consensus_round")

    async def benchmark_cache_operations(
        self, iterations: int = 1000, warmup: int = 100
    ) -> BenchmarkResult:
        """
        Benchmark cache key computation and lookup.

        Tests:
        - Hash function performance
        - Dict lookup speed
        - Eviction overhead
        """
        self.log(f"Benchmarking cache operations ({iterations} iterations, {warmup} warmup)...")

        try:
            from core.sovereign.runtime import SovereignRuntime, SovereignQuery

            runtime = SovereignRuntime()
            query = SovereignQuery(content="Test query for cache benchmark", max_depth=3)

            latencies = []
            cpu_samples = []
            memory_samples = []

            # Warmup
            for i in range(warmup):
                key = runtime._cache_key(query)
                _ = runtime._cache.get(key)

            # Actual benchmark
            for i in range(iterations):
                cpu_before = self.process.cpu_percent(interval=None)
                mem_before = self.process.memory_info().rss / 1024 / 1024

                start = time.perf_counter()
                key = runtime._cache_key(query)
                cached = runtime._cache.get(key)
                latency_ms = (time.perf_counter() - start) * 1000

                cpu_after = self.process.cpu_percent(interval=None)
                mem_after = self.process.memory_info().rss / 1024 / 1024

                latencies.append(latency_ms)
                cpu_samples.append((cpu_before + cpu_after) / 2)
                memory_samples.append(mem_after)

            return self._create_result(
                "cache_operations", len(latencies), latencies, cpu_samples, memory_samples
            )

        except ImportError as e:
            self.log(f"Skipping cache benchmark: {e}")
            return self._empty_result("cache_operations")

    async def benchmark_pci_gate_chain(
        self, iterations: int = 1000, warmup: int = 100
    ) -> BenchmarkResult:
        """
        Benchmark PCI gate chain validation.

        Tests:
        - Signature verification
        - Timestamp validation
        - Nonce checking
        - Ihsan/SNR thresholds
        """
        self.log(f"Benchmarking PCI gate chain ({iterations} iterations, {warmup} warmup)...")

        try:
            from core.pci.envelope import PCIEnvelope, PCISender, PCISignature, PCIMetadata
            from core.pci.gates import PCIGateKeeper
            from core.pci.crypto import generate_keypair, sign_message
            from datetime import datetime, timezone

            private_key, public_key = generate_keypair()
            gatekeeper = PCIGateKeeper()

            # Create test envelope
            sender = PCISender(node_id="test_node", public_key=public_key)
            metadata = PCIMetadata(ihsan_score=0.96, snr_score=0.97)

            latencies = []
            cpu_samples = []
            memory_samples = []

            # Warmup
            for i in range(warmup):
                envelope = PCIEnvelope(
                    sender=sender,
                    nonce=f"warmup_{i}",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    payload={"test": "data"},
                    metadata=metadata,
                    signature=PCISignature(algorithm="ed25519", value="dummy"),
                )
                digest = envelope.compute_digest()
                envelope.signature = PCISignature(
                    algorithm="ed25519", value=sign_message(digest, private_key)
                )
                gatekeeper.verify(envelope)

            # Actual benchmark
            for i in range(iterations):
                envelope = PCIEnvelope(
                    sender=sender,
                    nonce=f"bench_{i}",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    payload={"test": "data", "iteration": i},
                    metadata=metadata,
                    signature=PCISignature(algorithm="ed25519", value="dummy"),
                )
                digest = envelope.compute_digest()
                envelope.signature = PCISignature(
                    algorithm="ed25519", value=sign_message(digest, private_key)
                )

                cpu_before = self.process.cpu_percent(interval=None)
                mem_before = self.process.memory_info().rss / 1024 / 1024

                start = time.perf_counter()
                result = gatekeeper.verify(envelope)
                latency_ms = (time.perf_counter() - start) * 1000

                cpu_after = self.process.cpu_percent(interval=None)
                mem_after = self.process.memory_info().rss / 1024 / 1024

                if result.passed:
                    latencies.append(latency_ms)
                    cpu_samples.append((cpu_before + cpu_after) / 2)
                    memory_samples.append(mem_after)

            return self._create_result(
                "pci_gate_chain", len(latencies), latencies, cpu_samples, memory_samples
            )

        except ImportError as e:
            self.log(f"Skipping PCI gate benchmark: {e}")
            return self._empty_result("pci_gate_chain")

    def _create_result(
        self,
        name: str,
        iterations: int,
        latencies: List[float],
        cpu_samples: List[float],
        memory_samples: List[float],
    ) -> BenchmarkResult:
        """Create BenchmarkResult from samples."""
        if not latencies:
            return self._empty_result(name)

        sorted_latencies = sorted(latencies)

        return BenchmarkResult(
            name=name,
            iterations=iterations,
            mean_ms=statistics.mean(latencies),
            median_ms=statistics.median(latencies),
            p95_ms=sorted_latencies[int(len(sorted_latencies) * 0.95)]
            if len(sorted_latencies) > 20
            else sorted_latencies[-1],
            p99_ms=sorted_latencies[int(len(sorted_latencies) * 0.99)]
            if len(sorted_latencies) > 100
            else sorted_latencies[-1],
            min_ms=min(latencies),
            max_ms=max(latencies),
            stddev_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            qps=1000.0 / statistics.mean(latencies) if statistics.mean(latencies) > 0 else 0.0,
            cpu_percent=statistics.mean(cpu_samples) if cpu_samples else 0.0,
            memory_mb=statistics.mean(memory_samples) if memory_samples else 0.0,
        )

    def _empty_result(self, name: str) -> BenchmarkResult:
        """Create empty result for skipped benchmarks."""
        return BenchmarkResult(
            name=name,
            iterations=0,
            mean_ms=0.0,
            median_ms=0.0,
            p95_ms=0.0,
            p99_ms=0.0,
            min_ms=0.0,
            max_ms=0.0,
            stddev_ms=0.0,
            qps=0.0,
            cpu_percent=0.0,
            memory_mb=0.0,
        )

    def print_report(self, baseline: Optional[Dict[str, BenchmarkResult]] = None):
        """Print performance report."""
        print("\n" + "=" * 80)
        print("BIZRA PERFORMANCE BENCHMARK RESULTS")
        print("Standing on Giants: Knuth (1968), Amdahl (1967), Shannon (1948)")
        print("=" * 80)

        for result in self.results:
            if result.iterations == 0:
                continue

            print(f"\n{result.name.upper()}")
            print("-" * 80)
            print(f"  Iterations:     {result.iterations}")
            print(f"  Mean Latency:   {result.mean_ms:.2f}ms (±{result.stddev_ms:.2f}ms)")
            print(f"  Median Latency: {result.median_ms:.2f}ms")
            print(f"  P95 Latency:    {result.p95_ms:.2f}ms")
            print(f"  P99 Latency:    {result.p99_ms:.2f}ms")
            print(f"  Min/Max:        {result.min_ms:.2f}ms / {result.max_ms:.2f}ms")
            print(f"  Throughput:     {result.qps:.2f} QPS")
            print(f"  CPU Usage:      {result.cpu_percent:.1f}%")
            print(f"  Memory:         {result.memory_mb:.1f} MB")

            if baseline and result.name in baseline:
                baseline_result = baseline[result.name]
                improvement = result.improvement_over(baseline_result)
                symbol = "🟢" if improvement > 0 else "🔴" if improvement < -5 else "🟡"
                print(f"  {symbol} Improvement:    {improvement:+.1f}% over baseline")

        print("=" * 80)

    def export_json(self, output_path: Path):
        """Export results as JSON."""
        data = {
            "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": [r.to_dict() for r in self.results],
        }

        output_path.write_text(json.dumps(data, indent=2))
        self.log(f"Exported results to {output_path}")


async def main():
    """Main benchmark entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="BIZRA Performance Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--query", action="store_true", help="Benchmark query pipeline")
    parser.add_argument("--inference", action="store_true", help="Benchmark inference only")
    parser.add_argument("--consensus", action="store_true", help="Benchmark consensus")
    parser.add_argument("--cache", action="store_true", help="Benchmark cache operations")
    parser.add_argument("--pci", action="store_true", help="Benchmark PCI gate chain")
    parser.add_argument(
        "--baseline", type=str, help="Path to baseline JSON for comparison"
    )
    parser.add_argument("--output", type=str, help="Export results to JSON file")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages")

    args = parser.parse_args()

    # Default to all if no specific benchmarks selected
    if not any([args.query, args.inference, args.consensus, args.cache, args.pci]):
        args.all = True

    benchmark = PerformanceBenchmark(verbose=not args.quiet)

    # Load baseline if provided
    baseline = None
    if args.baseline:
        baseline_path = Path(args.baseline)
        if baseline_path.exists():
            baseline_data = json.loads(baseline_path.read_text())
            baseline = {r["name"]: BenchmarkResult(**r) for r in baseline_data["results"]}
            benchmark.log(f"Loaded baseline from {baseline_path}")

    # Run selected benchmarks
    if args.all or args.cache:
        result = await benchmark.benchmark_cache_operations()
        benchmark.results.append(result)

    if args.all or args.pci:
        result = await benchmark.benchmark_pci_gate_chain()
        benchmark.results.append(result)

    if args.all or args.consensus:
        result = await benchmark.benchmark_consensus_round()
        benchmark.results.append(result)

    if args.all or args.inference:
        result = await benchmark.benchmark_inference_only(iterations=20)
        benchmark.results.append(result)

    if args.all or args.query:
        result = await benchmark.benchmark_query_pipeline(iterations=10)
        benchmark.results.append(result)

    # Print report
    benchmark.print_report(baseline)

    # Export if requested
    if args.output:
        benchmark.export_json(Path(args.output))


if __name__ == "__main__":
    asyncio.run(main())
