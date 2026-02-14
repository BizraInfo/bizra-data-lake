"""
BIZRA Python Baseline Benchmarks

Establishes current performance baselines for Python implementations.
These will be compared against Rust targets after migration.

Run with:
    pytest benchmark_suite/e2e/test_python_baseline.py -v --benchmark-only
    pytest benchmark_suite/e2e/test_python_baseline.py -v --benchmark-compare

Target Metrics (from PERFORMANCE_ENGINEERING_PLAN.md):
- NTU observe: 8ms (Python) -> 100ns (Rust target)
- FATE validate: 1-5ms (Python) -> <10us (Rust target)
- SNR compute: 30-50ms (Python) -> <1ms (Rust target)
"""

import sys
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

import pytest
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def ntu_instance():
    """Create NTU instance for benchmarks."""
    from core.ntu import NTU, NTUConfig
    config = NTUConfig(window_size=64, ihsan_threshold=0.95)
    return NTU(config)


@pytest.fixture
def fate_gate():
    """Create FATE gate for benchmarks."""
    from core.elite.hooks import FATEGate, HookContext
    return FATEGate(ihsan_threshold=0.95, snr_threshold=0.85)


@pytest.fixture
def snr_calculator():
    """Create SNR calculator for benchmarks."""
    from core.iaas.snr_v2 import SNRCalculatorV2
    return SNRCalculatorV2()


@pytest.fixture
def sample_observations():
    """Generate sample observations for NTU benchmarks."""
    np.random.seed(42)
    return list(np.random.uniform(0.3, 0.9, size=1000))


@pytest.fixture
def sample_texts():
    """Generate sample texts for SNR benchmarks."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming industries.",
        "Data quality is essential for reliable inference.",
        "Shannon entropy measures information content.",
        "Byzantine fault tolerance ensures consensus.",
    ] * 20  # 100 texts


# ============================================================================
# NTU BENCHMARKS
# ============================================================================

class TestNTUBenchmarks:
    """Benchmarks for NeuroTemporal Unit (NTU)."""

    @pytest.mark.benchmark(group="ntu", min_rounds=100)
    def test_ntu_single_observation(self, benchmark, ntu_instance):
        """Benchmark single NTU observation.

        Target: 8ms (Python) -> 100ns (Rust)
        """
        def observe_single():
            ntu_instance.observe(0.75)

        result = benchmark(observe_single)

        # Record baseline
        stats = benchmark.stats
        print(f"\nNTU single observe: mean={stats.mean*1000:.3f}ms, "
              f"median={stats.median*1000:.3f}ms, stddev={stats.stddev*1000:.3f}ms")

    @pytest.mark.benchmark(group="ntu", min_rounds=50)
    def test_ntu_batch_100_observations(self, benchmark, ntu_instance, sample_observations):
        """Benchmark batch of 100 observations.

        Target: 800ms (Python) -> 10us (Rust)
        """
        batch = sample_observations[:100]

        def observe_batch():
            for obs in batch:
                ntu_instance.observe(obs)

        result = benchmark(observe_batch)

        stats = benchmark.stats
        print(f"\nNTU batch 100: mean={stats.mean*1000:.3f}ms, "
              f"per_obs={stats.mean*10:.3f}ms")

    @pytest.mark.benchmark(group="ntu", min_rounds=10)
    def test_ntu_pattern_detection_1000(self, benchmark, sample_observations):
        """Benchmark pattern detection on 1000 observations.

        Target: 8s (Python) -> 1ms (Rust)
        """
        from core.ntu import NTU, NTUConfig

        def detect_pattern():
            ntu = NTU(NTUConfig(window_size=64))
            detected, state = ntu.detect_pattern(sample_observations)
            return detected, state.belief

        result = benchmark(detect_pattern)

        stats = benchmark.stats
        print(f"\nNTU pattern detection 1000: mean={stats.mean*1000:.1f}ms, "
              f"throughput={1000/stats.mean:.0f} obs/sec")

    @pytest.mark.benchmark(group="ntu", min_rounds=50)
    def test_ntu_temporal_consistency(self, benchmark, ntu_instance, sample_observations):
        """Benchmark temporal consistency calculation.

        This is a key bottleneck identified in profiling.
        """
        # Pre-populate window
        for obs in sample_observations[:64]:
            ntu_instance.observe(obs)

        def compute_consistency():
            return ntu_instance._compute_temporal_consistency()

        result = benchmark(compute_consistency)

        stats = benchmark.stats
        print(f"\nNTU temporal consistency: mean={stats.mean*1000:.3f}ms")

    @pytest.mark.benchmark(group="ntu", min_rounds=20)
    def test_ntu_stationary_distribution(self, benchmark, ntu_instance):
        """Benchmark stationary distribution computation.

        Uses eigendecomposition - expensive but O(1) for 3x3 matrix.
        """
        def compute_stationary():
            return ntu_instance.stationary_distribution

        result = benchmark(compute_stationary)

        stats = benchmark.stats
        print(f"\nNTU stationary dist: mean={stats.mean*1000:.3f}ms")


# ============================================================================
# FATE GATE BENCHMARKS
# ============================================================================

class TestFATEBenchmarks:
    """Benchmarks for FATE Gate validation."""

    @pytest.mark.benchmark(group="fate", min_rounds=100)
    def test_fate_single_validation(self, benchmark, fate_gate):
        """Benchmark single FATE gate validation.

        Target: 1-5ms (Python) -> <10us (Rust)
        """
        from core.elite.hooks import HookContext

        context = HookContext(
            operation_name="test_operation",
            operation_type="function",
            input_data={"test": "data"},
            metadata={"description": "test operation"}
        )

        def validate_single():
            return fate_gate.validate(context, "test intent", 0.9)

        result = benchmark(validate_single)

        stats = benchmark.stats
        print(f"\nFATE validate: mean={stats.mean*1000:.3f}ms, "
              f"throughput={1/stats.mean:.0f} validations/sec")

    @pytest.mark.benchmark(group="fate", min_rounds=50)
    def test_fate_batch_100_validations(self, benchmark, fate_gate):
        """Benchmark batch of 100 validations."""
        from core.elite.hooks import HookContext

        contexts = [
            HookContext(
                operation_name=f"operation_{i}",
                operation_type="function",
                input_data={"index": i},
            )
            for i in range(100)
        ]

        def validate_batch():
            return [
                fate_gate.validate(ctx, "batch intent", 0.85 + 0.1 * (i % 2))
                for i, ctx in enumerate(contexts)
            ]

        result = benchmark(validate_batch)

        stats = benchmark.stats
        print(f"\nFATE batch 100: mean={stats.mean*1000:.1f}ms, "
              f"per_validation={stats.mean*10:.3f}ms")

    @pytest.mark.benchmark(group="fate", min_rounds=100)
    def test_fate_score_computation(self, benchmark):
        """Benchmark FATEScore computation."""
        from core.elite.hooks import FATEScore

        def compute_score():
            score = FATEScore(
                fidelity=0.85,
                accountability=0.90,
                transparency=0.88,
                ethics=0.92
            )
            return score.overall, score.passed, score.weakest_dimension

        result = benchmark(compute_score)

        stats = benchmark.stats
        print(f"\nFATE score compute: mean={stats.mean*1e6:.1f}us")


# ============================================================================
# SNR BENCHMARKS
# ============================================================================

class TestSNRBenchmarks:
    """Benchmarks for Signal-to-Noise Ratio calculation."""

    @pytest.mark.benchmark(group="snr", min_rounds=20)
    def test_snr_simple_calculation(self, benchmark, snr_calculator, sample_texts):
        """Benchmark simple SNR calculation (no embeddings).

        Target: 30-50ms (Python) -> <1ms (Rust)
        """
        query = "What is data quality and why does it matter?"
        texts = sample_texts[:10]

        def calculate_snr():
            return snr_calculator.calculate_simple(query, texts, iaas_score=0.9)

        result = benchmark(calculate_snr)

        stats = benchmark.stats
        print(f"\nSNR simple (10 texts): mean={stats.mean*1000:.1f}ms")

    @pytest.mark.benchmark(group="snr", min_rounds=10)
    def test_snr_full_calculation_with_embeddings(self, benchmark, snr_calculator, sample_texts):
        """Benchmark full SNR calculation with mock embeddings.

        This tests the full computation including semantic similarity.
        """
        np.random.seed(42)
        query = "What is data quality and why does it matter?"
        texts = sample_texts[:50]
        query_embedding = np.random.randn(384)
        text_embeddings = np.random.randn(50, 384)

        def calculate_snr():
            return snr_calculator.compute_snr(
                query=query,
                texts=texts,
                query_embedding=query_embedding,
                text_embeddings=text_embeddings,
                iaas_score=0.9,
            )

        result = benchmark(calculate_snr)

        stats = benchmark.stats
        print(f"\nSNR full (50 texts): mean={stats.mean*1000:.1f}ms")

    @pytest.mark.benchmark(group="snr", min_rounds=5)
    def test_snr_diversity_calculation(self, benchmark, snr_calculator, sample_texts):
        """Benchmark diversity calculation (O(n^2) bottleneck)."""
        np.random.seed(42)
        texts = sample_texts[:100]
        embeddings = np.random.randn(100, 384)

        def calculate_diversity():
            return snr_calculator._compute_diversity(texts, embeddings)

        result = benchmark(calculate_diversity)

        stats = benchmark.stats
        print(f"\nSNR diversity (100 texts, O(n^2)): mean={stats.mean*1000:.1f}ms")


# ============================================================================
# MEMORY BENCHMARKS
# ============================================================================

class TestMemoryBenchmarks:
    """Memory usage benchmarks."""

    def test_ntu_memory_usage(self):
        """Measure NTU instance memory usage.

        Target: 2KB (Python) -> 500 bytes (Rust)
        """
        from core.ntu import NTU, NTUConfig

        tracemalloc.start()

        # Create 100 NTU instances
        instances = [NTU(NTUConfig(window_size=64)) for _ in range(100)]

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        per_instance_kb = current / 100 / 1024
        print(f"\nNTU memory: {per_instance_kb:.2f} KB per instance "
              f"(target: 0.5 KB in Rust)")

        # Assert Python baseline is within expected range
        assert per_instance_kb < 10, f"NTU memory too high: {per_instance_kb} KB"

    def test_window_memory_at_scale(self):
        """Measure sliding window memory at scale.

        Problem: 8B nodes * 64-element windows = 512B observations
        """
        from core.ntu import NTU, NTUConfig

        tracemalloc.start()

        # Create NTU with large window
        ntu = NTU(NTUConfig(window_size=64))

        # Fill window
        for i in range(64):
            ntu.observe(i / 64)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        window_bytes = current
        per_observation = window_bytes / 64

        print(f"\nWindow memory: {window_bytes} bytes total, "
              f"{per_observation:.1f} bytes per observation "
              f"(target: 24 bytes in Rust)")


# ============================================================================
# THROUGHPUT BENCHMARKS
# ============================================================================

class TestThroughputBenchmarks:
    """Throughput-focused benchmarks."""

    def test_ntu_sustained_throughput(self):
        """Measure sustained NTU throughput over 1 second.

        Target: 100K ops/sec (Rust)
        """
        from core.ntu import NTU, NTUConfig

        ntu = NTU(NTUConfig(window_size=64))
        np.random.seed(42)
        observations = np.random.uniform(0.3, 0.9, size=10000)

        start = time.perf_counter()
        for obs in observations:
            ntu.observe(obs)
        elapsed = time.perf_counter() - start

        ops_per_sec = len(observations) / elapsed
        latency_us = elapsed / len(observations) * 1e6

        print(f"\nNTU throughput: {ops_per_sec:.0f} ops/sec, "
              f"latency: {latency_us:.1f} us/op "
              f"(target: 10M ops/sec in Rust)")

    def test_fate_sustained_throughput(self):
        """Measure sustained FATE validation throughput."""
        from core.elite.hooks import FATEGate, HookContext

        gate = FATEGate(ihsan_threshold=0.95, snr_threshold=0.85)
        contexts = [
            HookContext(
                operation_name=f"op_{i}",
                operation_type="function",
            )
            for i in range(1000)
        ]

        start = time.perf_counter()
        for ctx in contexts:
            gate.validate(ctx, "test", 0.9)
        elapsed = time.perf_counter() - start

        ops_per_sec = len(contexts) / elapsed
        latency_ms = elapsed / len(contexts) * 1000

        print(f"\nFATE throughput: {ops_per_sec:.0f} validations/sec, "
              f"latency: {latency_ms:.2f} ms/validation "
              f"(target: 100K ops/sec in Rust)")


# ============================================================================
# SCALE SIMULATION BENCHMARKS
# ============================================================================

class TestScaleSimulationBenchmarks:
    """Simulate planetary scale operations."""

    @pytest.mark.slow
    def test_simulate_million_observations(self):
        """Simulate 1M observations to estimate 8B scale.

        At 8B nodes with 1 obs/sec = 8B obs/sec required
        """
        from core.ntu import NTU, NTUConfig

        OBSERVATIONS = 100_000  # Reduced for CI
        TARGET_SCALE = 8_000_000_000

        ntu = NTU(NTUConfig(window_size=64))
        np.random.seed(42)
        observations = np.random.uniform(0.3, 0.9, size=OBSERVATIONS)

        start = time.perf_counter()
        for obs in observations:
            ntu.observe(obs)
        elapsed = time.perf_counter() - start

        ops_per_sec = OBSERVATIONS / elapsed
        time_for_8b = TARGET_SCALE / ops_per_sec / 3600  # hours

        print(f"\nScale simulation:")
        print(f"  Processed: {OBSERVATIONS:,} observations in {elapsed:.2f}s")
        print(f"  Throughput: {ops_per_sec:,.0f} obs/sec")
        print(f"  Time for 8B (Python): {time_for_8b:,.0f} hours")
        print(f"  Target (Rust): {TARGET_SCALE / 10_000_000:.0f} seconds "
              f"(at 10M ops/sec)")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only", "-s"])
