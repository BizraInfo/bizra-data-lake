"""
BIZRA Python Profiling Utilities

Tools for profiling Python implementations to identify bottlenecks
before Rust migration.

Usage:
    python benchmark_suite/profiling/python_profile.py

Output:
    - CPU profile (cProfile)
    - Memory profile (tracemalloc)
    - Line-by-line timing (if line_profiler available)
    - Flame graph data (for external visualization)
"""

import cProfile
import pstats
import tracemalloc
import time
import sys
import io
import json
from pathlib import Path
from functools import wraps
from typing import Callable, Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from contextlib import contextmanager

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# PROFILING DECORATORS
# ============================================================================

def profile_cpu(func: Callable) -> Callable:
    """Decorator to profile function CPU usage with cProfile."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            profiler.disable()

            # Print stats
            print(f"\n{'='*70}")
            print(f"CPU Profile: {func.__name__}")
            print(f"{'='*70}")

            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            stats.print_stats(20)

    return wrapper


def profile_memory(func: Callable) -> Callable:
    """Decorator to profile function memory usage with tracemalloc."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            print(f"\n{'='*70}")
            print(f"Memory Profile: {func.__name__}")
            print(f"{'='*70}")
            print(f"Current memory: {current / 1024:.2f} KB")
            print(f"Peak memory: {peak / 1024:.2f} KB")

    return wrapper


def profile_time(func: Callable) -> Callable:
    """Decorator to profile function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter_ns()

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed_ns = time.perf_counter_ns() - start

            print(f"\n{'='*70}")
            print(f"Time Profile: {func.__name__}")
            print(f"{'='*70}")
            print(f"Execution time: {elapsed_ns / 1e6:.3f} ms")
            print(f"Execution time: {elapsed_ns / 1e3:.3f} us")
            print(f"Execution time: {elapsed_ns} ns")

    return wrapper


def profile_all(func: Callable) -> Callable:
    """Decorator to profile CPU, memory, and time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Start tracemalloc
        tracemalloc.start()

        # Start cProfile
        profiler = cProfile.Profile()
        profiler.enable()

        # Start timer
        start = time.perf_counter_ns()

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Stop timer
            elapsed_ns = time.perf_counter_ns() - start

            # Stop profilers
            profiler.disable()
            current_mem, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Print combined report
            print(f"\n{'='*70}")
            print(f"FULL PROFILE: {func.__name__}")
            print(f"{'='*70}")
            print(f"\n--- TIMING ---")
            print(f"Total time: {elapsed_ns / 1e6:.3f} ms ({elapsed_ns} ns)")

            print(f"\n--- MEMORY ---")
            print(f"Current: {current_mem / 1024:.2f} KB")
            print(f"Peak: {peak_mem / 1024:.2f} KB")

            print(f"\n--- CPU (top 10 by cumulative time) ---")
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            stats.print_stats(10)

    return wrapper


# ============================================================================
# PROFILING CONTEXT MANAGERS
# ============================================================================

@contextmanager
def profile_section(name: str):
    """Context manager for profiling code sections."""
    tracemalloc.start()
    start = time.perf_counter_ns()

    try:
        yield
    finally:
        elapsed_ns = time.perf_counter_ns() - start
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"[{name}] time={elapsed_ns/1e6:.3f}ms, "
              f"mem_current={current_mem/1024:.1f}KB, "
              f"mem_peak={peak_mem/1024:.1f}KB")


# ============================================================================
# PROFILE REPORT GENERATION
# ============================================================================

@dataclass
class ProfileResult:
    """Structured profiling result."""
    function_name: str
    execution_time_ns: int
    memory_current_bytes: int
    memory_peak_bytes: int
    top_functions: List[Dict[str, Any]]


class ProfileReportGenerator:
    """Generate profiling reports for analysis."""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path(__file__).parent / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[ProfileResult] = []

    def profile_function(self, func: Callable, *args, **kwargs) -> ProfileResult:
        """Profile a function and return structured result."""
        tracemalloc.start()
        profiler = cProfile.Profile()
        profiler.enable()
        start = time.perf_counter_ns()

        try:
            func(*args, **kwargs)
        finally:
            elapsed_ns = time.perf_counter_ns() - start
            profiler.disable()
            current_mem, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        # Extract top functions
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats('cumulative')
        stats.print_stats(10)

        # Parse stats output (simplified)
        top_functions = []
        for line in stream.getvalue().split('\n'):
            if 'cumtime' in line or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 6:
                try:
                    top_functions.append({
                        "ncalls": parts[0],
                        "tottime": float(parts[1]),
                        "cumtime": float(parts[3]),
                        "function": parts[5] if len(parts) > 5 else "unknown"
                    })
                except (ValueError, IndexError):
                    continue

        result = ProfileResult(
            function_name=func.__name__,
            execution_time_ns=elapsed_ns,
            memory_current_bytes=current_mem,
            memory_peak_bytes=peak_mem,
            top_functions=top_functions[:10]
        )

        self.results.append(result)
        return result

    def save_report(self, filename: str = "profile_report.json"):
        """Save profiling results to JSON."""
        report_path = self.output_dir / filename
        with open(report_path, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        print(f"Report saved to: {report_path}")


# ============================================================================
# SPECIFIC PROFILERS
# ============================================================================

class NTUProfiler:
    """Profiler specific to NTU implementation."""

    def __init__(self):
        from core.ntu import NTU, NTUConfig
        self.NTU = NTU
        self.NTUConfig = NTUConfig

    @profile_all
    def profile_init(self, window_size: int = 64):
        """Profile NTU initialization."""
        config = self.NTUConfig(window_size=window_size)
        ntu = self.NTU(config)
        return ntu

    @profile_all
    def profile_observe_single(self, ntu=None):
        """Profile single observation."""
        if ntu is None:
            ntu = self.NTU(self.NTUConfig())
        ntu.observe(0.75)

    @profile_all
    def profile_observe_batch(self, n: int = 1000, ntu=None):
        """Profile batch observations."""
        if ntu is None:
            ntu = self.NTU(self.NTUConfig(window_size=64))

        np.random.seed(42)
        observations = np.random.uniform(0.3, 0.9, size=n)

        for obs in observations:
            ntu.observe(obs)

    @profile_all
    def profile_pattern_detection(self, n: int = 1000):
        """Profile pattern detection."""
        np.random.seed(42)
        observations = list(np.random.uniform(0.3, 0.9, size=n))

        ntu = self.NTU(self.NTUConfig(window_size=64))
        detected, state = ntu.detect_pattern(observations)
        return detected, state.belief

    def profile_breakdown(self, n: int = 1000):
        """Profile NTU with detailed breakdown of each operation."""
        print(f"\n{'='*70}")
        print(f"NTU DETAILED BREAKDOWN ({n} observations)")
        print(f"{'='*70}")

        ntu = self.NTU(self.NTUConfig(window_size=64))
        np.random.seed(42)
        observations = np.random.uniform(0.3, 0.9, size=n)

        # Pre-populate window
        for obs in observations[:64]:
            ntu.observe(obs)

        # Profile individual operations
        with profile_section("temporal_consistency"):
            for _ in range(100):
                ntu._compute_temporal_consistency()

        with profile_section("neural_prior (100 lookups)"):
            from core.ntu.ntu import Observation
            obs = Observation(value=0.75)
            for _ in range(100):
                ntu._compute_neural_prior(obs)

        with profile_section("state_update (100 updates)"):
            for _ in range(100):
                ntu._update_state(0.8, np.array([0.7, 0.2, 0.1]))

        with profile_section("stationary_distribution"):
            _ = ntu.stationary_distribution


class FATEProfiler:
    """Profiler specific to FATE Gate implementation."""

    def __init__(self):
        from core.elite.hooks import FATEGate, HookContext
        self.FATEGate = FATEGate
        self.HookContext = HookContext

    @profile_all
    def profile_validate_single(self):
        """Profile single FATE validation."""
        gate = self.FATEGate(ihsan_threshold=0.95, snr_threshold=0.85)
        context = self.HookContext(
            operation_name="test_op",
            operation_type="function",
            input_data={"test": "data"},
            metadata={"description": "test"}
        )
        return gate.validate(context, "test intent", 0.9)

    @profile_all
    def profile_validate_batch(self, n: int = 1000):
        """Profile batch FATE validations."""
        gate = self.FATEGate(ihsan_threshold=0.95, snr_threshold=0.85)
        contexts = [
            self.HookContext(
                operation_name=f"op_{i}",
                operation_type="function",
            )
            for i in range(n)
        ]

        for ctx in contexts:
            gate.validate(ctx, "batch intent", 0.9)


class SNRProfiler:
    """Profiler specific to SNR calculation."""

    def __init__(self):
        from core.iaas.snr_v2 import SNRCalculatorV2
        self.SNRCalculatorV2 = SNRCalculatorV2

    @profile_all
    def profile_simple_snr(self, n_texts: int = 100):
        """Profile simple SNR calculation."""
        calc = self.SNRCalculatorV2()
        query = "What is data quality and why does it matter?"
        texts = [f"Sample text {i} about data quality metrics." for i in range(n_texts)]

        return calc.calculate_simple(query, texts, iaas_score=0.9)

    @profile_all
    def profile_full_snr(self, n_texts: int = 100):
        """Profile full SNR calculation with embeddings."""
        np.random.seed(42)
        calc = self.SNRCalculatorV2()
        query = "What is data quality and why does it matter?"
        texts = [f"Sample text {i} about data quality metrics." for i in range(n_texts)]
        query_embedding = np.random.randn(384)
        text_embeddings = np.random.randn(n_texts, 384)

        return calc.compute_snr(
            query=query,
            texts=texts,
            query_embedding=query_embedding,
            text_embeddings=text_embeddings,
            iaas_score=0.9,
        )

    def profile_diversity_bottleneck(self, n_texts: int = 100):
        """Profile the O(n^2) diversity bottleneck."""
        print(f"\n{'='*70}")
        print(f"SNR DIVERSITY BOTTLENECK ({n_texts} texts)")
        print(f"{'='*70}")

        np.random.seed(42)
        calc = self.SNRCalculatorV2()
        texts = [f"Sample text {i}" for i in range(n_texts)]
        embeddings = np.random.randn(n_texts, 384)

        with profile_section(f"diversity O(n^2) with n={n_texts}"):
            calc._compute_diversity(texts, embeddings)

        # Compare with larger n
        n_large = n_texts * 2
        texts_large = texts * 2
        embeddings_large = np.random.randn(n_large, 384)

        with profile_section(f"diversity O(n^2) with n={n_large}"):
            calc._compute_diversity(texts_large, embeddings_large)

        print(f"\nExpected ratio: {(n_large/n_texts)**2:.1f}x (O(n^2) scaling)")


# ============================================================================
# MAIN
# ============================================================================

def run_all_profiles():
    """Run all profilers and generate comprehensive report."""
    print("=" * 70)
    print("BIZRA PYTHON PROFILING SUITE")
    print("=" * 70)

    # NTU Profiling
    print("\n\n### NTU PROFILING ###")
    ntu_profiler = NTUProfiler()
    ntu_profiler.profile_init()
    ntu_profiler.profile_observe_single()
    ntu_profiler.profile_observe_batch(n=1000)
    ntu_profiler.profile_pattern_detection(n=1000)
    ntu_profiler.profile_breakdown(n=1000)

    # FATE Profiling
    print("\n\n### FATE GATE PROFILING ###")
    fate_profiler = FATEProfiler()
    fate_profiler.profile_validate_single()
    fate_profiler.profile_validate_batch(n=1000)

    # SNR Profiling
    print("\n\n### SNR PROFILING ###")
    snr_profiler = SNRProfiler()
    snr_profiler.profile_simple_snr(n_texts=100)
    snr_profiler.profile_full_snr(n_texts=100)
    snr_profiler.profile_diversity_bottleneck(n_texts=50)

    print("\n" + "=" * 70)
    print("PROFILING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_all_profiles()
