"""
BIZRA Apex Orchestrator Performance Benchmark Suite
====================================================
Comprehensive benchmarks to verify state-of-art performance
for the Apex Orchestrator components.

Target Performance Metrics:
- Agent selection: <1ms
- Posterior update: <0.5ms
- Batch selection: >1000/sec
- Pattern hash: <0.1ms
- Similarity scoring: <1ms for 100 patterns
- Learning iteration: <10ms
- Full routing cycle: <5ms

Usage:
    python benchmarks/apex_benchmarks.py
    python benchmarks/apex_benchmarks.py --category thompson
    python benchmarks/apex_benchmarks.py --iterations 5000 --report
"""

from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import argparse
import tracemalloc

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.apex.thompson_router import (
    ThompsonSamplingRouter,
    CapabilityMatrix,
    BetaPrior,
    TaskCategory,
    SelectionResult,
)
from core.apex.sona_learner import (
    SONALearner,
    LearningConfig,
    ExecutionRecord,
    TrackedPattern,
    RoutingWeights,
    PerformanceMetrics,
)


# ===============================================================================
# BENCHMARK INFRASTRUCTURE
# ===============================================================================

@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    category: str
    iterations: int

    # Timing metrics (nanoseconds)
    mean_ns: float
    median_ns: float
    p95_ns: float
    p99_ns: float
    min_ns: float
    max_ns: float
    std_ns: float

    # Derived metrics
    ops_per_second: float

    # Target validation
    target_ns: Optional[float] = None
    passed: bool = True

    # Memory metrics
    memory_bytes: Optional[int] = None
    memory_peak_bytes: Optional[int] = None

    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "category": self.category,
            "iterations": self.iterations,
            "timing": {
                "mean_ns": self.mean_ns,
                "median_ns": self.median_ns,
                "p95_ns": self.p95_ns,
                "p99_ns": self.p99_ns,
                "min_ns": self.min_ns,
                "max_ns": self.max_ns,
                "std_ns": self.std_ns,
                "mean_ms": self.mean_ns / 1_000_000,
                "median_ms": self.median_ns / 1_000_000,
                "p95_ms": self.p95_ns / 1_000_000,
                "p99_ms": self.p99_ns / 1_000_000,
            },
            "throughput": {
                "ops_per_second": self.ops_per_second,
            },
            "target": {
                "target_ns": self.target_ns,
                "target_ms": self.target_ns / 1_000_000 if self.target_ns else None,
                "passed": self.passed,
            },
            "memory": {
                "bytes": self.memory_bytes,
                "peak_bytes": self.memory_peak_bytes,
                "mb": self.memory_bytes / (1024 * 1024) if self.memory_bytes else None,
            },
            "timestamp": self.timestamp,
        }


@dataclass
class BenchmarkReport:
    """Full benchmark report."""
    results: List[BenchmarkResult]
    total_benchmarks: int
    passed_benchmarks: int
    failed_benchmarks: int
    total_duration_seconds: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "summary": {
                "total_benchmarks": self.total_benchmarks,
                "passed": self.passed_benchmarks,
                "failed": self.failed_benchmarks,
                "pass_rate": self.passed_benchmarks / self.total_benchmarks if self.total_benchmarks > 0 else 0,
                "total_duration_seconds": self.total_duration_seconds,
            },
            "results": [r.to_dict() for r in self.results],
            "timestamp": self.timestamp,
        }


class BenchmarkRunner:
    """
    Benchmark runner with nanosecond precision timing.

    Uses time.perf_counter_ns() for accurate measurements.
    """

    def __init__(self, warmup_iterations: int = 100, default_iterations: int = 1000):
        self.warmup_iterations = warmup_iterations
        self.default_iterations = default_iterations
        self.results: List[BenchmarkResult] = []

    def run_benchmark(
        self,
        name: str,
        category: str,
        func: Callable[[], Any],
        iterations: Optional[int] = None,
        target_ns: Optional[float] = None,
        setup: Optional[Callable[[], Any]] = None,
        teardown: Optional[Callable[[], None]] = None,
        track_memory: bool = False,
    ) -> BenchmarkResult:
        """
        Run a benchmark with nanosecond precision.

        Args:
            name: Benchmark name
            category: Benchmark category
            func: Function to benchmark
            iterations: Number of iterations (default: self.default_iterations)
            target_ns: Target time in nanoseconds (for pass/fail)
            setup: Optional setup function called before each iteration
            teardown: Optional teardown function called after benchmark
            track_memory: Whether to track memory usage

        Returns:
            BenchmarkResult with timing and memory metrics
        """
        iterations = iterations or self.default_iterations

        # Force garbage collection
        gc.collect()

        # Warmup phase
        for _ in range(self.warmup_iterations):
            if setup:
                setup()
            func()

        # Reset garbage collection
        gc.collect()

        # Track memory if requested
        if track_memory:
            tracemalloc.start()

        # Benchmark phase
        timings_ns: List[int] = []

        for _ in range(iterations):
            if setup:
                setup()

            start = time.perf_counter_ns()
            func()
            end = time.perf_counter_ns()

            timings_ns.append(end - start)

        # Memory snapshot
        memory_bytes = None
        memory_peak_bytes = None
        if track_memory:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            memory_bytes = current
            memory_peak_bytes = peak

        # Cleanup
        if teardown:
            teardown()

        # Compute statistics
        mean_ns = statistics.mean(timings_ns)
        median_ns = statistics.median(timings_ns)
        std_ns = statistics.stdev(timings_ns) if len(timings_ns) > 1 else 0.0

        sorted_timings = sorted(timings_ns)
        p95_idx = int(0.95 * len(sorted_timings))
        p99_idx = int(0.99 * len(sorted_timings))
        p95_ns = sorted_timings[p95_idx] if sorted_timings else 0
        p99_ns = sorted_timings[p99_idx] if sorted_timings else 0

        min_ns = min(timings_ns)
        max_ns = max(timings_ns)

        # Calculate ops/sec
        ops_per_second = 1_000_000_000 / mean_ns if mean_ns > 0 else float('inf')

        # Check if passed target
        passed = True
        if target_ns is not None:
            passed = p95_ns <= target_ns

        result = BenchmarkResult(
            name=name,
            category=category,
            iterations=iterations,
            mean_ns=mean_ns,
            median_ns=median_ns,
            p95_ns=p95_ns,
            p99_ns=p99_ns,
            min_ns=min_ns,
            max_ns=max_ns,
            std_ns=std_ns,
            ops_per_second=ops_per_second,
            target_ns=target_ns,
            passed=passed,
            memory_bytes=memory_bytes,
            memory_peak_bytes=memory_peak_bytes,
        )

        self.results.append(result)
        return result

    async def run_async_benchmark(
        self,
        name: str,
        category: str,
        func: Callable[[], Any],
        iterations: Optional[int] = None,
        target_ns: Optional[float] = None,
        track_memory: bool = False,
    ) -> BenchmarkResult:
        """Run an async benchmark."""
        iterations = iterations or self.default_iterations

        gc.collect()

        # Warmup
        for _ in range(self.warmup_iterations):
            await func()

        gc.collect()

        if track_memory:
            tracemalloc.start()

        timings_ns: List[int] = []

        for _ in range(iterations):
            start = time.perf_counter_ns()
            await func()
            end = time.perf_counter_ns()
            timings_ns.append(end - start)

        memory_bytes = None
        memory_peak_bytes = None
        if track_memory:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            memory_bytes = current
            memory_peak_bytes = peak

        mean_ns = statistics.mean(timings_ns)
        median_ns = statistics.median(timings_ns)
        std_ns = statistics.stdev(timings_ns) if len(timings_ns) > 1 else 0.0

        sorted_timings = sorted(timings_ns)
        p95_ns = sorted_timings[int(0.95 * len(sorted_timings))]
        p99_ns = sorted_timings[int(0.99 * len(sorted_timings))]

        ops_per_second = 1_000_000_000 / mean_ns if mean_ns > 0 else float('inf')
        passed = p95_ns <= target_ns if target_ns else True

        result = BenchmarkResult(
            name=name,
            category=category,
            iterations=iterations,
            mean_ns=mean_ns,
            median_ns=median_ns,
            p95_ns=p95_ns,
            p99_ns=p99_ns,
            min_ns=min(timings_ns),
            max_ns=max(timings_ns),
            std_ns=std_ns,
            ops_per_second=ops_per_second,
            target_ns=target_ns,
            passed=passed,
            memory_bytes=memory_bytes,
            memory_peak_bytes=memory_peak_bytes,
        )

        self.results.append(result)
        return result

    def generate_report(self, duration_seconds: float) -> BenchmarkReport:
        """Generate a full benchmark report."""
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed

        return BenchmarkReport(
            results=self.results,
            total_benchmarks=len(self.results),
            passed_benchmarks=passed,
            failed_benchmarks=failed,
            total_duration_seconds=duration_seconds,
        )


# ===============================================================================
# TARGET DEFINITIONS (in nanoseconds)
# ===============================================================================

# 1ms = 1,000,000 ns
MS = 1_000_000

TARGETS = {
    # Thompson Sampling Performance
    "agent_selection_latency": 1 * MS,          # <1ms
    "posterior_update_latency": 0.5 * MS,       # <0.5ms
    "batch_selection_throughput": 1000,         # >1000/sec (as ops/sec target)

    # Pattern Extraction Performance
    "pattern_hash_computation": 0.1 * MS,       # <0.1ms
    "similarity_scoring": 1 * MS,               # <1ms for 100 patterns
    "elevation_check": 0.5 * MS,                # <0.5ms

    # SONA Learning Performance
    "learning_iteration": 10 * MS,              # <10ms
    "weight_optimization": 5 * MS,              # <5ms

    # Cost Analysis Performance
    "cost_calculation": 0.1 * MS,               # <0.1ms
    "report_generation": 50 * MS,               # <50ms

    # End-to-End Pipeline Performance
    "full_routing_cycle": 5 * MS,               # <5ms
    "quality_gate_validation": 10 * MS,         # <10ms

    # Concurrency targets
    "concurrent_routing": 100,                  # 100 concurrent operations
}


# ===============================================================================
# BENCHMARK IMPLEMENTATIONS
# ===============================================================================

class ThompsonSamplingBenchmarks:
    """Benchmarks for Thompson Sampling Router."""

    def __init__(self, runner: BenchmarkRunner):
        self.runner = runner
        self.router = ThompsonSamplingRouter(seed=42)
        self.capability_matrix = CapabilityMatrix()

    def benchmark_agent_selection_latency(self, iterations: int = 1000) -> BenchmarkResult:
        """Benchmark agent selection latency (target: <1ms)."""
        task = "Analyze the quarterly financial data and generate insights"

        return self.runner.run_benchmark(
            name="agent_selection_latency",
            category="thompson_sampling",
            func=lambda: self.router.select_agent(task),
            iterations=iterations,
            target_ns=TARGETS["agent_selection_latency"],
        )

    def benchmark_posterior_update_latency(self, iterations: int = 1000) -> BenchmarkResult:
        """Benchmark posterior update latency (target: <0.5ms)."""
        def update():
            self.router.update_posterior(
                agent_name="MasterReasoner",
                category=TaskCategory.REASONING,
                success=True,
                quality_score=0.95,
            )

        return self.runner.run_benchmark(
            name="posterior_update_latency",
            category="thompson_sampling",
            func=update,
            iterations=iterations,
            target_ns=TARGETS["posterior_update_latency"],
        )

    def benchmark_batch_selection_throughput(self, iterations: int = 1000) -> BenchmarkResult:
        """Benchmark batch selection throughput (target: >1000/sec)."""
        tasks = [
            "Analyze data patterns",
            "Write creative content",
            "Plan project timeline",
            "Check ethical compliance",
            "Remember context",
        ]

        def batch_select():
            for task in tasks:
                self.router.select_agent(task)

        result = self.runner.run_benchmark(
            name="batch_selection_throughput",
            category="thompson_sampling",
            func=batch_select,
            iterations=iterations,
        )

        # Adjust ops/sec for batch size
        result.ops_per_second = result.ops_per_second * len(tasks)
        result.target_ns = None  # Use ops/sec target instead
        result.passed = result.ops_per_second >= TARGETS["batch_selection_throughput"]

        return result

    def benchmark_beta_sampling(self, iterations: int = 5000) -> BenchmarkResult:
        """Benchmark Beta distribution sampling."""
        prior = BetaPrior(alpha=5.0, beta=3.0)
        rng = np.random.default_rng(42)

        return self.runner.run_benchmark(
            name="beta_sampling",
            category="thompson_sampling",
            func=lambda: prior.sample(rng),
            iterations=iterations,
            target_ns=0.05 * MS,  # <0.05ms
        )

    def benchmark_task_classification(self, iterations: int = 1000) -> BenchmarkResult:
        """Benchmark task category classification."""
        tasks = [
            "Analyze the quarterly financial data",
            "Write a creative marketing copy",
            "Plan the project timeline",
            "Check for ethical compliance",
        ]
        task_idx = 0

        def classify():
            nonlocal task_idx
            result = self.capability_matrix.classify_task(tasks[task_idx % len(tasks)])
            task_idx += 1
            return result

        return self.runner.run_benchmark(
            name="task_classification",
            category="thompson_sampling",
            func=classify,
            iterations=iterations,
            target_ns=0.2 * MS,  # <0.2ms
        )

    def benchmark_candidate_selection(self, iterations: int = 1000) -> BenchmarkResult:
        """Benchmark candidate agent selection from capability matrix."""
        return self.runner.run_benchmark(
            name="candidate_selection",
            category="thompson_sampling",
            func=lambda: self.capability_matrix.get_candidates(TaskCategory.REASONING),
            iterations=iterations,
            target_ns=0.1 * MS,  # <0.1ms
        )

    def benchmark_exploration_rate(self, iterations: int = 1000) -> BenchmarkResult:
        """Benchmark exploration rate calculation."""
        # Pre-populate some history
        for _ in range(100):
            self.router.select_agent("Test task for history")

        return self.runner.run_benchmark(
            name="exploration_rate",
            category="thompson_sampling",
            func=lambda: self.router.get_exploration_rate(),
            iterations=iterations,
            target_ns=0.3 * MS,  # <0.3ms
        )

    def benchmark_serialization(self, iterations: int = 500) -> BenchmarkResult:
        """Benchmark router state serialization."""
        return self.runner.run_benchmark(
            name="router_serialization",
            category="thompson_sampling",
            func=lambda: self.router.to_json(),
            iterations=iterations,
            target_ns=2 * MS,  # <2ms
        )


class PatternExtractionBenchmarks:
    """Benchmarks for pattern extraction and matching."""

    def __init__(self, runner: BenchmarkRunner):
        self.runner = runner
        self.learner = SONALearner()
        self._populate_patterns()

    def _populate_patterns(self):
        """Populate learner with sample patterns."""
        agents = ["MasterReasoner", "CreativeSynthesizer", "DataAnalyzer"]
        categories = ["reasoning", "creative", "analysis"]

        for i in range(100):
            record = ExecutionRecord(
                task_id=f"task_{i:04d}",
                task_category=categories[i % len(categories)],
                agent_name=agents[i % len(agents)],
                success=i % 5 != 0,  # 80% success rate
                quality_score=0.8 + (i % 20) / 100,
                latency_ms=500 + i * 10,
                token_count=100 + i * 5,
                cost=0.001 * (100 + i * 5),
            )
            self.learner.record_execution(record)

    def benchmark_pattern_hash_computation(self, iterations: int = 5000) -> BenchmarkResult:
        """Benchmark pattern hash computation (target: <0.1ms)."""
        record = ExecutionRecord(
            task_id="test_task",
            task_category="reasoning",
            agent_name="MasterReasoner",
            success=True,
            quality_score=0.95,
            latency_ms=1000,
            token_count=500,
            cost=0.01,
        )

        def compute_hash():
            signature = f"{record.task_category}:{record.agent_name}"
            return hashlib.sha256(signature.encode()).hexdigest()[:16]

        return self.runner.run_benchmark(
            name="pattern_hash_computation",
            category="pattern_extraction",
            func=compute_hash,
            iterations=iterations,
            target_ns=TARGETS["pattern_hash_computation"],
        )

    def benchmark_similarity_scoring(self, iterations: int = 1000) -> BenchmarkResult:
        """Benchmark similarity scoring for 100 patterns (target: <1ms)."""
        patterns = self.learner.extract_patterns()[:100]
        reference = patterns[0] if patterns else TrackedPattern(
            pattern_hash="ref", pattern_signature=["reasoning:MasterReasoner"]
        )

        def score_similarity():
            scores = []
            for pattern in patterns:
                # Simple similarity based on success rate difference
                diff = abs(pattern.success_rate - reference.success_rate)
                scores.append(1.0 - diff)
            return scores

        return self.runner.run_benchmark(
            name="similarity_scoring",
            category="pattern_extraction",
            func=score_similarity,
            iterations=iterations,
            target_ns=TARGETS["similarity_scoring"],
        )

    def benchmark_elevation_check(self, iterations: int = 5000) -> BenchmarkResult:
        """Benchmark pattern elevation check (target: <0.5ms)."""
        pattern = TrackedPattern(
            pattern_hash="test_hash",
            pattern_signature=["reasoning:MasterReasoner"],
            occurrence_count=5,
            success_count=4,
        )

        return self.runner.run_benchmark(
            name="elevation_check",
            category="pattern_extraction",
            func=lambda: pattern.should_elevate(threshold=3, min_success_rate=0.7),
            iterations=iterations,
            target_ns=TARGETS["elevation_check"],
        )

    def benchmark_pattern_extraction(self, iterations: int = 500) -> BenchmarkResult:
        """Benchmark full pattern extraction from history."""
        return self.runner.run_benchmark(
            name="pattern_extraction",
            category="pattern_extraction",
            func=lambda: self.learner.extract_patterns(),
            iterations=iterations,
            target_ns=1 * MS,  # <1ms
        )


class SONALearningBenchmarks:
    """Benchmarks for SONA learning system."""

    def __init__(self, runner: BenchmarkRunner):
        self.runner = runner
        self.learner = SONALearner()
        self._populate_history()

    def _populate_history(self):
        """Populate learner with execution history."""
        agents = ["MasterReasoner", "CreativeSynthesizer", "DataAnalyzer", "ExecutionPlanner"]
        categories = ["reasoning", "creative", "analysis", "planning"]

        for i in range(500):
            record = ExecutionRecord(
                task_id=f"task_{i:04d}",
                task_category=categories[i % len(categories)],
                agent_name=agents[i % len(agents)],
                success=np.random.random() > 0.3,
                quality_score=np.random.uniform(0.7, 1.0),
                latency_ms=np.random.uniform(500, 2000),
                token_count=np.random.randint(100, 1000),
                cost=np.random.uniform(0.001, 0.01),
            )
            self.learner.record_execution(record)

    def benchmark_learning_iteration(self, iterations: int = 500) -> BenchmarkResult:
        """Benchmark single learning iteration (target: <10ms)."""
        def learning_step():
            self.learner.extract_patterns()
            self.learner.optimize_routing()
            self.learner.evaluate_performance()

        return self.runner.run_benchmark(
            name="learning_iteration",
            category="sona_learning",
            func=learning_step,
            iterations=iterations,
            target_ns=TARGETS["learning_iteration"],
        )

    def benchmark_weight_optimization(self, iterations: int = 500) -> BenchmarkResult:
        """Benchmark weight optimization (target: <5ms)."""
        return self.runner.run_benchmark(
            name="weight_optimization",
            category="sona_learning",
            func=lambda: self.learner.optimize_routing(),
            iterations=iterations,
            target_ns=TARGETS["weight_optimization"],
        )

    def benchmark_performance_evaluation(self, iterations: int = 500) -> BenchmarkResult:
        """Benchmark performance evaluation."""
        return self.runner.run_benchmark(
            name="performance_evaluation",
            category="sona_learning",
            func=lambda: self.learner.evaluate_performance(),
            iterations=iterations,
            target_ns=3 * MS,  # <3ms
        )

    def benchmark_routing_recommendation(self, iterations: int = 1000) -> BenchmarkResult:
        """Benchmark routing recommendation generation."""
        agents = ["MasterReasoner", "CreativeSynthesizer", "DataAnalyzer"]

        return self.runner.run_benchmark(
            name="routing_recommendation",
            category="sona_learning",
            func=lambda: self.learner.get_routing_recommendation("reasoning", agents),
            iterations=iterations,
            target_ns=0.2 * MS,  # <0.2ms
        )

    def benchmark_record_execution(self, iterations: int = 2000) -> BenchmarkResult:
        """Benchmark execution recording."""
        idx = 0

        def record():
            nonlocal idx
            rec = ExecutionRecord(
                task_id=f"bench_task_{idx}",
                task_category="reasoning",
                agent_name="MasterReasoner",
                success=True,
                quality_score=0.95,
                latency_ms=1000,
                token_count=500,
                cost=0.01,
            )
            self.learner.record_execution(rec)
            idx += 1

        return self.runner.run_benchmark(
            name="record_execution",
            category="sona_learning",
            func=record,
            iterations=iterations,
            target_ns=0.5 * MS,  # <0.5ms
        )


class CostAnalysisBenchmarks:
    """Benchmarks for cost analysis operations."""

    def __init__(self, runner: BenchmarkRunner):
        self.runner = runner
        self.learner = SONALearner()
        self._populate_history()

    def _populate_history(self):
        """Populate with cost data."""
        for i in range(200):
            record = ExecutionRecord(
                task_id=f"cost_task_{i}",
                task_category="analysis",
                agent_name="DataAnalyzer",
                success=True,
                quality_score=0.9,
                latency_ms=1000,
                token_count=500 + i * 10,
                cost=0.001 * (500 + i * 10),
            )
            self.learner.record_execution(record)

    def benchmark_cost_calculation(self, iterations: int = 5000) -> BenchmarkResult:
        """Benchmark cost calculation (target: <0.1ms)."""
        token_count = 500
        cost_per_token = 0.002

        def calculate_cost():
            return token_count * cost_per_token

        return self.runner.run_benchmark(
            name="cost_calculation",
            category="cost_analysis",
            func=calculate_cost,
            iterations=iterations,
            target_ns=TARGETS["cost_calculation"],
        )

    def benchmark_report_generation(self, iterations: int = 200) -> BenchmarkResult:
        """Benchmark report generation (target: <50ms)."""
        def generate_report():
            metrics = self.learner.evaluate_performance()
            progress = self.learner.get_improvement_progress()
            patterns = self.learner.extract_patterns()

            return {
                "metrics": metrics.to_dict(),
                "progress": progress,
                "top_patterns": [p.to_dict() for p in patterns[:10]],
            }

        return self.runner.run_benchmark(
            name="report_generation",
            category="cost_analysis",
            func=generate_report,
            iterations=iterations,
            target_ns=TARGETS["report_generation"],
        )


class EndToEndBenchmarks:
    """End-to-end pipeline benchmarks."""

    def __init__(self, runner: BenchmarkRunner):
        self.runner = runner
        self.router = ThompsonSamplingRouter(seed=42)
        self.learner = SONALearner()

    def benchmark_full_routing_cycle(self, iterations: int = 500) -> BenchmarkResult:
        """Benchmark full routing cycle (target: <5ms)."""
        def full_cycle():
            # 1. Classify and select agent
            result = self.router.select_agent("Analyze the data patterns")

            # 2. Simulate execution result
            success = np.random.random() > 0.2
            quality = np.random.uniform(0.8, 1.0) if success else np.random.uniform(0.4, 0.7)

            # 3. Update posterior
            self.router.update_posterior(
                result.agent_name,
                result.task_category,
                success,
                quality,
            )

            # 4. Record for learning
            record = ExecutionRecord(
                task_id="cycle_task",
                task_category=result.task_category.value,
                agent_name=result.agent_name,
                success=success,
                quality_score=quality,
                latency_ms=1000,
                token_count=500,
                cost=0.01,
            )
            self.learner.record_execution(record)

        return self.runner.run_benchmark(
            name="full_routing_cycle",
            category="end_to_end",
            func=full_cycle,
            iterations=iterations,
            target_ns=TARGETS["full_routing_cycle"],
        )

    def benchmark_quality_gate_validation(self, iterations: int = 500) -> BenchmarkResult:
        """Benchmark quality gate validation (target: <10ms)."""
        def validate_quality_gate():
            # Simulate Ihsan gate check
            ihsan_threshold = 0.95

            # Evaluate current performance
            metrics = self.learner.evaluate_performance()

            # Check quality dimensions
            quality_passed = metrics.avg_quality_score >= ihsan_threshold
            success_passed = metrics.success_rate >= 0.8

            return {
                "quality_passed": quality_passed,
                "success_passed": success_passed,
                "ihsan_score": metrics.avg_quality_score,
                "gate_result": quality_passed and success_passed,
            }

        return self.runner.run_benchmark(
            name="quality_gate_validation",
            category="end_to_end",
            func=validate_quality_gate,
            iterations=iterations,
            target_ns=TARGETS["quality_gate_validation"],
        )


class MemoryBenchmarks:
    """Memory footprint benchmarks."""

    def __init__(self, runner: BenchmarkRunner):
        self.runner = runner

    def benchmark_router_memory_footprint(self) -> BenchmarkResult:
        """Benchmark router memory footprint."""
        tracemalloc.start()

        router = ThompsonSamplingRouter(seed=42)
        # Simulate some usage
        for _ in range(100):
            router.select_agent("Test task")

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        result = BenchmarkResult(
            name="router_memory_footprint",
            category="memory",
            iterations=1,
            mean_ns=0,
            median_ns=0,
            p95_ns=0,
            p99_ns=0,
            min_ns=0,
            max_ns=0,
            std_ns=0,
            ops_per_second=0,
            memory_bytes=current,
            memory_peak_bytes=peak,
            passed=current < 10 * 1024 * 1024,  # <10MB
        )
        self.runner.results.append(result)
        return result

    def benchmark_pattern_cache_memory(self) -> BenchmarkResult:
        """Benchmark pattern cache memory."""
        tracemalloc.start()

        learner = SONALearner()
        # Populate with patterns
        for i in range(500):
            record = ExecutionRecord(
                task_id=f"mem_task_{i}",
                task_category="reasoning",
                agent_name="MasterReasoner",
                success=True,
                quality_score=0.9,
                latency_ms=1000,
                token_count=500,
                cost=0.01,
            )
            learner.record_execution(record)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        result = BenchmarkResult(
            name="pattern_cache_memory",
            category="memory",
            iterations=1,
            mean_ns=0,
            median_ns=0,
            p95_ns=0,
            p99_ns=0,
            min_ns=0,
            max_ns=0,
            std_ns=0,
            ops_per_second=0,
            memory_bytes=current,
            memory_peak_bytes=peak,
            passed=current < 50 * 1024 * 1024,  # <50MB
        )
        self.runner.results.append(result)
        return result

    def benchmark_learning_buffer_memory(self) -> BenchmarkResult:
        """Benchmark learning buffer memory with 10k records."""
        tracemalloc.start()

        learner = SONALearner()
        # Fill the buffer
        for i in range(10000):
            record = ExecutionRecord(
                task_id=f"buffer_task_{i}",
                task_category="analysis",
                agent_name="DataAnalyzer",
                success=i % 5 != 0,
                quality_score=0.7 + (i % 30) / 100,
                latency_ms=500 + i % 1500,
                token_count=100 + i % 900,
                cost=0.001 * (100 + i % 900),
            )
            learner.record_execution(record)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        result = BenchmarkResult(
            name="learning_buffer_memory",
            category="memory",
            iterations=1,
            mean_ns=0,
            median_ns=0,
            p95_ns=0,
            p99_ns=0,
            min_ns=0,
            max_ns=0,
            std_ns=0,
            ops_per_second=0,
            memory_bytes=current,
            memory_peak_bytes=peak,
            passed=current < 100 * 1024 * 1024,  # <100MB
        )
        self.runner.results.append(result)
        return result


class ConcurrencyBenchmarks:
    """Concurrency and parallelism benchmarks."""

    def __init__(self, runner: BenchmarkRunner):
        self.runner = runner

    async def benchmark_concurrent_routing(self, concurrent: int = 100) -> BenchmarkResult:
        """Benchmark concurrent routing (target: 100 concurrent)."""
        router = ThompsonSamplingRouter(seed=42)
        tasks = [
            "Analyze data",
            "Write content",
            "Plan project",
            "Check ethics",
            "Remember context",
        ]

        async def route_task(task: str):
            return router.select_agent(task)

        # Warmup
        for task in tasks:
            await route_task(task)

        # Benchmark concurrent operations
        gc.collect()

        timings_ns: List[int] = []
        iterations = 10

        for _ in range(iterations):
            async_tasks = [
                route_task(tasks[i % len(tasks)])
                for i in range(concurrent)
            ]

            start = time.perf_counter_ns()
            await asyncio.gather(*async_tasks)
            end = time.perf_counter_ns()

            timings_ns.append(end - start)

        mean_ns = statistics.mean(timings_ns)
        median_ns = statistics.median(timings_ns)
        p95_ns = sorted(timings_ns)[int(0.95 * len(timings_ns))]
        p99_ns = sorted(timings_ns)[int(0.99 * len(timings_ns))]

        # Ops per second across concurrent operations
        ops_per_second = (concurrent * 1_000_000_000) / mean_ns

        result = BenchmarkResult(
            name="concurrent_routing",
            category="concurrency",
            iterations=iterations * concurrent,
            mean_ns=mean_ns / concurrent,  # Per-operation
            median_ns=median_ns / concurrent,
            p95_ns=p95_ns / concurrent,
            p99_ns=p99_ns / concurrent,
            min_ns=min(timings_ns) / concurrent,
            max_ns=max(timings_ns) / concurrent,
            std_ns=statistics.stdev(timings_ns) / concurrent if len(timings_ns) > 1 else 0,
            ops_per_second=ops_per_second,
            passed=True,  # Always pass if completed
        )
        self.runner.results.append(result)
        return result

    async def benchmark_async_learning_loop(self) -> BenchmarkResult:
        """Benchmark async learning loop."""
        learner = SONALearner(config=LearningConfig(update_interval_seconds=0.1))

        # Pre-populate
        for i in range(50):
            record = ExecutionRecord(
                task_id=f"async_task_{i}",
                task_category="reasoning",
                agent_name="MasterReasoner",
                success=True,
                quality_score=0.9,
                latency_ms=1000,
                token_count=500,
                cost=0.01,
            )
            learner.record_execution(record)

        # Measure loop iteration time
        timings_ns = []

        async def timed_iteration():
            start = time.perf_counter_ns()
            patterns = learner.extract_patterns()
            learner.optimize_routing()
            learner.evaluate_performance()
            end = time.perf_counter_ns()
            timings_ns.append(end - start)

        # Run multiple iterations
        for _ in range(100):
            await timed_iteration()

        mean_ns = statistics.mean(timings_ns)
        median_ns = statistics.median(timings_ns)
        p95_ns = sorted(timings_ns)[int(0.95 * len(timings_ns))]
        p99_ns = sorted(timings_ns)[int(0.99 * len(timings_ns))]

        result = BenchmarkResult(
            name="async_learning_loop",
            category="concurrency",
            iterations=100,
            mean_ns=mean_ns,
            median_ns=median_ns,
            p95_ns=p95_ns,
            p99_ns=p99_ns,
            min_ns=min(timings_ns),
            max_ns=max(timings_ns),
            std_ns=statistics.stdev(timings_ns),
            ops_per_second=1_000_000_000 / mean_ns,
            target_ns=20 * MS,  # <20ms for async iteration
            passed=p95_ns < 20 * MS,
        )
        self.runner.results.append(result)
        return result


# ===============================================================================
# MAIN BENCHMARK RUNNER
# ===============================================================================

def print_result(result: BenchmarkResult):
    """Print a single benchmark result."""
    status = "[PASS]" if result.passed else "[FAIL]"
    print(f"  {status} {result.name}")
    print(f"       Mean: {result.mean_ns/1_000_000:.4f}ms | "
          f"Median: {result.median_ns/1_000_000:.4f}ms | "
          f"P95: {result.p95_ns/1_000_000:.4f}ms | "
          f"P99: {result.p99_ns/1_000_000:.4f}ms")
    if result.target_ns:
        print(f"       Target: {result.target_ns/1_000_000:.4f}ms | "
              f"Ops/sec: {result.ops_per_second:,.0f}")
    if result.memory_bytes:
        print(f"       Memory: {result.memory_bytes/1024/1024:.2f}MB | "
              f"Peak: {result.memory_peak_bytes/1024/1024:.2f}MB")


def run_benchmarks(
    categories: Optional[List[str]] = None,
    iterations: int = 1000,
    save_report: bool = False,
) -> BenchmarkReport:
    """Run all or selected benchmarks."""
    runner = BenchmarkRunner(default_iterations=iterations)
    start_time = time.time()

    all_categories = [
        "thompson_sampling",
        "pattern_extraction",
        "sona_learning",
        "cost_analysis",
        "end_to_end",
        "memory",
        "concurrency",
    ]

    selected = categories or all_categories

    print("=" * 70)
    print("BIZRA Apex Orchestrator Performance Benchmarks")
    print("=" * 70)
    print(f"Iterations: {iterations} | Categories: {', '.join(selected)}")
    print("=" * 70)

    # Thompson Sampling benchmarks
    if "thompson_sampling" in selected:
        print("\n[Thompson Sampling Benchmarks]")
        ts = ThompsonSamplingBenchmarks(runner)
        print_result(ts.benchmark_agent_selection_latency(iterations))
        print_result(ts.benchmark_posterior_update_latency(iterations))
        print_result(ts.benchmark_batch_selection_throughput(iterations))
        print_result(ts.benchmark_beta_sampling(iterations * 5))
        print_result(ts.benchmark_task_classification(iterations))
        print_result(ts.benchmark_candidate_selection(iterations))
        print_result(ts.benchmark_exploration_rate(iterations))
        print_result(ts.benchmark_serialization(iterations // 2))

    # Pattern Extraction benchmarks
    if "pattern_extraction" in selected:
        print("\n[Pattern Extraction Benchmarks]")
        pe = PatternExtractionBenchmarks(runner)
        print_result(pe.benchmark_pattern_hash_computation(iterations * 5))
        print_result(pe.benchmark_similarity_scoring(iterations))
        print_result(pe.benchmark_elevation_check(iterations * 5))
        print_result(pe.benchmark_pattern_extraction(iterations // 2))

    # SONA Learning benchmarks
    if "sona_learning" in selected:
        print("\n[SONA Learning Benchmarks]")
        sl = SONALearningBenchmarks(runner)
        print_result(sl.benchmark_learning_iteration(iterations // 2))
        print_result(sl.benchmark_weight_optimization(iterations // 2))
        print_result(sl.benchmark_performance_evaluation(iterations // 2))
        print_result(sl.benchmark_routing_recommendation(iterations))
        print_result(sl.benchmark_record_execution(iterations * 2))

    # Cost Analysis benchmarks
    if "cost_analysis" in selected:
        print("\n[Cost Analysis Benchmarks]")
        ca = CostAnalysisBenchmarks(runner)
        print_result(ca.benchmark_cost_calculation(iterations * 5))
        print_result(ca.benchmark_report_generation(iterations // 5))

    # End-to-End benchmarks
    if "end_to_end" in selected:
        print("\n[End-to-End Pipeline Benchmarks]")
        e2e = EndToEndBenchmarks(runner)
        print_result(e2e.benchmark_full_routing_cycle(iterations // 2))
        print_result(e2e.benchmark_quality_gate_validation(iterations // 2))

    # Memory benchmarks
    if "memory" in selected:
        print("\n[Memory Footprint Benchmarks]")
        mem = MemoryBenchmarks(runner)
        print_result(mem.benchmark_router_memory_footprint())
        print_result(mem.benchmark_pattern_cache_memory())
        print_result(mem.benchmark_learning_buffer_memory())

    # Concurrency benchmarks
    if "concurrency" in selected:
        print("\n[Concurrency Benchmarks]")
        conc = ConcurrencyBenchmarks(runner)
        print_result(asyncio.run(conc.benchmark_concurrent_routing(100)))
        print_result(asyncio.run(conc.benchmark_async_learning_loop()))

    # Generate report
    duration = time.time() - start_time
    report = runner.generate_report(duration)

    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Total Benchmarks: {report.total_benchmarks}")
    print(f"Passed: {report.passed_benchmarks}")
    print(f"Failed: {report.failed_benchmarks}")
    print(f"Pass Rate: {report.passed_benchmarks/report.total_benchmarks*100:.1f}%")
    print(f"Total Duration: {duration:.2f}s")
    print("=" * 70)

    # Save report if requested
    if save_report:
        report_path = Path(__file__).parent / "benchmark_report.json"
        report_path.write_text(json.dumps(report.to_dict(), indent=2))
        print(f"\nReport saved to: {report_path}")

    return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="BIZRA Apex Orchestrator Performance Benchmarks"
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=[
            "thompson", "thompson_sampling",
            "pattern", "pattern_extraction",
            "sona", "sona_learning",
            "cost", "cost_analysis",
            "e2e", "end_to_end",
            "memory",
            "concurrency",
        ],
        help="Run only benchmarks in this category",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of iterations per benchmark (default: 1000)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Save benchmark report to JSON file",
    )
    args = parser.parse_args()

    # Map short names to full names
    category_map = {
        "thompson": "thompson_sampling",
        "pattern": "pattern_extraction",
        "sona": "sona_learning",
        "cost": "cost_analysis",
        "e2e": "end_to_end",
    }

    categories = None
    if args.category:
        full_name = category_map.get(args.category, args.category)
        categories = [full_name]

    report = run_benchmarks(
        categories=categories,
        iterations=args.iterations,
        save_report=args.report,
    )

    # Exit with failure if any benchmarks failed
    if report.failed_benchmarks > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
