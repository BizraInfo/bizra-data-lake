"""
BIZRA Benchmark Suite
Minimal local benchmarking framework for Node0 evaluation.

Measures: latency_ms, throughput_qps, memory_mb
Reports: min/avg/max/p95 per benchmark

Suites:
- inference: Local backend latency, batch throughput, token generation
- security: Crypto operations, replay detection, timing-safe comparison
- quality: SNR calculation, Ihsān scoring, type validation

Usage:
    python -m benchmark [suite] [--iterations N] [--json output.json]

Standing on Giants: Knuth (measure, don't guess), Amdahl (identify bottlenecks)
"""

__version__ = "0.1.0"
__author__ = "BIZRA Node0"

from benchmark.runner import BenchmarkRunner, BenchmarkResult, MetricResult

__all__ = [
    "BenchmarkRunner",
    "BenchmarkResult",
    "MetricResult",
]
