"""
Benchmark Suites
Collection of specialized benchmark tests for different BIZRA components.
"""

from benchmark.suites.inference import InferenceBenchmark
from benchmark.suites.security import SecurityBenchmark
from benchmark.suites.quality import QualityBenchmark

__all__ = [
    "InferenceBenchmark",
    "SecurityBenchmark",
    "QualityBenchmark",
]
