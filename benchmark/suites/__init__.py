"""
Benchmark Suites
Collection of specialized benchmark tests for different BIZRA components.
"""

from benchmark.suites.inference import InferenceBenchmark
from benchmark.suites.security import SecurityBenchmark
from benchmark.suites.quality import QualityBenchmark
from benchmark.suites.spearpoint import SpearPointBenchmark

__all__ = [
    "InferenceBenchmark",
    "SecurityBenchmark",
    "QualityBenchmark",
    "SpearPointBenchmark",
]
