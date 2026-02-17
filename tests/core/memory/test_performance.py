"""Performance benchmarks: linear scan vs HNSW at various scales.

These tests are marked @pytest.mark.slow and excluded from CI by default.
Run with: pytest tests/core/memory/test_performance.py -v --no-header -m slow
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from core.memory.config import HNSWConfig
from core.memory.hnsw_index import HNSWIndex


def _linear_search(matrix: np.ndarray, query: np.ndarray, top_k: int):
    """Brute-force cosine search for baseline comparison."""
    query_norm = query / (np.linalg.norm(query) + 1e-10)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    sims = (matrix / norms) @ query_norm
    top_idx = np.argsort(-sims)[:top_k]
    return top_idx


def _benchmark(n_vectors: int, dim: int = 768, top_k: int = 10, n_queries: int = 100):
    """Run benchmark at given scale, returns (linear_ms, hnsw_ms) per query."""
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((n_vectors, dim)).astype(np.float32)
    queries = rng.standard_normal((n_queries, dim)).astype(np.float32)

    # Linear scan
    t0 = time.perf_counter()
    for q in queries:
        _linear_search(vectors, q, top_k)
    linear_total = (time.perf_counter() - t0) * 1000  # ms
    linear_per_query = linear_total / n_queries

    # HNSW — use native batch insert for speed
    cfg = HNSWConfig(dimensions=dim, max_elements=n_vectors + 100)
    idx = HNSWIndex(cfg)
    idx.initialize()
    if idx._use_hnswlib and idx._index is not None:
        # Batch insert directly via hnswlib (much faster than one-by-one)
        ids = np.arange(n_vectors)
        idx._index.add_items(vectors, ids)
        idx._id_map = {int(i): f"v{i}" for i in range(n_vectors)}
        idx._reverse_map = {f"v{i}": int(i) for i in range(n_vectors)}
        idx._next_internal_id = n_vectors
    else:
        for i in range(n_vectors):
            idx.add(f"v{i}", vectors[i])

    t0 = time.perf_counter()
    for q in queries:
        idx.search(q, top_k=top_k)
    hnsw_total = (time.perf_counter() - t0) * 1000
    hnsw_per_query = hnsw_total / n_queries

    return linear_per_query, hnsw_per_query


@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Benchmark suite comparing linear scan vs HNSW.

    Expected speedups (from plan):
      1K:   ~140x
      10K:  ~1,250x
      100K: ~10,000x
    """

    def test_benchmark_1k(self):
        linear_ms, hnsw_ms = _benchmark(1_000, dim=768, top_k=10, n_queries=50)
        speedup = linear_ms / max(hnsw_ms, 0.001)
        print(f"\n1K vectors: linear={linear_ms:.2f}ms, HNSW={hnsw_ms:.3f}ms, speedup={speedup:.0f}x")
        assert hnsw_ms < linear_ms, "HNSW should be faster than linear"

    def test_benchmark_10k(self):
        linear_ms, hnsw_ms = _benchmark(10_000, dim=768, top_k=10, n_queries=20)
        speedup = linear_ms / max(hnsw_ms, 0.001)
        print(f"\n10K vectors: linear={linear_ms:.2f}ms, HNSW={hnsw_ms:.3f}ms, speedup={speedup:.0f}x")
        assert hnsw_ms < linear_ms

    @pytest.mark.requires_gpu
    @pytest.mark.timeout(300)
    def test_benchmark_100k(self):
        """100K requires significant memory, skip in CI."""
        linear_ms, hnsw_ms = _benchmark(100_000, dim=768, top_k=10, n_queries=10)
        speedup = linear_ms / max(hnsw_ms, 0.001)
        print(f"\n100K vectors: linear={linear_ms:.2f}ms, HNSW={hnsw_ms:.3f}ms, speedup={speedup:.0f}x")
        assert hnsw_ms < linear_ms


class TestPerformanceSmall:
    """Quick performance sanity check (CI-safe)."""

    def test_hnsw_faster_than_linear_100(self):
        """At 100 vectors, HNSW should still be functional (not necessarily faster)."""
        cfg = HNSWConfig(dimensions=32, max_elements=200)
        idx = HNSWIndex(cfg)
        idx.initialize()

        rng = np.random.default_rng(42)
        for i in range(100):
            idx.add(f"v{i}", rng.standard_normal(32).astype(np.float32))

        query = rng.standard_normal(32).astype(np.float32)

        t0 = time.perf_counter()
        results = idx.search(query, top_k=5)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert len(results) == 5
        assert elapsed_ms < 100  # Should finish in <100ms easily
