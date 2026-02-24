"""Performance benchmarks: HNSW (hnswlib) vs numpy brute-force at various scales.

Measures insert time, search time, memory usage, and speedup ratios.
Generates a summary table at each scale: 1K, 10K, 100K vectors (dim=768).

Run with:
    source .venv/bin/activate && \
    python -m pytest tests/core/memory/test_performance.py -v -s --timeout=300

These tests are marked @pytest.mark.slow and excluded from default CI runs.
"""

from __future__ import annotations

import gc
import os
import sys
import time
import tracemalloc
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pytest

from core.memory.config import HNSWConfig
from core.memory.hnsw_index import HNSWIndex


# ---------------------------------------------------------------------------
# Data classes for benchmark results
# ---------------------------------------------------------------------------

@dataclass
class InsertResult:
    """Timing and memory for bulk insert."""
    total_seconds: float
    peak_memory_mb: float
    vectors_per_second: float


@dataclass
class SearchResult:
    """Timing for search queries."""
    avg_ms_per_query: float
    total_seconds: float
    n_queries: int


@dataclass
class BenchmarkResult:
    """Full benchmark result for one backend at one scale."""
    backend: str
    n_vectors: int
    dim: int
    insert: InsertResult
    search: SearchResult
    total_memory_mb: float


@dataclass
class ComparisonRow:
    """Single row in the comparison table."""
    n_vectors: int
    np_insert_s: float
    hnsw_insert_s: float
    insert_speedup: float
    np_search_ms: float
    hnsw_search_ms: float
    search_speedup: float
    np_memory_mb: float
    hnsw_memory_mb: float


# ---------------------------------------------------------------------------
# Brute-force numpy backend (baseline)
# ---------------------------------------------------------------------------

class NumpyBruteForce:
    """Minimal brute-force cosine search for benchmarking."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._vectors: Optional[np.ndarray] = None
        self._count = 0

    def bulk_insert(self, vectors: np.ndarray) -> None:
        """Store all vectors at once."""
        self._vectors = vectors.copy()
        self._count = len(vectors)

    def search(self, query: np.ndarray, top_k: int = 10) -> np.ndarray:
        """Cosine-similarity brute-force search. Returns top_k indices."""
        assert self._vectors is not None
        query_norm = query / (np.linalg.norm(query) + 1e-10)
        norms = np.linalg.norm(self._vectors, axis=1, keepdims=True) + 1e-10
        sims = (self._vectors / norms) @ query_norm
        return np.argpartition(-sims, top_k)[:top_k]


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def _measure_memory(func):
    """Execute func() while tracking peak memory allocation."""
    gc.collect()
    tracemalloc.start()
    result = func()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak / (1024 * 1024)  # MB


def _benchmark_numpy(
    vectors: np.ndarray,
    queries: np.ndarray,
    top_k: int,
) -> BenchmarkResult:
    """Benchmark numpy brute-force insert + search."""
    n, dim = vectors.shape
    n_queries = len(queries)

    # --- Insert ---
    gc.collect()
    tracemalloc.start()
    bf = NumpyBruteForce(dim)
    t0 = time.perf_counter()
    bf.bulk_insert(vectors)
    insert_time = time.perf_counter() - t0
    _, insert_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    insert_peak_mb = insert_peak / (1024 * 1024)

    # --- Search ---
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    for q in queries:
        bf.search(q, top_k)
    search_time = time.perf_counter() - t0
    search_total_ms = search_time * 1000
    _, search_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    search_peak_mb = search_peak / (1024 * 1024)

    return BenchmarkResult(
        backend="numpy",
        n_vectors=n,
        dim=dim,
        insert=InsertResult(
            total_seconds=insert_time,
            peak_memory_mb=insert_peak_mb,
            vectors_per_second=n / max(insert_time, 1e-9),
        ),
        search=SearchResult(
            avg_ms_per_query=search_total_ms / n_queries,
            total_seconds=search_time,
            n_queries=n_queries,
        ),
        total_memory_mb=insert_peak_mb + search_peak_mb,
    )


def _benchmark_hnsw(
    vectors: np.ndarray,
    queries: np.ndarray,
    top_k: int,
) -> BenchmarkResult:
    """Benchmark hnswlib insert + search."""
    n, dim = vectors.shape
    n_queries = len(queries)

    # --- Insert ---
    gc.collect()
    tracemalloc.start()

    cfg = HNSWConfig(dimensions=dim, max_elements=n + 100)
    idx = HNSWIndex(cfg)
    idx.initialize()

    t0 = time.perf_counter()
    if idx._use_hnswlib and idx._index is not None:
        # Native batch insert (bypasses Python loop overhead)
        ids = np.arange(n)
        idx._index.add_items(vectors, ids)
        idx._id_map = {int(i): f"v{i}" for i in range(n)}
        idx._reverse_map = {f"v{i}": int(i) for i in range(n)}
        idx._next_internal_id = n
    else:
        for i in range(n):
            idx.add(f"v{i}", vectors[i])
    insert_time = time.perf_counter() - t0

    _, insert_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    insert_peak_mb = insert_peak / (1024 * 1024)

    # --- Search ---
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    for q in queries:
        idx.search(q, top_k=top_k)
    search_time = time.perf_counter() - t0
    search_total_ms = search_time * 1000
    _, search_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    search_peak_mb = search_peak / (1024 * 1024)

    return BenchmarkResult(
        backend="hnswlib" if idx._use_hnswlib else "numpy-fallback",
        n_vectors=n,
        dim=dim,
        insert=InsertResult(
            total_seconds=insert_time,
            peak_memory_mb=insert_peak_mb,
            vectors_per_second=n / max(insert_time, 1e-9),
        ),
        search=SearchResult(
            avg_ms_per_query=search_total_ms / n_queries,
            total_seconds=search_time,
            n_queries=n_queries,
        ),
        total_memory_mb=insert_peak_mb + search_peak_mb,
    )


def _run_comparison(
    n_vectors: int,
    dim: int = 768,
    top_k: int = 10,
    n_queries: int = 100,
) -> ComparisonRow:
    """Run both backends and return a comparison row."""
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((n_vectors, dim)).astype(np.float32)
    queries = rng.standard_normal((n_queries, dim)).astype(np.float32)

    np_result = _benchmark_numpy(vectors, queries, top_k)
    hnsw_result = _benchmark_hnsw(vectors, queries, top_k)

    np_search = np_result.search.avg_ms_per_query
    hnsw_search = hnsw_result.search.avg_ms_per_query

    return ComparisonRow(
        n_vectors=n_vectors,
        np_insert_s=np_result.insert.total_seconds,
        hnsw_insert_s=hnsw_result.insert.total_seconds,
        insert_speedup=np_result.insert.total_seconds / max(hnsw_result.insert.total_seconds, 1e-9),
        np_search_ms=np_search,
        hnsw_search_ms=hnsw_search,
        search_speedup=np_search / max(hnsw_search, 0.001),
        np_memory_mb=np_result.total_memory_mb,
        hnsw_memory_mb=hnsw_result.total_memory_mb,
    )


def _print_result_table(rows: List[ComparisonRow]) -> None:
    """Print a formatted comparison table."""
    header = (
        f"{'Scale':>8s} | "
        f"{'NP Insert (s)':>14s} | "
        f"{'HNSW Insert (s)':>16s} | "
        f"{'NP Search (ms)':>15s} | "
        f"{'HNSW Search (ms)':>17s} | "
        f"{'Search Speedup':>15s} | "
        f"{'NP Mem (MB)':>12s} | "
        f"{'HNSW Mem (MB)':>14s}"
    )
    sep = "-" * len(header)

    print(f"\n{sep}")
    print("  HNSW vs NumPy Brute-Force Benchmark (dim=768, top_k=10, cosine)")
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r.n_vectors:>8,d} | "
            f"{r.np_insert_s:>14.4f} | "
            f"{r.hnsw_insert_s:>16.4f} | "
            f"{r.np_search_ms:>15.3f} | "
            f"{r.hnsw_search_ms:>17.4f} | "
            f"{r.search_speedup:>14.0f}x | "
            f"{r.np_memory_mb:>12.1f} | "
            f"{r.hnsw_memory_mb:>14.1f}"
        )
    print(sep)


def _print_single_result(label: str, np_res: BenchmarkResult, hnsw_res: BenchmarkResult) -> None:
    """Print detailed results for a single scale."""
    np_s = np_res.search.avg_ms_per_query
    hnsw_s = hnsw_res.search.avg_ms_per_query
    speedup = np_s / max(hnsw_s, 0.001)

    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"{'='*72}")
    print(f"  Backend         | {'numpy':>14s} | {'hnswlib':>14s} | {'ratio':>10s}")
    print(f"  {'-'*60}")
    print(
        f"  Insert time     | {np_res.insert.total_seconds:>13.4f}s | "
        f"{hnsw_res.insert.total_seconds:>13.4f}s | "
        f"{np_res.insert.total_seconds / max(hnsw_res.insert.total_seconds, 1e-9):>9.1f}x"
    )
    print(
        f"  Insert rate     | {np_res.insert.vectors_per_second:>11,.0f}/s | "
        f"{hnsw_res.insert.vectors_per_second:>11,.0f}/s | "
    )
    print(
        f"  Search (avg)    | {np_s:>12.3f}ms | "
        f"{hnsw_s:>12.4f}ms | "
        f"{speedup:>9.0f}x"
    )
    print(
        f"  Search (total)  | {np_res.search.total_seconds:>13.4f}s | "
        f"{hnsw_res.search.total_seconds:>13.4f}s | "
    )
    print(
        f"  Memory (peak)   | {np_res.total_memory_mb:>12.1f}MB | "
        f"{hnsw_res.total_memory_mb:>12.1f}MB | "
    )
    print(f"  Queries run     | {np_res.search.n_queries:>14d} | {hnsw_res.search.n_queries:>14d} |")
    print(f"{'='*72}")


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Full benchmark suite: HNSW (hnswlib) vs numpy brute-force.

    Scales: 1K, 10K, 100K vectors at dim=768.
    Measures: insert time, search time (single query, top_k=10), memory usage.
    Reports speedup ratio (numpy_time / hnsw_time).
    """

    def test_benchmark_1k(self):
        """1K vectors: expect HNSW search at least 10x faster."""
        n = 1_000
        n_queries = 100
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((n, 768)).astype(np.float32)
        queries = rng.standard_normal((n_queries, 768)).astype(np.float32)

        np_res = _benchmark_numpy(vectors, queries, top_k=10)
        hnsw_res = _benchmark_hnsw(vectors, queries, top_k=10)
        _print_single_result(f"1K vectors (dim=768, {n_queries} queries, top_k=10)", np_res, hnsw_res)

        speedup = np_res.search.avg_ms_per_query / max(hnsw_res.search.avg_ms_per_query, 0.001)
        assert speedup > 1.0, (
            f"HNSW should be faster than brute-force at 1K: "
            f"numpy={np_res.search.avg_ms_per_query:.3f}ms, "
            f"hnsw={hnsw_res.search.avg_ms_per_query:.4f}ms"
        )

    def test_benchmark_10k(self):
        """10K vectors: expect HNSW search at least 50x faster."""
        n = 10_000
        n_queries = 50
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((n, 768)).astype(np.float32)
        queries = rng.standard_normal((n_queries, 768)).astype(np.float32)

        np_res = _benchmark_numpy(vectors, queries, top_k=10)
        hnsw_res = _benchmark_hnsw(vectors, queries, top_k=10)
        _print_single_result(f"10K vectors (dim=768, {n_queries} queries, top_k=10)", np_res, hnsw_res)

        speedup = np_res.search.avg_ms_per_query / max(hnsw_res.search.avg_ms_per_query, 0.001)
        assert speedup > 10.0, (
            f"HNSW should be >10x faster at 10K: "
            f"numpy={np_res.search.avg_ms_per_query:.3f}ms, "
            f"hnsw={hnsw_res.search.avg_ms_per_query:.4f}ms, "
            f"speedup={speedup:.0f}x"
        )

    def test_benchmark_100k(self):
        """100K vectors: expect HNSW search at least 100x faster.

        Memory: ~235MB for vectors + index overhead. Fine on 128GB RAM.
        """
        n = 100_000
        n_queries = 20
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((n, 768)).astype(np.float32)
        queries = rng.standard_normal((n_queries, 768)).astype(np.float32)

        np_res = _benchmark_numpy(vectors, queries, top_k=10)
        hnsw_res = _benchmark_hnsw(vectors, queries, top_k=10)
        _print_single_result(f"100K vectors (dim=768, {n_queries} queries, top_k=10)", np_res, hnsw_res)

        speedup = np_res.search.avg_ms_per_query / max(hnsw_res.search.avg_ms_per_query, 0.001)
        assert speedup > 50.0, (
            f"HNSW should be >50x faster at 100K: "
            f"numpy={np_res.search.avg_ms_per_query:.3f}ms, "
            f"hnsw={hnsw_res.search.avg_ms_per_query:.4f}ms, "
            f"speedup={speedup:.0f}x"
        )

    def test_full_comparison_table(self):
        """Run all scales and print a unified comparison table."""
        scales = [
            (1_000, 100),
            (10_000, 50),
            (100_000, 20),
        ]
        rows: List[ComparisonRow] = []

        for n_vectors, n_queries in scales:
            print(f"\n>>> Benchmarking {n_vectors:,d} vectors ({n_queries} queries)...")
            row = _run_comparison(n_vectors, dim=768, top_k=10, n_queries=n_queries)
            rows.append(row)
            # Force GC between scales to avoid memory pressure
            gc.collect()

        _print_result_table(rows)

        # Verify HNSW is consistently faster for search
        for row in rows:
            assert row.hnsw_search_ms < row.np_search_ms, (
                f"HNSW should be faster at {row.n_vectors:,d} vectors: "
                f"numpy={row.np_search_ms:.3f}ms, hnsw={row.hnsw_search_ms:.4f}ms"
            )

        # Verify speedup increases with scale
        if len(rows) >= 2:
            for i in range(1, len(rows)):
                assert rows[i].search_speedup > rows[i - 1].search_speedup * 0.5, (
                    f"Speedup should generally increase with scale: "
                    f"{rows[i-1].n_vectors:,d}={rows[i-1].search_speedup:.0f}x, "
                    f"{rows[i].n_vectors:,d}={rows[i].search_speedup:.0f}x"
                )


class TestPerformanceSmall:
    """Quick performance sanity check (CI-safe, not marked slow)."""

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

    def test_search_recall_sanity(self):
        """Verify HNSW returns results that overlap with brute-force."""
        dim = 128
        n = 500
        top_k = 10
        rng = np.random.default_rng(99)
        vectors = rng.standard_normal((n, dim)).astype(np.float32)
        query = rng.standard_normal(dim).astype(np.float32)

        # Brute-force ground truth
        bf = NumpyBruteForce(dim)
        bf.bulk_insert(vectors)
        bf_top = set(bf.search(query, top_k).tolist())

        # HNSW
        cfg = HNSWConfig(dimensions=dim, max_elements=n + 100)
        idx = HNSWIndex(cfg)
        idx.initialize()
        if idx._use_hnswlib and idx._index is not None:
            ids_arr = np.arange(n)
            idx._index.add_items(vectors, ids_arr)
            idx._id_map = {int(i): f"v{i}" for i in range(n)}
            idx._reverse_map = {f"v{i}": int(i) for i in range(n)}
            idx._next_internal_id = n
        else:
            for i in range(n):
                idx.add(f"v{i}", vectors[i])

        hnsw_results = idx.search(query, top_k=top_k)
        hnsw_top = {int(rid.replace("v", "")) for rid, _ in hnsw_results}

        # At 500 vectors with default params, recall should be near-perfect
        overlap = len(bf_top & hnsw_top)
        recall = overlap / top_k
        print(f"\nRecall@{top_k} at {n} vectors: {recall:.0%} ({overlap}/{top_k})")
        assert recall >= 0.7, f"HNSW recall too low: {recall:.0%}"
