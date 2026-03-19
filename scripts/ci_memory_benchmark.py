#!/usr/bin/env python3
"""
Memory System Performance Benchmark — CI Quality Gate.

Measures latency and throughput for AgentDB core operations with
ratcheted thresholds that prevent performance regressions.

Standing on Giants:
  Deming (1950) — PDCA continuous improvement cycle
  Shannon (1948) — SNR for signal/noise in benchmarks

Usage:
    python scripts/ci_memory_benchmark.py [--report-out PATH] [--strict]

Exit codes:
    0 — All benchmarks within ratcheted thresholds
    1 — One or more benchmarks exceeded threshold (regression)
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.memory.agent_db import AgentDB
from core.memory.config import HNSWConfig, MemoryConfig
from core.memory.memory_patterns import (
    ContextSynthesizer,
    FactStore,
    HierarchicalMemory,
    MemoryConsolidator,
    MemoryTier,
    SessionMemory,
)
from core.memory.types import MemoryKind

# ── Ratcheted Thresholds (ms) ─────────────────────────────────────────
# These are the max allowed latencies. Tighten over time (never loosen).
THRESHOLDS = {
    "store_single_ms": 50.0,
    "store_batch_100_ms": 500.0,
    "search_keyword_ms": 100.0,
    "search_hybrid_ms": 200.0,
    "mmr_rerank_ms": 100.0,
    "find_by_source_ms": 50.0,
    "session_store_turn_ms": 50.0,
    "session_get_history_ms": 50.0,
    "fact_store_retrieve_ms": 50.0,
    "hierarchical_store_ms": 50.0,
    "consolidate_1000_ms": 2000.0,
    "deduplicate_ms": 1000.0,
    "context_synthesize_ms": 200.0,
    "rebuild_indexes_ms": 2000.0,
}


@dataclass
class BenchmarkResult:
    """Single benchmark measurement."""

    name: str
    latency_ms: float
    threshold_ms: float
    passed: bool
    ops_per_sec: float = 0.0
    detail: str = ""


@dataclass
class BenchmarkReport:
    """Full benchmark report."""

    timestamp: str = ""
    results: List[Dict[str, Any]] = field(default_factory=list)
    total_passed: int = 0
    total_failed: int = 0
    all_passed: bool = True


def _time_ms(fn, *args, **kwargs):
    """Execute fn and return (result, elapsed_ms)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = (time.perf_counter() - start) * 1000
    return result, elapsed


def _make_db(tmp_dir: Path) -> AgentDB:
    """Create a fresh AgentDB in a temp directory."""
    config = MemoryConfig(
        data_dir=tmp_dir / "agent_db",
        hnsw=HNSWConfig(dimensions=64, max_elements=10000),
        auto_embed=False,
    )
    db = AgentDB(config)
    db.initialize()
    return db


def _fake_embedding(dim: int = 64, seed: int = 0) -> List[float]:
    """Deterministic pseudo-random embedding for benchmarks."""
    import hashlib

    h = hashlib.sha256(seed.to_bytes(4, "big")).digest()
    values = []
    for i in range(dim):
        byte_val = h[i % len(h)]
        values.append((byte_val / 255.0) * 2 - 1)
    return values


def run_benchmarks(tmp_dir: Path) -> List[BenchmarkResult]:
    """Run all benchmarks and return results."""
    results: List[BenchmarkResult] = []
    db = _make_db(tmp_dir)

    # ── 1. Store single record ────────────────────────────────────────
    _, ms = _time_ms(
        db.store,
        "Benchmark record content",
        importance=0.8,
        embedding=_fake_embedding(64, 0),
    )
    results.append(
        BenchmarkResult(
            "store_single_ms",
            ms,
            THRESHOLDS["store_single_ms"],
            ms <= THRESHOLDS["store_single_ms"],
        )
    )

    # ── 2. Store batch of 100 records ─────────────────────────────────
    start = time.perf_counter()
    for i in range(100):
        db.store(
            f"Batch record {i} with content for keyword search testing",
            importance=0.3 + (i % 7) * 0.1,
            embedding=_fake_embedding(64, i + 1),
            source=f"bench_src_{i % 5}",
            tags=[f"batch", f"group:{i % 3}"],
        )
    ms = (time.perf_counter() - start) * 1000
    results.append(
        BenchmarkResult(
            "store_batch_100_ms",
            ms,
            THRESHOLDS["store_batch_100_ms"],
            ms <= THRESHOLDS["store_batch_100_ms"],
            ops_per_sec=100 / (ms / 1000) if ms > 0 else 0,
        )
    )

    # ── 3. Keyword search ─────────────────────────────────────────────
    _, ms = _time_ms(db.search, query="batch record content keyword", top_k=10)
    results.append(
        BenchmarkResult(
            "search_keyword_ms",
            ms,
            THRESHOLDS["search_keyword_ms"],
            ms <= THRESHOLDS["search_keyword_ms"],
        )
    )

    # ── 4. Hybrid search (vector + keyword) ───────────────────────────
    _, ms = _time_ms(
        db.search,
        query="batch record",
        query_embedding=_fake_embedding(64, 42),
        top_k=10,
    )
    results.append(
        BenchmarkResult(
            "search_hybrid_ms",
            ms,
            THRESHOLDS["search_hybrid_ms"],
            ms <= THRESHOLDS["search_hybrid_ms"],
        )
    )

    # ── 5. Find by source (metadata-only) ─────────────────────────────
    _, ms = _time_ms(db.find, source="bench_src_0", limit=50)
    results.append(
        BenchmarkResult(
            "find_by_source_ms",
            ms,
            THRESHOLDS["find_by_source_ms"],
            ms <= THRESHOLDS["find_by_source_ms"],
        )
    )

    # ── 6. Session Memory — store turn ────────────────────────────────
    sm = SessionMemory(db, "bench-session")
    _, ms = _time_ms(sm.store_turn, "user", "Hello, this is a benchmark turn")
    results.append(
        BenchmarkResult(
            "session_store_turn_ms",
            ms,
            THRESHOLDS["session_store_turn_ms"],
            ms <= THRESHOLDS["session_store_turn_ms"],
        )
    )
    # Add more turns for history test
    for i in range(9):
        sm.store_turn("assistant" if i % 2 else "user", f"Turn {i + 2}")

    # ── 7. Session Memory — get history ───────────────────────────────
    _, ms = _time_ms(sm.get_history, limit=10)
    results.append(
        BenchmarkResult(
            "session_get_history_ms",
            ms,
            THRESHOLDS["session_get_history_ms"],
            ms <= THRESHOLDS["session_get_history_ms"],
        )
    )

    # ── 8. Fact Store — store + retrieve ──────────────────────────────
    fs = FactStore(db)
    fs.store_fact("bench_cat", "key1", "Benchmark fact value", confidence=0.95)
    _, ms = _time_ms(fs.get_fact, "bench_cat", "key1")
    results.append(
        BenchmarkResult(
            "fact_store_retrieve_ms",
            ms,
            THRESHOLDS["fact_store_retrieve_ms"],
            ms <= THRESHOLDS["fact_store_retrieve_ms"],
        )
    )

    # ── 9. Hierarchical Memory — store ────────────────────────────────
    hm = HierarchicalMemory(db)
    _, ms = _time_ms(
        hm.store, "Benchmark hierarchical memory", tier=MemoryTier.LONG_TERM
    )
    results.append(
        BenchmarkResult(
            "hierarchical_store_ms",
            ms,
            THRESHOLDS["hierarchical_store_ms"],
            ms <= THRESHOLDS["hierarchical_store_ms"],
        )
    )

    # ── 10. Consolidator — consolidate 100+ records ───────────────────
    mc = MemoryConsolidator(db)
    _, ms = _time_ms(mc.consolidate, max_records=10000, min_importance=0.05)
    results.append(
        BenchmarkResult(
            "consolidate_1000_ms",
            ms,
            THRESHOLDS["consolidate_1000_ms"],
            ms <= THRESHOLDS["consolidate_1000_ms"],
        )
    )

    # ── 11. Consolidator — deduplicate ────────────────────────────────
    # Add some duplicates
    for i in range(5):
        db.store("Exact duplicate content for dedup benchmark", source=f"dup_{i}")
    _, ms = _time_ms(mc.deduplicate)
    results.append(
        BenchmarkResult(
            "deduplicate_ms",
            ms,
            THRESHOLDS["deduplicate_ms"],
            ms <= THRESHOLDS["deduplicate_ms"],
        )
    )

    # ── 12. Context Synthesizer ───────────────────────────────────────
    cs = ContextSynthesizer(db)
    _, ms = _time_ms(cs.synthesize, "batch record keyword search")
    results.append(
        BenchmarkResult(
            "context_synthesize_ms",
            ms,
            THRESHOLDS["context_synthesize_ms"],
            ms <= THRESHOLDS["context_synthesize_ms"],
        )
    )

    # ── 13. Rebuild indexes ───────────────────────────────────────────
    _, ms = _time_ms(db.rebuild_indexes)
    results.append(
        BenchmarkResult(
            "rebuild_indexes_ms",
            ms,
            THRESHOLDS["rebuild_indexes_ms"],
            ms <= THRESHOLDS["rebuild_indexes_ms"],
        )
    )

    db.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Memory system performance benchmark")
    parser.add_argument(
        "--report-out",
        type=Path,
        default=None,
        help="Path to write JSON report",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 on any threshold breach (CI mode)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  BIZRA Memory System — Performance Benchmark")
    print("  Standing on Giants: Deming (PDCA, 1950)")
    print("=" * 60)
    print()

    with tempfile.TemporaryDirectory() as tmp:
        results = run_benchmarks(Path(tmp))

    report = BenchmarkReport(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    passed = 0
    failed = 0
    for r in results:
        status = "✅ PASS" if r.passed else "❌ FAIL"
        detail = f"  ({r.ops_per_sec:.0f} ops/s)" if r.ops_per_sec else ""
        print(
            f"  {status}  {r.name:<30s}  {r.latency_ms:8.2f}ms  (limit: {r.threshold_ms:.0f}ms){detail}"
        )
        report.results.append(asdict(r))
        if r.passed:
            passed += 1
        else:
            failed += 1

    report.total_passed = passed
    report.total_failed = failed
    report.all_passed = failed == 0

    print()
    print(f"  Results: {passed}/{passed + failed} passed")
    if failed:
        print(f"  ⚠️  {failed} benchmark(s) exceeded ratcheted threshold")
    else:
        print("  🏆 All benchmarks within performance budget")

    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(
            json.dumps(asdict(report), indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\n  Report written to: {args.report_out}")

    if args.strict and failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
