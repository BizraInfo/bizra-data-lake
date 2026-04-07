"""Recall@k Benchmark — D6 Deliverable (BIZRA-STS-001)

Known-item search evaluation over the GOLD corpus (84,795 chunks, 384-dim).
Measures Recall@1, Recall@5, Recall@10, MRR, and nDCG@10 for:
  - FAISS (flat/IVF, exact)
  - RuVector (HNSW, approximate)
  - Hybrid RRF (fused)

Methodology: Sample N chunks from the corpus. For each chunk, extract a
distinguishing phrase (first sentence or key phrase) as the query. Ground
truth: the query's source chunk_id must appear in the top-k results.
This is "known-item search" — the standard evaluation when human relevance
judgments are unavailable.

Standing on Giants:
  Cormack, Clarke & Buettcher (2009) — RRF
  Voorhees (2001) — known-item search evaluation
  Jarvelin & Kekalainen (2002) — nDCG

Usage:
    pytest tests/core/search/test_recall_benchmark.py -v -s
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
import pytest

logger = logging.getLogger(__name__)

# ─── Configuration ──────────────────────────────────────────────────
SAMPLE_SIZE = 100  # Number of known-item queries
RANDOM_SEED = 42  # Reproducibility
TOP_K_VALUES = [1, 5, 10]  # k values for Recall@k
MIN_CHUNK_LENGTH = 80  # Skip very short chunks (poor queries)
MAX_QUERY_LENGTH = 200  # Truncate query to first ~200 chars


def _resolve_root() -> Path:
    if env_root := os.getenv("BIZRA_DATA_LAKE_ROOT"):
        return Path(env_root)
    return Path(__file__).resolve().parent.parent.parent.parent


@dataclass
class BenchmarkResult:
    """Metrics for one search engine."""

    engine: str
    recall_at: dict[int, float] = field(default_factory=dict)  # k -> recall
    mrr: float = 0.0
    ndcg_at_10: float = 0.0
    avg_latency_ms: float = 0.0
    queries_run: int = 0
    queries_found: dict[int, int] = field(default_factory=dict)  # k -> count found


def _extract_query(chunk_text: str) -> str:
    """Extract a search query from chunk text.

    Uses the first sentence or first MAX_QUERY_LENGTH chars, whichever is
    shorter. Strips common boilerplate.
    """
    text = chunk_text.strip()
    # Try first sentence
    for sep in [". ", ".\n", "? ", "! "]:
        idx = text.find(sep)
        if 20 < idx < MAX_QUERY_LENGTH:
            return text[: idx + 1].strip()
    # Fallback: first N chars
    return text[:MAX_QUERY_LENGTH].strip()


def _dcg(relevances: list[float], k: int) -> float:
    """Discounted Cumulative Gain at k."""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))


def _ndcg(relevances: list[float], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    dcg_val = _dcg(relevances, k)
    ideal = _dcg(sorted(relevances, reverse=True), k)
    return dcg_val / ideal if ideal > 0 else 0.0


def _load_sample_chunks(root: Path, n: int, seed: int) -> list[dict]:
    """Load N random chunks from the GOLD corpus."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        pytest.skip("pyarrow not available")

    chunks_path = root / "04_GOLD" / "chunks.parquet"
    if not chunks_path.exists():
        pytest.skip(f"GOLD corpus not found: {chunks_path}")

    table = pq.read_table(
        chunks_path, columns=["chunk_id", "doc_id", "chunk_text", "token_est"]
    )
    total = table.num_rows
    logger.info("GOLD corpus: %d chunks", total)

    # Filter to chunks with sufficient text
    rng = random.Random(seed)
    candidates = []
    # Sample more than needed to account for filtering
    indices = rng.sample(range(total), min(n * 3, total))

    for idx in indices:
        text = str(table.column("chunk_text")[idx])
        if len(text) >= MIN_CHUNK_LENGTH:
            candidates.append(
                {
                    "chunk_id": str(table.column("chunk_id")[idx]),
                    "doc_id": str(table.column("doc_id")[idx]),
                    "chunk_text": text,
                    "token_est": table.column("token_est")[idx].as_py(),
                }
            )
        if len(candidates) >= n:
            break

    logger.info(
        "Selected %d evaluation chunks (from %d candidates)",
        len(candidates),
        len(indices),
    )
    return candidates[:n]


def _run_engine_benchmark(
    engine: object,
    engine_name: str,
    queries: list[tuple[str, str]],  # (query_text, ground_truth_chunk_id)
) -> BenchmarkResult:
    """Run all queries against one engine and compute metrics."""
    result = BenchmarkResult(engine=engine_name)
    max_k = max(TOP_K_VALUES)
    reciprocal_ranks: list[float] = []
    ndcg_scores: list[float] = []
    latencies: list[float] = []

    for query_text, gt_chunk_id in queries:
        t0 = time.perf_counter()
        try:
            hits = engine.search(query_text, top_k=max_k, min_score=0.0)
        except Exception as exc:
            logger.warning("%s search failed for query: %s", engine_name, exc)
            hits = []
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        # Extract chunk_ids from results (via source_id or content match)
        hit_ids = []
        for h in hits:
            sid = getattr(h.record, "source_id", None) or ""
            hit_ids.append(sid)

        # Recall@k: is gt_chunk_id in top-k?
        for k in TOP_K_VALUES:
            found = gt_chunk_id in hit_ids[:k]
            result.queries_found[k] = result.queries_found.get(k, 0) + (
                1 if found else 0
            )

        # MRR: reciprocal rank of first hit
        if gt_chunk_id in hit_ids:
            rank = hit_ids.index(gt_chunk_id) + 1
            reciprocal_ranks.append(1.0 / rank)
            relevances = [1.0 if hid == gt_chunk_id else 0.0 for hid in hit_ids[:10]]
        else:
            reciprocal_ranks.append(0.0)
            relevances = [0.0] * min(10, len(hit_ids))

        # nDCG@10
        ndcg_scores.append(_ndcg(relevances, 10))

    result.queries_run = len(queries)
    for k in TOP_K_VALUES:
        result.recall_at[k] = result.queries_found.get(k, 0) / max(1, len(queries))
    result.mrr = sum(reciprocal_ranks) / max(1, len(reciprocal_ranks))
    result.ndcg_at_10 = sum(ndcg_scores) / max(1, len(ndcg_scores))
    result.avg_latency_ms = sum(latencies) / max(1, len(latencies))

    return result


def _format_results(results: list[BenchmarkResult]) -> str:
    """Format benchmark results as a readable table."""
    lines = [
        "",
        "=" * 72,
        "  RECALL@k BENCHMARK RESULTS — D6 Deliverable",
        "=" * 72,
        "",
        f"  {'Engine':<20} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'MRR':>8} {'nDCG@10':>8} {'Latency':>10}",
        f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}",
    ]
    for r in results:
        lines.append(
            f"  {r.engine:<20} "
            f"{r.recall_at.get(1, 0):.4f}   "
            f"{r.recall_at.get(5, 0):.4f}   "
            f"{r.recall_at.get(10, 0):.4f}   "
            f"{r.mrr:.4f}   "
            f"{r.ndcg_at_10:.4f}   "
            f"{r.avg_latency_ms:>7.1f} ms"
        )
    lines.extend(["", f"  Queries: {results[0].queries_run if results else 0}", ""])
    return "\n".join(lines)


def _write_metrics_canonical(
    root: Path, results: list[BenchmarkResult], sample_size: int
) -> None:
    """Append or update recall@k section in METRICS_CANONICAL.md."""
    metrics_path = root / "docs" / "canon" / "METRICS_CANONICAL.md"

    section = [
        "",
        "## Recall@k Benchmark (D6)",
        "",
        f"**Date:** 2026-04-07 | **Method:** Known-item search | **N={sample_size}** | **Seed:** {RANDOM_SEED}",
        "**Corpus:** 84,795 chunks (384-dim embeddings) from 1,437 documents",
        "",
        "| Engine | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 | Avg Latency |",
        "|--------|----------|----------|-----------|-----|---------|-------------|",
    ]
    for r in results:
        section.append(
            f"| {r.engine} | {r.recall_at.get(1, 0):.4f} | {r.recall_at.get(5, 0):.4f} | "
            f"{r.recall_at.get(10, 0):.4f} | {r.mrr:.4f} | {r.ndcg_at_10:.4f} | "
            f"{r.avg_latency_ms:.1f} ms |"
        )
    section.extend(
        [
            "",
            "**Methodology:** For each query, a chunk is randomly sampled from the GOLD corpus.",
            "The first sentence (or first 200 chars) is used as the query. Ground truth: the source",
            "chunk_id must appear in the top-k results. This is known-item search evaluation",
            "(Voorhees, 2001). Queries are deterministic (seed=42) for reproducibility.",
            "",
            "**Verdict:** See values above. If Recall@10 < 0.50 for any engine, the previous",
            "SNR claims are OVERCLAIMED and labeled accordingly. If Recall@10 >= 0.70 for hybrid,",
            "the retrieval pipeline is VERIFIED.",
            "",
        ]
    )

    if metrics_path.exists():
        existing = metrics_path.read_text()
        # Remove old section if present
        marker = "## Recall@k Benchmark (D6)"
        if marker in existing:
            before = existing[: existing.index(marker)]
            # Find next ## header or end of file
            rest = existing[existing.index(marker) + len(marker) :]
            next_header = rest.find("\n## ")
            if next_header >= 0:
                after = rest[next_header:]
            else:
                after = ""
            existing = before.rstrip() + after
        content = existing.rstrip() + "\n" + "\n".join(section)
    else:
        content = (
            "# METRICS_CANONICAL\n\nCanonical benchmark results for the BIZRA system.\n"
            + "\n".join(section)
        )

    metrics_path.write_text(content)


@pytest.fixture
def bizra_root() -> Path:
    """Resolve project root."""
    return _resolve_root()


@pytest.mark.slow
def test_recall_benchmark(bizra_root: Path) -> None:
    """Run the full recall@k benchmark and publish results.

    This is a slow test (~2-5 min) that evaluates retrieval quality.
    Run explicitly: pytest tests/core/search/test_recall_benchmark.py -v -s -m slow
    """
    # 1. Load sample chunks
    chunks = _load_sample_chunks(bizra_root, SAMPLE_SIZE, RANDOM_SEED)
    assert len(chunks) >= 50, f"Need at least 50 chunks, got {len(chunks)}"

    # 2. Build query-relevance pairs
    queries = []
    for chunk in chunks:
        query_text = _extract_query(chunk["chunk_text"])
        if len(query_text) >= 20:
            queries.append((query_text, chunk["chunk_id"]))
    logger.info("Built %d query-relevance pairs", len(queries))
    assert len(queries) >= 50, f"Need at least 50 valid queries, got {len(queries)}"

    # 3. Initialize engines
    results: list[BenchmarkResult] = []

    # FAISS
    try:
        from core.search.vector_search import VectorSearchEngine

        faiss_engine = VectorSearchEngine(root=bizra_root)
        faiss_engine._ensure_loaded()
        logger.info("FAISS engine loaded: %d vectors", faiss_engine.vector_count)
        faiss_result = _run_engine_benchmark(faiss_engine, "FAISS", queries)
        results.append(faiss_result)
        logger.info(
            "FAISS: R@1=%.4f R@5=%.4f R@10=%.4f MRR=%.4f",
            faiss_result.recall_at[1],
            faiss_result.recall_at[5],
            faiss_result.recall_at[10],
            faiss_result.mrr,
        )
    except Exception as exc:
        logger.warning("FAISS benchmark skipped: %s", exc)

    # HNSW (native hnswlib — replaces RuVector subprocess bridge)
    try:
        from core.search.hnsw_search import HnswSearchEngine

        hnsw_engine = HnswSearchEngine(root=bizra_root)
        if hnsw_engine.is_available:
            hnsw_engine._ensure_loaded()
            logger.info("HNSW engine loaded: %d vectors", hnsw_engine.vector_count)
            hnsw_result = _run_engine_benchmark(hnsw_engine, "HNSW", queries)
            results.append(hnsw_result)
            logger.info(
                "HNSW: R@1=%.4f R@5=%.4f R@10=%.4f MRR=%.4f",
                hnsw_result.recall_at[1],
                hnsw_result.recall_at[5],
                hnsw_result.recall_at[10],
                hnsw_result.mrr,
            )
        else:
            logger.warning("HNSW benchmark skipped: chunks.parquet not found")
    except Exception as exc:
        logger.warning("HNSW benchmark skipped: %s", exc)

    # Hybrid RRF
    if len(results) >= 1:
        try:
            from core.search.hybrid_search import HybridSearchEngine

            hybrid = HybridSearchEngine()
            hybrid_result = _run_engine_benchmark(hybrid, "Hybrid RRF", queries)
            results.append(hybrid_result)
            logger.info(
                "Hybrid: R@1=%.4f R@5=%.4f R@10=%.4f MRR=%.4f",
                hybrid_result.recall_at[1],
                hybrid_result.recall_at[5],
                hybrid_result.recall_at[10],
                hybrid_result.mrr,
            )
        except Exception as exc:
            logger.warning("Hybrid benchmark skipped: %s", exc)

    # 4. Report
    assert len(results) >= 1, "No search engines available for benchmark"
    report = _format_results(results)
    print(report)

    # 5. Publish to METRICS_CANONICAL.md
    _write_metrics_canonical(bizra_root, results, len(queries))
    logger.info("Results published to docs/canon/METRICS_CANONICAL.md")

    # 6. Write raw results as evidence
    evidence_dir = bizra_root / "evidence" / "d6_recall_benchmark"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = evidence_dir / "recall_benchmark_results.json"
    evidence = {
        "date": "2026-04-07",
        "spearpoint": "BIZRA-STS-001",
        "deliverable": "D6",
        "sample_size": len(queries),
        "seed": RANDOM_SEED,
        "corpus_chunks": 84795,
        "corpus_docs": 1437,
        "engines": [
            {
                "name": r.engine,
                "recall_at_1": r.recall_at.get(1, 0),
                "recall_at_5": r.recall_at.get(5, 0),
                "recall_at_10": r.recall_at.get(10, 0),
                "mrr": r.mrr,
                "ndcg_at_10": r.ndcg_at_10,
                "avg_latency_ms": round(r.avg_latency_ms, 2),
                "queries_run": r.queries_run,
            }
            for r in results
        ],
    }
    evidence_path.write_text(json.dumps(evidence, indent=2))

    # 7. Verdicts — honest assessment
    for r in results:
        r10 = r.recall_at.get(10, 0)
        if r10 < 0.50:
            logger.warning(
                "OVERCLAIM: %s Recall@10=%.4f — below 0.50 threshold", r.engine, r10
            )
        elif r10 >= 0.70:
            logger.info(
                "VERIFIED: %s Recall@10=%.4f — meets 0.70 threshold", r.engine, r10
            )
        else:
            logger.info(
                "PARTIAL: %s Recall@10=%.4f — between 0.50 and 0.70", r.engine, r10
            )
