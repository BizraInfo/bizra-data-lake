"""
Hybrid Query Engine — Score fusion across vector, keyword, recency,
importance, and graph signals.

Implements the FusableScore pattern: each signal produces a [0,1] score,
then a weighted linear combination yields the final ranking.

Weights (from plan, tunable in MemoryConfig):
  vector:     0.40  — semantic similarity via HNSW
  keyword:    0.15  — BM25 via FTS5
  recency:    0.20  — exponential decay on last_accessed
  importance: 0.15  — record.importance field
  graph:      0.10  — related_ids overlap (future: graph distance)

Standing on Giants: Robertson & Zaragoza (2009) — BM25; Malkov & Yashunin (2016) — HNSW
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .config import MemoryConfig
from .hnsw_index import HNSWIndex
from .types import QueryOptions, SearchResult
from .unified_store import UnifiedStore

# Recency half-life in hours (1 week)
_RECENCY_HALF_LIFE_HOURS = 168.0


def _recency_score(last_accessed: datetime) -> float:
    """Exponential recency decay. Returns 0.0-1.0."""
    now = datetime.now(timezone.utc)
    hours = max(0.0, (now - last_accessed).total_seconds() / 3600.0)
    return math.exp(-0.693 * hours / _RECENCY_HALF_LIFE_HOURS)


def _graph_overlap(record_related: List[str], context_ids: List[str]) -> float:
    """Simple graph score: fraction of record's related_ids in context."""
    if not record_related or not context_ids:
        return 0.0
    overlap = len(set(record_related) & set(context_ids))
    return min(1.0, overlap / max(len(record_related), 1))


class HybridQueryEngine:
    """Fuses multiple retrieval signals into a single ranked result list."""

    def __init__(
        self,
        store: UnifiedStore,
        hnsw: HNSWIndex,
        config: Optional[MemoryConfig] = None,
    ) -> None:
        self._store = store
        self._hnsw = hnsw
        self._config = config or MemoryConfig()

    def search(
        self,
        options: QueryOptions,
        context_ids: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """Execute a hybrid search combining all signals.

        Args:
            options: Query parameters (text, embedding, top_k, filters).
            context_ids: IDs of records in current working context (for graph score).

        Returns:
            Sorted list of SearchResult (highest score first).
        """
        candidate_scores: Dict[str, Dict[str, float]] = {}
        context_ids = context_ids or []

        # Expand candidate pool size to allow re-ranking
        fetch_k = min(options.top_k * 3, 200)

        # ── Signal 1: Vector similarity (HNSW) ─────────────────────────
        if options.query_embedding is not None and self._hnsw.count > 0:
            hnsw_results = self._hnsw.search(options.query_embedding, top_k=fetch_k)
            for record_id, distance in hnsw_results:
                # Cosine distance -> similarity: sim = 1 - dist
                sim = max(0.0, 1.0 - distance)
                candidate_scores.setdefault(record_id, {})["vector"] = sim

        # ── Signal 2: Keyword (FTS5 BM25) ──────────────────────────────
        if options.query_text:
            try:
                fts_results = self._store.keyword_search(
                    options.query_text, top_k=fetch_k
                )
                for record_id, score in fts_results:
                    candidate_scores.setdefault(record_id, {})["keyword"] = score
            except Exception:  # noqa: BLE001 — boundary boundary
                pass  # FTS query syntax error — skip keyword signal

        # If no candidates from vector or keyword, return empty
        if not candidate_scores:
            return []

        # ── Fetch records for remaining signals ────────────────────────
        results: List[SearchResult] = []

        for record_id, signals in candidate_scores.items():
            record = self._store.get(record_id)
            if record is None:
                continue

            # Apply filters
            if not options.include_archived and record.state.value != "active":
                continue
            if options.kinds and record.kind not in options.kinds:
                continue
            if options.tags and not set(options.tags) & set(record.tags):
                continue
            if options.source and record.source != options.source:
                continue

            # ── Signal 3: Recency ──────────────────────────────────────
            recency = _recency_score(record.last_accessed)

            # ── Signal 4: Importance ───────────────────────────────────
            importance = min(1.0, max(0.0, record.importance))

            # ── Signal 5: Graph overlap ────────────────────────────────
            graph = _graph_overlap(record.related_ids, context_ids)

            # ── Weighted fusion ────────────────────────────────────────
            vector_s = signals.get("vector", 0.0)
            keyword_s = signals.get("keyword", 0.0)

            fused = (
                self._config.weight_vector * vector_s
                + self._config.weight_keyword * keyword_s
                + self._config.weight_recency * recency
                + self._config.weight_importance * importance
                + self._config.weight_graph * graph
            )

            if fused >= options.min_score:
                results.append(
                    SearchResult(
                        record=record,
                        score=fused,
                        vector_score=vector_s,
                        keyword_score=keyword_s,
                        recency_score=recency,
                        importance_score=importance,
                        graph_score=graph,
                    )
                )

        # Sort by fused score descending
        results.sort(key=lambda r: r.score, reverse=True)

        # Update access stats for returned results
        for result in results[: options.top_k]:
            self._store.touch_access(result.record.id)

        return results[: options.top_k]
