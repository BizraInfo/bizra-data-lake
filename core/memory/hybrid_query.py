"""
Hybrid Query Engine — Score fusion across vector, keyword, recency,
importance, and graph signals. Includes MMR diversity re-ranking.

Implements the FusableScore pattern: each signal produces a [0,1] score,
then a weighted linear combination yields the final ranking.

Weights (from plan, tunable in MemoryConfig):
  vector:     0.40  — semantic similarity via HNSW
  keyword:    0.15  — BM25 via FTS5
  recency:    0.20  — exponential decay on last_accessed
  importance: 0.15  — record.importance field
  graph:      0.10  — related_ids overlap (future: graph distance)

MMR (Maximal Marginal Relevance) re-ranking prevents redundant results
by balancing relevance against diversity in embedding space.

Standing on Giants:
  Robertson & Zaragoza (2009) — BM25
  Malkov & Yashunin (2016) — HNSW
  Carbonell & Goldstein (1998) — MMR
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

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


def _match_metadata_filters(metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """Check if record metadata passes all filter conditions.

    Supported operators:
      exact:      {"key": value}           — equality
      $gte/$lte:  {"key": {"$gte": 5}}     — range comparisons
      $in:        {"key": {"$in": [a, b]}} — membership
      $contains:  {"key": {"$contains": x}} — substring/element check
    """
    for key, condition in filters.items():
        val = metadata.get(key)
        if isinstance(condition, dict):
            # Operator-based filter
            if "$gte" in condition and (val is None or val < condition["$gte"]):
                return False
            if "$lte" in condition and (val is None or val > condition["$lte"]):
                return False
            if "$in" in condition and val not in condition["$in"]:
                return False
            if "$contains" in condition:
                needle = condition["$contains"]
                if isinstance(val, (list, tuple, set)):
                    if needle not in val:
                        return False
                elif isinstance(val, str):
                    if needle not in val:
                        return False
                else:
                    return False
        else:
            # Exact match
            if val != condition:
                return False
    return True


def _mmr_rerank(
    results: List[SearchResult],
    query_embedding: Optional[Any],
    lam: float,
    top_k: int,
) -> List[SearchResult]:
    """Maximal Marginal Relevance re-ranking.

    Iteratively selects results that are both relevant to the query
    and diverse relative to already-selected results.

    Standing on Giants: Carbonell & Goldstein (1998) — MMR

    Args:
        results: Candidate results (pre-ranked by fused score).
        query_embedding: Query vector (None → skip MMR, return as-is).
        lam: Balance factor. 0=max diversity, 1=max relevance.
        top_k: Number of results to return.

    Returns:
        Re-ranked list of SearchResult.
    """
    if query_embedding is None or len(results) <= 1:
        return results[:top_k]

    # Collect embeddings; skip records without them
    candidates = []
    for r in results:
        if r.record.embedding is not None:
            candidates.append(r)
    if len(candidates) <= 1:
        return results[:top_k]

    q_vec = np.asarray(query_embedding, dtype=np.float32)
    q_norm = np.linalg.norm(q_vec) + 1e-10

    # Pre-compute cosine similarities to query
    embeddings = np.array([r.record.embedding for r in candidates], dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    normed = embeddings / norms
    q_normed = q_vec / q_norm
    query_sims = normed @ q_normed  # shape: (n,)

    selected: List[int] = []
    remaining = set(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = -1
        best_mmr = -float("inf")

        for idx in remaining:
            relevance = query_sims[idx]

            # Max similarity to already-selected
            if selected:
                sel_vecs = normed[selected]
                sims_to_selected = sel_vecs @ normed[idx]
                max_sim = float(np.max(sims_to_selected))
            else:
                max_sim = 0.0

            mmr_score = lam * relevance - (1.0 - lam) * max_sim
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx

        if best_idx < 0:
            break
        selected.append(best_idx)
        remaining.discard(best_idx)

    return [candidates[i] for i in selected]


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

            # Apply metadata filters
            if options.metadata_filters:
                if not _match_metadata_filters(
                    record.metadata, options.metadata_filters
                ):
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

        # MMR diversity re-ranking (Carbonell & Goldstein, 1998)
        if options.use_mmr and options.query_embedding is not None:
            results = _mmr_rerank(
                results,
                query_embedding=options.query_embedding,
                lam=options.mmr_lambda,
                top_k=options.top_k,
            )
        else:
            results = results[: options.top_k]

        # Update access stats for returned results
        for result in results:
            self._store.touch_access(result.record.id)

        return results
