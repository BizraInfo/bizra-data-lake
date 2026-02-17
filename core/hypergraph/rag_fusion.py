"""
HyperGraphRAGFusion -- triple-source retrieval with graph-augmented ranking.

Fuses three retrieval signals:

1. **Vector** -- dense cosine similarity (via ``agent_db.search``).
2. **Keyword** -- sparse BM25/lexical (via ``agent_db.keyword_search``).
3. **Graph hop** -- one-hop expansion from seed results through hyperedges.

Plus two static priors: recency and importance.

Standing on Giants: Croft (2009) -- Fusion retrieval, Shannon (1948) -- SNR
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .hypergraph_store import HyperGraphStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retrieval result
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """A single retrieval candidate with per-source scores.

    Attributes:
        id:               Unique identifier of the retrieved item.
        content:          Text content of the item.
        fused_score:      Weighted combination of all sub-scores.
        vector_score:     Cosine similarity from dense retrieval.
        keyword_score:    Lexical match score.
        graph_score:      Hypergraph expansion score.
        recency_score:    Time-decay prior.
        importance_score: Static importance prior.
        domain:           Knowledge domain of the item.
    """

    id: str
    content: str
    fused_score: float = 0.0
    vector_score: float = 0.0
    keyword_score: float = 0.0
    graph_score: float = 0.0
    recency_score: float = 0.0
    importance_score: float = 0.0
    domain: str = ""


# ---------------------------------------------------------------------------
# Fusion weights
# ---------------------------------------------------------------------------

# These five weights MUST sum to 1.0.
_WEIGHT_VECTOR: float = 0.40
_WEIGHT_KEYWORD: float = 0.15
_WEIGHT_GRAPH_HOP: float = 0.25
_WEIGHT_RECENCY: float = 0.10
_WEIGHT_IMPORTANCE: float = 0.10

FUSION_WEIGHTS: Dict[str, float] = {
    "vector": _WEIGHT_VECTOR,
    "keyword": _WEIGHT_KEYWORD,
    "graph_hop": _WEIGHT_GRAPH_HOP,
    "recency": _WEIGHT_RECENCY,
    "importance": _WEIGHT_IMPORTANCE,
}


# ---------------------------------------------------------------------------
# HyperGraphRAGFusion
# ---------------------------------------------------------------------------

class HyperGraphRAGFusion:
    """Triple-source retrieval with hypergraph-augmented re-ranking.

    Args:
        store:    The :class:`HyperGraphStore` used for graph expansion.
        agent_db: Optional external database supporting ``search()`` and
                  ``keyword_search()`` methods.  When *None*, vector and
                  keyword sources return empty candidate lists.
    """

    def __init__(
        self,
        store: HyperGraphStore,
        agent_db: Optional[Any] = None,
    ) -> None:
        self._store = store
        self._agent_db = agent_db

    # -- public API ---------------------------------------------------------

    def retrieve(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """Execute triple-source retrieval and return fused results.

        Args:
            query:           Natural-language query string.
            query_embedding: Dense vector for the query (used by vector and
                             graph sources).
            top_k:           Maximum results to return.

        Returns:
            Sorted list of :class:`RetrievalResult` (highest fused_score
            first), capped at *top_k*.
        """
        candidates: Dict[str, RetrievalResult] = {}

        # Source 1 -- Vector retrieval
        vector_results = self._vector_search(query_embedding)
        for item_id, score, content, domain in vector_results:
            self._ensure_candidate(candidates, item_id, content, domain)
            candidates[item_id].vector_score = score

        # Source 2 -- Keyword retrieval
        keyword_results = self._keyword_search(query)
        for item_id, score, content, domain in keyword_results:
            self._ensure_candidate(candidates, item_id, content, domain)
            candidates[item_id].keyword_score = score

        # Source 3 -- Graph-hop expansion (seeded from vector results)
        seed_node_ids = {item_id for item_id, *_ in vector_results}
        graph_results = self._graph_expand(seed_node_ids)
        for item_id, score, domain in graph_results:
            self._ensure_candidate(candidates, item_id, "", domain)
            candidates[item_id].graph_score = score

        # Compute fused scores
        for result in candidates.values():
            result.fused_score = (
                _WEIGHT_VECTOR * result.vector_score
                + _WEIGHT_KEYWORD * result.keyword_score
                + _WEIGHT_GRAPH_HOP * result.graph_score
                + _WEIGHT_RECENCY * result.recency_score
                + _WEIGHT_IMPORTANCE * result.importance_score
            )

        ranked = sorted(
            candidates.values(),
            key=lambda r: r.fused_score,
            reverse=True,
        )
        return ranked[:top_k]

    # -- private source methods ---------------------------------------------

    def _vector_search(
        self,
        query_embedding: Optional[List[float]],
    ) -> List[tuple[str, float, str, str]]:
        """(id, score, content, domain) from dense retrieval."""
        if self._agent_db is None or query_embedding is None:
            return []
        try:
            raw = self._agent_db.search(query_embedding)
            return [
                (
                    str(r.get("id", "")),
                    float(r.get("score", 0.0)),
                    str(r.get("content", "")),
                    str(r.get("domain", "")),
                )
                for r in raw
            ]
        except Exception:
            logger.warning("Vector search failed; returning empty list.")
            return []

    def _keyword_search(
        self,
        query: str,
    ) -> List[tuple[str, float, str, str]]:
        """(id, score, content, domain) from keyword/lexical retrieval."""
        if self._agent_db is None:
            return []
        search_fn = getattr(self._agent_db, "keyword_search", None)
        if search_fn is None:
            return []
        try:
            raw = search_fn(query)
            return [
                (
                    str(r.get("id", "")),
                    float(r.get("score", 0.0)),
                    str(r.get("content", "")),
                    str(r.get("domain", "")),
                )
                for r in raw
            ]
        except Exception:
            logger.warning("Keyword search failed; returning empty list.")
            return []

    def _graph_expand(
        self,
        seed_ids: set[str],
    ) -> List[tuple[str, float, str]]:
        """(id, graph_score, domain) from one-hop graph expansion.

        For each seed node present in the store, expand to its neighbors.
        Score each neighbor by the mean edge weight of connecting edges
        plus a log-cardinality bonus.
        """
        results: Dict[str, tuple[float, str]] = {}

        for seed_id in seed_ids:
            neighbors = self._store.get_neighbors(seed_id)
            for neighbor_id in neighbors:
                if neighbor_id in seed_ids:
                    continue  # skip items already in the seed set
                edges = self._store.get_hyperedges(neighbor_id)
                if not edges:
                    continue

                weights = [e.weight for e in edges]
                cardinalities = [e.cardinality for e in edges]
                avg_weight = statistics.mean(weights)
                cardinality_bonus = math.log(
                    1 + statistics.mean(cardinalities),
                )
                score = min(avg_weight + 0.1 * cardinality_bonus, 1.0)

                # Determine domain from node if available
                node = self._store._nodes.get(neighbor_id)
                domain = node.domain if node else ""

                # Keep the best score seen for this neighbor
                if neighbor_id not in results or score > results[neighbor_id][0]:
                    results[neighbor_id] = (score, domain)

        return [
            (nid, score, domain)
            for nid, (score, domain) in results.items()
        ]

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _ensure_candidate(
        candidates: Dict[str, RetrievalResult],
        item_id: str,
        content: str,
        domain: str,
    ) -> None:
        """Lazily initialise a candidate in *candidates* if absent."""
        if item_id not in candidates:
            candidates[item_id] = RetrievalResult(
                id=item_id,
                content=content,
                domain=domain,
            )
        else:
            # Backfill content/domain if the existing entry was a stub.
            if content and not candidates[item_id].content:
                candidates[item_id].content = content
            if domain and not candidates[item_id].domain:
                candidates[item_id].domain = domain
