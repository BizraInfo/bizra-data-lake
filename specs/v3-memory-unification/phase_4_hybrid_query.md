# Phase 4: Hybrid Query Engine

> ADR-009 | Hybrid Memory Backend — Score Fusion
> Standing on Giants: Shannon (information duality) · Robertson (BM25/TF-IDF fusion)

## 4.1 — `core/memory/hybrid_query.py`

### Requirements
- Fuse 5 scoring signals: vector, keyword, recency, importance, graph
- Weights configurable via `QueryWeights` (default: 0.40 + 0.15 + 0.20 + 0.15 + 0.10 = 1.0)
- Each signal normalized to [0, 1] before fusion
- Quality gate: filter results below min_ihsan before returning
- The scoring is a monoid — `FusableScore` with associative combine

### Pseudocode

```
IMPORT time, math, logging FROM stdlib
IMPORT numpy as np
IMPORT List, Optional, Dict, Set FROM typing
IMPORT MemoryRecord, SearchResult, QueryOptions FROM core.memory.types
IMPORT QueryWeights, QualityConfig FROM core.memory.config
IMPORT HNSWVectorIndex FROM core.memory.hnsw_index
IMPORT UnifiedSQLiteStore FROM core.memory.unified_store

logger = GET_LOGGER(__name__)

CLASS HybridQueryEngine:
    """
    Multi-signal score fusion for memory retrieval.

    Combines vector similarity, keyword relevance, temporal recency,
    importance weight, and graph connectivity into a single composite score.

    The 5-factor scoring forms a monoid: each factor maps to [0,1],
    the weighted sum preserves associativity, and identity is the
    zero-weight vector.
    """

    FUNCTION __init__(
        vector_index: HNSWVectorIndex,
        store: UnifiedSQLiteStore,
        weights: QueryWeights = QueryWeights(),
        quality: QualityConfig = QualityConfig(),
    ):
        self._vector_index = vector_index
        self._store = store
        self._weights = weights
        self._quality = quality

    FUNCTION search(
        query_text: Optional[str] = None,
        query_embedding: Optional[np.ndarray] = None,
        options: QueryOptions = QueryOptions(),
    ) -> List[SearchResult]:
        """
        Execute a hybrid search across all memory signals.

        Flow:
        1. Collect candidate IDs from vector + keyword searches
        2. Load full records for candidates
        3. Compute per-signal scores for each candidate
        4. Fuse scores using weighted sum
        5. Apply quality gates (min_ihsan, min_score)
        6. Sort by composite score, return top_k
        """

        candidates: Dict[str, Dict[str, float]] = {}
        # Dict of id -> {vector_score, keyword_score}

        # ── Signal 1: Vector Search (weight: 0.40) ──
        IF query_embedding IS NOT None AND len(self._vector_index) > 0:
            # Fetch 3x top_k to have enough candidates after fusion
            vector_results = self._vector_index.search(
                query_embedding,
                top_k=options.top_k * 3,
            )
            FOR id, score IN vector_results:
                candidates.setdefault(id, {})["vector_score"] = max(0, score)

        # ── Signal 2: Keyword Search (weight: 0.15) ──
        IF query_text IS NOT None AND len(query_text.strip()) > 0:
            keyword_results = self._store.keyword_search(
                query_text,
                top_k=options.top_k * 3,
            )
            FOR id, score IN keyword_results:
                candidates.setdefault(id, {})["keyword_score"] = max(0, score)

        IF NOT candidates:
            RETURN []

        # ── Load full records for all candidates ──
        results: List[SearchResult] = []
        now = datetime.now(timezone.utc)

        FOR id, signal_scores IN candidates.items():
            record = self._store.get(id)
            IF record IS None:
                CONTINUE

            # Apply type/state filters
            IF options.memory_types AND record.memory_type NOT IN options.memory_types:
                CONTINUE
            IF options.states AND record.state NOT IN options.states:
                CONTINUE
            IF options.source_filter AND record.adapter_source != options.source_filter:
                CONTINUE

            # ── Signal 3: Recency Score (weight: 0.20) ──
            age_hours = (now - record.last_accessed).total_seconds() / 3600
            recency_score = self._recency_decay(age_hours)

            # ── Signal 4: Importance Score (weight: 0.15) ──
            importance_score = min(1.0, record.importance)

            # ── Signal 5: Graph Score (weight: 0.10) ──
            graph_score = self._graph_connectivity(record)

            # ── Fuse all signals ──
            vector_s = signal_scores.get("vector_score", 0.0)
            keyword_s = signal_scores.get("keyword_score", 0.0)

            composite = (
                self._weights.vector * vector_s
                + self._weights.keyword * keyword_s
                + self._weights.recency * recency_score
                + self._weights.importance * importance_score
                + self._weights.graph * graph_score
            )

            # Quality gate
            IF record.ihsan_score < options.min_ihsan:
                CONTINUE
            IF composite < options.min_score:
                CONTINUE

            result = SearchResult(
                record=record IF options.include_embeddings ELSE record._without_embedding(),
                score=composite,
                vector_score=vector_s,
                keyword_score=keyword_s,
                recency_score=recency_score,
                importance_score=importance_score,
                graph_score=graph_score,
            )
            results.append(result)

        # Sort by composite score descending, take top_k
        results.sort(key=LAMBDA r: r.score, reverse=True)
        RETURN results[:options.top_k]

    # ── Signal Computation ──

    FUNCTION _recency_decay(age_hours: float) -> float:
        """Exponential decay: score = exp(-lambda * age).
        Half-life = 168 hours (1 week). lambda = ln(2) / 168."""
        LAMBDA = 0.00413  # ln(2) / 168
        RETURN math.exp(-LAMBDA * age_hours)

    FUNCTION _graph_connectivity(record: MemoryRecord) -> float:
        """Score based on graph connections.
        More related_ids + having a parent = higher score.
        Normalized to [0, 1] with diminishing returns."""
        connections = len(record.related_ids) + (1 IF record.parent_id ELSE 0)
        # Sigmoid-like saturation: score = 1 - 1/(1 + connections/3)
        RETURN 1.0 - 1.0 / (1.0 + connections / 3.0)
```

### TDD Anchors

```
TEST test_hybrid_search_vector_only():
    # Setup: index with 3 vectors, no keyword match
    engine = HybridQueryEngine(vector_index, store, QueryWeights())
    results = engine.search(
        query_embedding=np.array([1, 0, 0, ...]),
        options=QueryOptions(top_k=2),
    )
    ASSERT len(results) <= 2
    ASSERT results[0].score >= results[1].score  # Sorted

TEST test_hybrid_search_keyword_only():
    engine = HybridQueryEngine(vector_index, store, QueryWeights())
    results = engine.search(query_text="quantum", options=QueryOptions(top_k=5))
    ASSERT all(r.keyword_score > 0 FOR r IN results)

TEST test_hybrid_search_both_signals():
    engine = HybridQueryEngine(vector_index, store, QueryWeights())
    results = engine.search(
        query_text="quantum",
        query_embedding=embedding,
        options=QueryOptions(top_k=5),
    )
    # Results should have both vector and keyword scores
    ASSERT any(r.vector_score > 0 AND r.keyword_score > 0 FOR r IN results)

TEST test_quality_gate_filters():
    engine = HybridQueryEngine(vector_index, store, QueryWeights())
    results = engine.search(
        query_embedding=embedding,
        options=QueryOptions(min_ihsan=0.95),
    )
    ASSERT all(r.record.ihsan_score >= 0.95 FOR r IN results)

TEST test_recency_decay():
    engine = HybridQueryEngine(vector_index, store, QueryWeights())
    fresh = engine._recency_decay(0)       # Just accessed
    old = engine._recency_decay(168)       # 1 week ago
    ancient = engine._recency_decay(720)   # 1 month ago
    ASSERT fresh > old > ancient
    ASSERT abs(fresh - 1.0) < 0.01
    ASSERT abs(old - 0.5) < 0.05  # Half-life at 168h

TEST test_graph_connectivity_score():
    engine = HybridQueryEngine(vector_index, store, QueryWeights())
    r0 = MemoryRecord(id="a", content="", memory_type=MemoryType.SEMANTIC, related_ids=set())
    r3 = MemoryRecord(id="b", content="", memory_type=MemoryType.SEMANTIC, related_ids={"x","y","z"})
    ASSERT engine._graph_connectivity(r0) < engine._graph_connectivity(r3)
    ASSERT 0 <= engine._graph_connectivity(r0) <= 1
    ASSERT 0 <= engine._graph_connectivity(r3) <= 1

TEST test_weights_affect_ranking():
    # With vector weight = 1.0, vector-close items rank first
    engine_v = HybridQueryEngine(vector_index, store, QueryWeights(vector=1, keyword=0, recency=0, importance=0, graph=0))
    # With keyword weight = 1.0, keyword-matched items rank first
    engine_k = HybridQueryEngine(vector_index, store, QueryWeights(vector=0, keyword=1, recency=0, importance=0, graph=0))
    # Rankings should differ
    res_v = engine_v.search(query_text="quantum", query_embedding=embedding)
    res_k = engine_k.search(query_text="quantum", query_embedding=embedding)
    # At least top-1 should differ (unless same doc is best in both)
```

### Edge Cases

1. **No query_text and no query_embedding** — return empty list
2. **All candidates filtered by quality gate** — return empty list
3. **FTS5 query with special chars** — escape or wrap in quotes
4. **Record in index but not in store** — skip (HNSW stale entry)
5. **Record with no embedding** — vector_score = 0.0 for that record
6. **Division by zero in rank normalization** — epsilon guard in keyword_search
