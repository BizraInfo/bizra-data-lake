"""Tests for HybridSearchEngine — Reciprocal Rank Fusion.

Tests the fusion logic without requiring FAISS/RuVector infrastructure.
"""

from __future__ import annotations

import uuid

from core.memory.types import MemoryKind, MemoryRecord, SearchResult
from core.search.hybrid_search import HybridSearchEngine, _rrf_score


class MockEngine:
    """Mock search engine returning pre-configured results."""

    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results

    def search(
        self, query: str, top_k: int = 5, min_score: float = 0.0
    ) -> list[SearchResult]:
        return self._results[:top_k]

    def search_by_vector(
        self, vector: object, top_k: int = 5, min_score: float = 0.0
    ) -> list[SearchResult]:
        return self._results[:top_k]


def _make_result(content: str, source_id: str, score: float) -> SearchResult:
    return SearchResult(
        record=MemoryRecord(
            id=str(uuid.uuid4()),
            content=content,
            kind=MemoryKind.SEMANTIC,
            source="test",
            source_id=source_id,
            metadata={"engine": "test"},
        ),
        score=score,
        vector_score=score,
    )


def test_rrf_score_formula():
    """RRF(rank=1, k=60) = 1/61, RRF(rank=2, k=60) = 1/62."""
    assert abs(_rrf_score(1, 60) - 1 / 61) < 1e-10
    assert abs(_rrf_score(2, 60) - 1 / 62) < 1e-10
    # Monotonically decreasing
    assert _rrf_score(1) > _rrf_score(2) > _rrf_score(10)


def test_single_engine_passthrough():
    """With only one engine, results pass through without fusion."""
    r1 = _make_result("alpha", "a1", 0.9)
    r2 = _make_result("beta", "b1", 0.7)
    engine = HybridSearchEngine(faiss_engine=MockEngine([r1, r2]))
    engine._initialized = True

    results = engine.search("test", top_k=5)
    assert len(results) == 2
    assert results[0].record.content == "alpha"


def test_rrf_fusion_boosts_shared_results():
    """Results appearing in both engines get higher RRF scores."""
    # Both engines return "shared" at rank 1
    shared = _make_result("shared doc", "s1", 0.9)
    faiss_only = _make_result("faiss only", "f1", 0.8)
    rv_only = _make_result("ruvector only", "r1", 0.85)

    faiss = MockEngine([shared, faiss_only])
    rv = MockEngine([shared, rv_only])

    engine = HybridSearchEngine(faiss_engine=faiss, ruvector_engine=rv)
    engine._initialized = True

    results = engine.search("test", top_k=5, min_score=0.0)
    # "shared doc" should be rank 1 (appears in both → double RRF)
    assert results[0].record.content == "shared doc"
    assert results[0].score == 1.0  # Normalized max
    # Other two should follow
    assert len(results) == 3


def test_dedup_by_source_id():
    """Same source_id from different engines is deduplicated."""
    r1 = _make_result("doc A version 1", "id_A", 0.9)
    r2 = _make_result("doc A version 2", "id_A", 0.85)

    engine = HybridSearchEngine(
        faiss_engine=MockEngine([r1]),
        ruvector_engine=MockEngine([r2]),
    )
    engine._initialized = True

    results = engine.search("test", top_k=5, min_score=0.0)
    # Should deduplicate to 1 result
    assert len(results) == 1
    # RRF score should be sum of both ranks
    assert results[0].record.metadata["rrf_score"] > _rrf_score(1)


def test_empty_engines():
    """No engines available returns empty list."""
    engine = HybridSearchEngine()
    engine._initialized = True
    assert engine.search("test") == []


def test_one_engine_fails_gracefully():
    """If one engine returns empty, other engine's results still come through."""
    r1 = _make_result("survivor", "s1", 0.9)
    engine = HybridSearchEngine(
        faiss_engine=MockEngine([r1]),
        ruvector_engine=MockEngine([]),
    )
    engine._initialized = True

    results = engine.search("test", top_k=5, min_score=0.0)
    assert len(results) == 1
    assert results[0].record.content == "survivor"


def test_top_k_limit():
    """Results are limited to top_k."""
    many = [_make_result(f"doc {i}", f"id_{i}", 0.9 - i * 0.01) for i in range(20)]
    engine = HybridSearchEngine(faiss_engine=MockEngine(many))
    engine._initialized = True

    results = engine.search("test", top_k=3, min_score=0.0)
    assert len(results) == 3


def test_search_by_vector():
    """search_by_vector uses same fusion logic."""
    r1 = _make_result("vec result", "v1", 0.95)
    engine = HybridSearchEngine(
        faiss_engine=MockEngine([r1]),
        ruvector_engine=MockEngine([r1]),
    )
    engine._initialized = True

    results = engine.search_by_vector([0.1] * 384, top_k=5, min_score=0.0)
    assert len(results) == 1
    assert results[0].record.metadata["engine"] == "hybrid_rrf"
