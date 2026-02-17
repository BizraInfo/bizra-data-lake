"""Tests for HybridQueryEngine score fusion."""

from __future__ import annotations

import pytest

from core.memory.config import MemoryConfig
from core.memory.hnsw_index import HNSWIndex
from core.memory.hybrid_query import HybridQueryEngine
from core.memory.types import MemoryKind, QueryOptions
from core.memory.unified_store import UnifiedStore

from .conftest import make_record, random_embedding


@pytest.fixture
def engine(memory_config):
    store = UnifiedStore(memory_config)
    store.initialize()
    hnsw = HNSWIndex(memory_config.hnsw)
    hnsw.initialize()
    engine = HybridQueryEngine(store, hnsw, memory_config)
    yield engine, store, hnsw
    store.close()


class TestHybridSearch:
    def test_keyword_only_search(self, engine):
        eng, store, hnsw = engine
        store.upsert(make_record("r1", content="quantum computing entanglement"))
        store.upsert(make_record("r2", content="classical music beethoven"))

        options = QueryOptions(query_text="quantum entanglement", top_k=5)
        results = eng.search(options)
        assert len(results) > 0
        assert results[0].record.id == "r1"
        assert results[0].keyword_score > 0

    def test_vector_only_search(self, engine):
        eng, store, hnsw = engine
        vec = random_embedding(8)
        store.upsert(make_record("r1", embedding=vec))
        hnsw.add("r1", vec)

        noise_vec = random_embedding(8)
        store.upsert(make_record("r2", content="noise", embedding=noise_vec))
        hnsw.add("r2", noise_vec)

        options = QueryOptions(query_embedding=vec, top_k=5)
        results = eng.search(options)
        assert len(results) > 0
        assert results[0].record.id == "r1"
        assert results[0].vector_score > 0.9

    def test_hybrid_search(self, engine):
        eng, store, hnsw = engine
        vec = random_embedding(8)
        store.upsert(
            make_record("r1", content="machine learning neural networks", embedding=vec)
        )
        hnsw.add("r1", vec)

        options = QueryOptions(
            query_text="machine learning",
            query_embedding=vec,
            top_k=5,
        )
        results = eng.search(options)
        assert len(results) > 0
        # Both signals should contribute
        r = results[0]
        assert r.vector_score > 0
        assert r.keyword_score > 0

    def test_min_score_filter(self, engine):
        eng, store, hnsw = engine
        store.upsert(make_record("r1", content="hello world"))
        options = QueryOptions(query_text="xyznonexistent", top_k=5, min_score=0.5)
        results = eng.search(options)
        assert len(results) == 0

    def test_kind_filter(self, engine):
        eng, store, hnsw = engine
        store.upsert(make_record("r1", content="episodic memory test", kind=MemoryKind.EPISODIC))
        store.upsert(make_record("r2", content="episodic semantic test", kind=MemoryKind.SEMANTIC))

        options = QueryOptions(
            query_text="episodic test",
            top_k=5,
            kinds=[MemoryKind.EPISODIC],
        )
        results = eng.search(options)
        for r in results:
            assert r.record.kind == MemoryKind.EPISODIC

    def test_empty_query_returns_empty(self, engine):
        eng, store, hnsw = engine
        store.upsert(make_record("r1", content="test content"))
        options = QueryOptions(top_k=5)
        results = eng.search(options)
        assert results == []

    def test_score_components_sum_correctly(self, engine):
        eng, store, hnsw = engine
        vec = random_embedding(8)
        store.upsert(make_record("r1", content="test data", embedding=vec, importance=0.8))
        hnsw.add("r1", vec)

        options = QueryOptions(query_text="test data", query_embedding=vec, top_k=5)
        results = eng.search(options)
        if results:
            r = results[0]
            cfg = eng._config
            expected = (
                cfg.weight_vector * r.vector_score
                + cfg.weight_keyword * r.keyword_score
                + cfg.weight_recency * r.recency_score
                + cfg.weight_importance * r.importance_score
                + cfg.weight_graph * r.graph_score
            )
            assert abs(r.score - expected) < 0.01
