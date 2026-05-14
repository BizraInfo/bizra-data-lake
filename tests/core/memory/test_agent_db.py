"""Tests for AgentDB facade."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.memory.agent_db import AgentDB
from core.memory.types import MemoryKind, MemoryRecord, RecordState

from .conftest import random_embedding


@pytest.fixture
def db(memory_config):
    agent_db = AgentDB(memory_config)
    agent_db.initialize()
    yield agent_db


class TestAgentDBStore:
    def test_store_creates_record(self, db):
        rec = db.store("Test content for storage")
        assert rec.id is not None
        assert rec.content == "Test content for storage"
        assert rec.kind == MemoryKind.SEMANTIC

    def test_store_with_embedding(self, db):
        emb = random_embedding(8)
        rec = db.store("Content with embedding", embedding=emb)
        assert rec.embedding is not None
        assert len(rec.embedding) == 8

    def test_store_deduplicates(self, db):
        db.store("Same content")
        db.store("Same content")
        # Content-addressable: same content -> same ID -> upsert
        assert db.count == 1

    def test_store_with_tags(self, db):
        rec = db.store("Tagged content", tags=["test", "memory"])
        assert rec.tags == ["test", "memory"]

    def test_store_with_kind(self, db):
        rec = db.store("Episode content", kind=MemoryKind.EPISODIC)
        assert rec.kind == MemoryKind.EPISODIC

    def test_store_drops_mismatched_auto_embedding(self, db):
        db.set_embedding_fn(lambda _text: [0.1, 0.2, 0.3])
        rec = db.store("Auto-embed mismatch should not fail")
        assert rec.embedding is None
        assert db.count == 1


class TestAgentDBSearch:
    def test_search_by_keyword(self, db):
        db.store("The quantum computer processes qubits")
        db.store("Classical music by Beethoven")
        results = db.search("quantum computer")
        assert len(results) > 0
        assert "quantum" in results[0].record.content.lower()

    def test_search_by_embedding(self, db):
        vec = random_embedding(8)
        db.store("Vector-indexed content", embedding=vec)
        results = db.search(query_embedding=vec, top_k=5)
        assert len(results) > 0

    def test_search_top_k(self, db):
        for i in range(10):
            db.store(f"Test content number {i} unique words set{i}")
        results = db.search("content", top_k=3)
        assert len(results) <= 3

    def test_search_empty_db(self, db):
        results = db.search("anything")
        assert results == []

    def test_search_with_source_filter(self, db):
        db.store("From agent A", source="agent_a")
        db.store("From agent B", source="agent_b")
        results = db.search("agent", source="agent_a")
        for r in results:
            assert r.record.source == "agent_a"

    def test_search_include_archived(self, db):
        now = datetime.now(timezone.utc)
        db.store_record(
            MemoryRecord(
                id="archived_001",
                content="Archived retrieval candidate",
                kind=MemoryKind.SEMANTIC,
                state=RecordState.ARCHIVED,
                source="test",
                tags=["archive"],
                created_at=now,
                updated_at=now,
                last_accessed=now,
            )
        )

        assert db.search("Archived retrieval", include_archived=False) == []
        results = db.search("Archived retrieval", include_archived=True)
        assert len(results) == 1
        assert results[0].record.state == RecordState.ARCHIVED

    def test_search_falls_back_when_auto_query_embedding_mismatched(self, db):
        db.store("Keyword fallback search still works")
        db.set_embedding_fn(lambda _text: [0.1, 0.2, 0.3])
        results = db.search("keyword fallback")
        assert len(results) >= 1
        assert "fallback" in results[0].record.content.lower()

    def test_search_cache_key_separates_query_embeddings(self, db):
        db.store(
            "alpha vector memory", embedding=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        db.store(
            "beta vector memory", embedding=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )

        first = db.search(
            query_embedding=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            top_k=1,
            min_score=0.0,
        )
        second = db.search(
            query_embedding=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            top_k=1,
            min_score=0.0,
        )

        assert first[0].record.content == "alpha vector memory"
        assert second[0].record.content == "beta vector memory"

    def test_search_exposes_mmr_diversification(self, db):
        query_embedding = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        db.store("similar memory one", embedding=query_embedding)
        db.store(
            "similar memory two", embedding=[0.99, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        db.store("diverse memory", embedding=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        standard = db.search(
            query_embedding=query_embedding,
            top_k=3,
            min_score=0.0,
        )
        diversified = db.search(
            query_embedding=query_embedding,
            top_k=3,
            min_score=0.0,
            use_mmr=True,
            mmr_lambda=0.2,
        )

        standard_contents = [result.record.content for result in standard]
        diversified_contents = [result.record.content for result in diversified]

        assert standard_contents[:2] == ["similar memory one", "similar memory two"]
        assert diversified_contents[0] == "similar memory one"
        assert "diverse memory" in diversified_contents[:2]

    def test_search_validates_mmr_lambda(self, db):
        with pytest.raises(ValueError, match="mmr_lambda"):
            db.search("anything", use_mmr=True, mmr_lambda=1.5)

    def test_search_applies_metadata_filters(self, db):
        db.store("research paper alpha", metadata={"category": "ml"})
        db.store("research paper beta", metadata={"category": "cv"})

        results = db.search(
            "research paper",
            top_k=5,
            min_score=0.0,
            metadata_filters={"category": "ml"},
        )

        assert len(results) == 1
        assert results[0].record.metadata["category"] == "ml"


class TestAgentDBRetrieve:
    def test_retrieve_by_id(self, db):
        rec = db.store("Retrievable content")
        loaded = db.retrieve(rec.id)
        assert loaded is not None
        assert loaded.content == "Retrievable content"

    def test_retrieve_nonexistent(self, db):
        assert db.retrieve("nonexistent") is None

    def test_retrieve_updates_access(self, db):
        rec = db.store("Access tracked content")
        db.retrieve(rec.id)
        loaded = db.retrieve(rec.id)
        assert loaded is not None
        assert loaded.access_count >= 1


class TestAgentDBForget:
    def test_forget_soft(self, db):
        rec = db.store("Forgettable content")
        db.forget(rec.id)
        # Soft delete: record still exists but state is DELETED
        loaded = db.backend.get(rec.id)
        assert loaded is not None
        assert loaded.state == RecordState.DELETED

    def test_forget_hard(self, db):
        rec = db.store("Hard delete content")
        db.forget(rec.id, hard=True)
        assert db.backend.get(rec.id) is None


class TestAgentDBPersistence:
    def test_save_and_reload(self, memory_config):
        # Store data
        db1 = AgentDB(memory_config)
        db1.initialize()
        db1.store("Persistent content", embedding=random_embedding(8))
        db1.save()

        # Reload from same path
        db2 = AgentDB(memory_config)
        db2.initialize()
        assert db2.count == 1
        assert db2.hnsw.count == 1

    def test_stats(self, db):
        db.store("Test content")
        stats = db.stats()
        assert stats["total_records"] == 1
        assert stats["active_records"] == 1
        assert "fts_row_count" in stats
        assert stats["indexed_vectors"] >= 0

    def test_rebuild_indexes_repairs_stale_fts(self, db):
        db.store("repair me", embedding=random_embedding(8))
        conn = db.backend._ensure_conn()
        conn.execute("DELETE FROM records_fts")
        conn.commit()

        stale_stats = db.stats()
        assert stale_stats["index_health"]["status"] == "stale"

        result = db.rebuild_indexes()
        repaired_stats = db.stats()

        assert result["fts_rows"] == repaired_stats["fts_row_count"]
        assert repaired_stats["index_health"]["status"] == "healthy"
        assert repaired_stats["last_rebuild_at"] is not None
        assert result["duration_ms"] >= 0

    def test_not_initialized_raises(self, memory_config):
        db = AgentDB(memory_config)
        with pytest.raises(RuntimeError, match="not initialized"):
            db.store("Should fail")
