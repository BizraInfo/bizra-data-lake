"""Tests for AgentDB facade."""

from __future__ import annotations

import pytest

from core.memory.agent_db import AgentDB
from core.memory.config import MemoryConfig
from core.memory.types import MemoryKind, RecordState

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
        assert stats["indexed_vectors"] >= 0

    def test_not_initialized_raises(self, memory_config):
        db = AgentDB(memory_config)
        with pytest.raises(RuntimeError, match="not initialized"):
            db.store("Should fail")
