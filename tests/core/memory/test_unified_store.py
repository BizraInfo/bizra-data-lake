"""Tests for UnifiedStore (SQLite v2 + FTS5)."""

from __future__ import annotations

import pytest

from core.memory.types import MemoryKind, RecordState
from core.memory.unified_store import UnifiedStore

from .conftest import make_record, random_embedding


@pytest.fixture
def store(memory_config):
    s = UnifiedStore(memory_config)
    s.initialize()
    yield s
    s.close()


class TestStoreBasic:
    def test_init_creates_db(self, store, memory_config):
        assert memory_config.sqlite_path.exists()

    def test_upsert_and_get(self, store):
        rec = make_record("r1")
        store.upsert(rec)
        loaded = store.get("r1")
        assert loaded is not None
        assert loaded.content == rec.content
        assert loaded.kind == MemoryKind.SEMANTIC

    def test_upsert_with_embedding(self, store):
        rec = make_record("r1", embedding=random_embedding(8))
        store.upsert(rec)
        loaded = store.get("r1")
        assert loaded is not None
        assert loaded.embedding is not None
        assert len(loaded.embedding) == 8

    def test_get_nonexistent(self, store):
        assert store.get("nonexistent") is None

    def test_count(self, store):
        assert store.count() == 0
        store.upsert(make_record("r1"))
        store.upsert(make_record("r2", content="different content"))
        assert store.count() == 2

    def test_count_by_state(self, store):
        store.upsert(make_record("r1"))
        assert store.count(state=RecordState.ACTIVE) == 1
        assert store.count(state=RecordState.ARCHIVED) == 0


class TestStoreDelete:
    def test_soft_delete(self, store):
        store.upsert(make_record("r1"))
        store.delete("r1", hard=False)
        rec = store.get("r1")
        assert rec is not None
        assert rec.state == RecordState.DELETED

    def test_hard_delete(self, store):
        store.upsert(make_record("r1"))
        store.delete("r1", hard=True)
        assert store.get("r1") is None

    def test_list_ids(self, store):
        store.upsert(make_record("r1"))
        store.upsert(make_record("r2", content="second"))
        ids = store.list_ids()
        assert set(ids) == {"r1", "r2"}

    def test_list_ids_filtered(self, store):
        store.upsert(make_record("r1", kind=MemoryKind.EPISODIC))
        store.upsert(make_record("r2", content="second", kind=MemoryKind.SEMANTIC))
        ids = store.list_ids(kind=MemoryKind.EPISODIC)
        assert ids == ["r1"]


class TestStoreBatch:
    def test_upsert_batch(self, store):
        records = [make_record(f"r{i}", content=f"content {i}") for i in range(10)]
        count = store.upsert_batch(records)
        assert count == 10
        assert store.count() == 10


class TestStoreFTS:
    def test_keyword_search(self, store):
        store.upsert(make_record("r1", content="The quick brown fox jumps"))
        store.upsert(make_record("r2", content="The lazy dog sleeps"))
        store.upsert(make_record("r3", content="Python programming language"))

        results = store.keyword_search("fox jumps", top_k=5)
        assert len(results) > 0
        assert results[0][0] == "r1"

    def test_keyword_search_no_match(self, store):
        store.upsert(make_record("r1", content="Hello world"))
        results = store.keyword_search("xyznonexistent", top_k=5)
        assert len(results) == 0

    def test_keyword_search_returns_scores(self, store):
        store.upsert(make_record("r1", content="machine learning neural network"))
        results = store.keyword_search("machine learning", top_k=5)
        assert len(results) > 0
        _, score = results[0]
        assert 0.0 <= score <= 1.0

    def test_keyword_search_sanitizes_hyphenated_queries(self, store):
        store.upsert(make_record("r1", content="Archive-only memory surface"))

        results = store.keyword_search("Archive-only memory", top_k=5)

        assert len(results) > 0
        assert results[0][0] == "r1"


class TestStoreAccessTracking:
    def test_touch_access(self, store):
        store.upsert(make_record("r1"))
        store.touch_access("r1")
        rec = store.get("r1")
        assert rec is not None
        assert rec.access_count == 1


class TestStoreLoadAll:
    def test_load_all_active(self, store):
        store.upsert(make_record("r1"))
        store.upsert(make_record("r2", content="second"))
        store.delete("r2", hard=False)  # soft delete
        active = store.load_all_active()
        assert len(active) == 1
        assert active[0].id == "r1"

    def test_load_with_embeddings(self, store):
        store.upsert(make_record("r1", embedding=random_embedding(8)))
        store.upsert(make_record("r2", content="no embedding"))
        rows = store.load_with_embeddings()
        assert len(rows) == 1
        assert rows[0][0] == "r1"
