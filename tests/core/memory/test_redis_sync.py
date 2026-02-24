"""Tests for FR-05: Cross-Agent Memory Sync via Redis."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from core.memory.agent_db import AgentDB
from core.memory.config import MemoryConfig
from core.memory.sync import MemorySyncPublisher, MemorySyncSubscriber
from core.memory.types import MemoryKind, MemoryRecord, RecordState


@pytest.fixture
def tmp_config(tmp_path: Path) -> MemoryConfig:
    cfg = MemoryConfig(data_dir=tmp_path / "agent_db")
    cfg.auto_embed = False
    return cfg


@pytest.fixture
def db(tmp_config: MemoryConfig) -> AgentDB:
    d = AgentDB(tmp_config)
    d.initialize()
    return d


def _make_record(
    record_id: str = "test123",
    content: str = "shared knowledge",
    kind: MemoryKind = MemoryKind.SEMANTIC,
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=record_id,
        content=content,
        kind=kind,
        state=RecordState.ACTIVE,
        importance=0.5,
        source="test",
        created_at=now,
        updated_at=now,
        last_accessed=now,
    )


class TestPublisher:
    def test_buffer_when_offline(self):
        pub = MemorySyncPublisher("agent_a", redis_url="redis://127.0.0.1:99999")
        assert not pub.connected

        record = _make_record()
        asyncio.get_event_loop().run_until_complete(pub.publish(record))

        assert pub.buffer_size == 1

    def test_buffer_max_size(self):
        pub = MemorySyncPublisher("agent_a", redis_url="redis://127.0.0.1:99999")
        pub._max_buffer = 3

        for i in range(5):
            record = _make_record(record_id=f"r{i}", content=f"item {i}")
            asyncio.get_event_loop().run_until_complete(pub.publish(record))

        assert pub.buffer_size == 3

    def test_stats(self):
        pub = MemorySyncPublisher("agent_a")
        stats = pub.stats()
        assert stats["connected"] is False
        assert stats["published"] == 0
        assert "channel" in stats


class TestSubscriberMessageHandling:
    def test_import_from_other_agent(self, db):
        sub = MemorySyncSubscriber(db, agent_id="agent_b")
        record = _make_record()
        message = json.dumps({
            "sender_id": "agent_a",
            "record": record.to_dict(),
        })

        asyncio.get_event_loop().run_until_complete(sub._handle_message(message))

        assert sub.imported_count == 1
        assert db.retrieve("test123") is not None

    def test_filter_own_messages(self, db):
        sub = MemorySyncSubscriber(db, agent_id="agent_b")
        record = _make_record()
        message = json.dumps({
            "sender_id": "agent_b",  # Same as subscriber
            "record": record.to_dict(),
        })

        asyncio.get_event_loop().run_until_complete(sub._handle_message(message))

        assert sub.imported_count == 0

    def test_dedup_by_id(self, db):
        sub = MemorySyncSubscriber(db, agent_id="agent_b")
        record = _make_record()
        message = json.dumps({
            "sender_id": "agent_a",
            "record": record.to_dict(),
        })

        asyncio.get_event_loop().run_until_complete(sub._handle_message(message))
        asyncio.get_event_loop().run_until_complete(sub._handle_message(message))

        assert sub.imported_count == 1
        assert sub.skipped_count == 1

    def test_kind_filter_rejects(self, db):
        sub = MemorySyncSubscriber(
            db, agent_id="agent_b", accept_kinds=[MemoryKind.SEMANTIC]
        )
        record = _make_record(kind=MemoryKind.EPISODIC)
        message = json.dumps({
            "sender_id": "agent_a",
            "record": record.to_dict(),
        })

        asyncio.get_event_loop().run_until_complete(sub._handle_message(message))

        assert sub.skipped_count == 1
        assert sub.imported_count == 0

    def test_kind_filter_accepts(self, db):
        sub = MemorySyncSubscriber(
            db, agent_id="agent_b", accept_kinds=[MemoryKind.SEMANTIC]
        )
        record = _make_record(kind=MemoryKind.SEMANTIC)
        message = json.dumps({
            "sender_id": "agent_a",
            "record": record.to_dict(),
        })

        asyncio.get_event_loop().run_until_complete(sub._handle_message(message))

        assert sub.imported_count == 1

    def test_invalid_json_handled(self, db):
        sub = MemorySyncSubscriber(db, agent_id="agent_b")
        asyncio.get_event_loop().run_until_complete(
            sub._handle_message("not valid json")
        )
        # Should not crash, just skip
        assert sub.imported_count == 0

    def test_synced_tag_added(self, db):
        sub = MemorySyncSubscriber(db, agent_id="agent_b")
        record = _make_record()
        message = json.dumps({
            "sender_id": "agent_a",
            "record": record.to_dict(),
        })

        asyncio.get_event_loop().run_until_complete(sub._handle_message(message))

        stored = db.retrieve("test123")
        assert stored is not None
        assert "synced" in stored.tags


class TestSubscriberStats:
    def test_stats_structure(self, db):
        sub = MemorySyncSubscriber(db, agent_id="agent_b")
        stats = sub.stats()
        assert stats["running"] is False
        assert stats["imported"] == 0
        assert stats["skipped"] == 0
        assert "channel" in stats
