"""Tests for memory adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Set

import numpy as np

from core.memory.adapters.experience_ledger import ExperienceLedgerAdapter
from core.memory.adapters.living_memory import LivingMemoryAdapter
from core.memory.adapters.pattern_memory import PatternMemoryAdapter
from core.memory.types import MemoryKind

# ── Mock LivingMemory ────────────────────────────────────────────────────


class MockMemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"
    PROSPECTIVE = "prospective"


class MockMemoryState(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


@dataclass
class MockMemoryEntry:
    id: str
    content: str
    memory_type: MockMemoryType = MockMemoryType.SEMANTIC
    embedding: Optional[np.ndarray] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    ihsan_score: float = 0.95
    snr_score: float = 0.90
    confidence: float = 0.85
    state: MockMemoryState = MockMemoryState.ACTIVE
    source: str = "test"
    related_ids: Set[str] = field(default_factory=set)
    parent_id: Optional[str] = None
    importance: float = 0.7
    emotional_weight: float = 0.5


class MockLivingMemoryCore:
    def __init__(self):
        self._memories = {}

    def add(self, entry: MockMemoryEntry):
        self._memories[entry.id] = entry


# ── Mock SEL ─────────────────────────────────────────────────────────────


@dataclass
class MockEpisode:
    content_hash: str
    query_text: str
    response_text: str
    snr_score: float = 0.9
    ihsan_score: float = 0.95
    importance: float = 0.7
    timestamp_secs: int = 1700000000
    prev_hash: str = "genesis"
    sequence_number: int = 0
    verdict: str = "SNR_OK"
    embedding: Optional[list] = None


class MockSEL:
    def __init__(self):
        self._episodes = []

    def add(self, episode: MockEpisode):
        self._episodes.append(episode)


# ── Tests ────────────────────────────────────────────────────────────────


class TestLivingMemoryAdapter:
    def test_export_all(self):
        lm = MockLivingMemoryCore()
        lm.add(MockMemoryEntry(id="m1", content="First memory"))
        lm.add(MockMemoryEntry(id="m2", content="Second memory"))

        adapter = LivingMemoryAdapter(lm)
        records = adapter.export_all()
        assert len(records) == 2

    def test_preserves_content(self):
        lm = MockLivingMemoryCore()
        lm.add(MockMemoryEntry(id="m1", content="Important fact"))
        adapter = LivingMemoryAdapter(lm)
        records = adapter.export_all()
        assert records[0].content == "Important fact"

    def test_maps_memory_type(self):
        lm = MockLivingMemoryCore()
        lm.add(
            MockMemoryEntry(
                id="m1", content="Episode", memory_type=MockMemoryType.EPISODIC
            )
        )
        adapter = LivingMemoryAdapter(lm)
        records = adapter.export_all()
        assert records[0].kind == MemoryKind.EPISODIC

    def test_maps_scores(self):
        lm = MockLivingMemoryCore()
        lm.add(
            MockMemoryEntry(id="m1", content="Scored", ihsan_score=0.88, snr_score=0.75)
        )
        adapter = LivingMemoryAdapter(lm)
        records = adapter.export_all()
        assert records[0].ihsan_score == 0.88
        assert records[0].snr_score == 0.75

    def test_converts_embedding(self):
        lm = MockLivingMemoryCore()
        lm.add(
            MockMemoryEntry(
                id="m1",
                content="With embedding",
                embedding=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            )
        )
        adapter = LivingMemoryAdapter(lm)
        records = adapter.export_all()
        assert records[0].embedding is not None
        assert len(records[0].embedding) == 3

    def test_skips_deleted(self):
        lm = MockLivingMemoryCore()
        lm.add(MockMemoryEntry(id="m1", content="Active"))
        lm.add(
            MockMemoryEntry(id="m2", content="Deleted", state=MockMemoryState.DELETED)
        )
        adapter = LivingMemoryAdapter(lm)
        records = adapter.export_all()
        assert len(records) == 1

    def test_metadata_includes_origin(self):
        lm = MockLivingMemoryCore()
        lm.add(MockMemoryEntry(id="m1", content="Meta test"))
        adapter = LivingMemoryAdapter(lm)
        records = adapter.export_all()
        assert records[0].metadata.get("origin") == "living_memory"


class TestExperienceLedgerAdapter:
    def test_export_all(self):
        sel = MockSEL()
        sel.add(
            MockEpisode(
                content_hash="abc123",
                query_text="What is AI?",
                response_text="AI is...",
            )
        )
        adapter = ExperienceLedgerAdapter(sel)
        records = adapter.export_all()
        assert len(records) == 1
        assert records[0].kind == MemoryKind.EPISODIC

    def test_preserves_query_response(self):
        sel = MockSEL()
        sel.add(MockEpisode(content_hash="h1", query_text="Q?", response_text="A!"))
        adapter = ExperienceLedgerAdapter(sel)
        records = adapter.export_all()
        assert "Q?" in records[0].content
        assert "A!" in records[0].content

    def test_source_is_experience_ledger(self):
        sel = MockSEL()
        sel.add(MockEpisode(content_hash="h1", query_text="Q", response_text="A"))
        adapter = ExperienceLedgerAdapter(sel)
        records = adapter.export_all()
        assert records[0].source == "experience_ledger"

    def test_export_recent(self):
        sel = MockSEL()
        for i in range(10):
            sel.add(
                MockEpisode(
                    content_hash=f"h{i}", query_text=f"Q{i}", response_text=f"A{i}"
                )
            )
        adapter = ExperienceLedgerAdapter(sel)
        records = adapter.export_recent(limit=3)
        assert len(records) == 3

    def test_metadata_includes_chain(self):
        sel = MockSEL()
        sel.add(
            MockEpisode(
                content_hash="h1",
                query_text="Q",
                response_text="A",
                prev_hash="genesis",
                sequence_number=0,
            )
        )
        adapter = ExperienceLedgerAdapter(sel)
        records = adapter.export_all()
        assert records[0].metadata.get("chain_prev") == "genesis"


class TestPatternMemoryAdapter:
    def test_not_available_returns_empty(self):
        adapter = PatternMemoryAdapter(pattern_memory=None)
        assert not adapter.available
        assert adapter.export_all() == []
