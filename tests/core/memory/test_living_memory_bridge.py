"""
Tests for the LivingMemory ↔ AgentDB Bidirectional Bridge.

Validates:
  - Forward conversion: MemoryEntry → MemoryRecord (all fields)
  - Reverse conversion: MemoryRecord → entry dict (round-trip fidelity)
  - Bridge bulk sync: sync_to_agentdb()
  - Bridge single sync: sync_entry()
  - HNSW-accelerated search via bridge
  - Reverse pull: sync_from_agentdb()
  - Cross-system stats
  - Edge cases: deleted entries, missing embeddings, empty memory

Standing on Giants: Deming (PDCA, 1950) — every gate must be testable.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np
import pytest

from core.memory.adapters.living_memory import (
    LivingMemoryAdapter,
    LivingMemoryBridge,
)
from core.memory.agent_db import AgentDB
from core.memory.config import MemoryConfig
from core.memory.types import MemoryKind, MemoryRecord, RecordState

# ── Minimal LivingMemory stubs (avoid importing core.living_memory) ─────


class _MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"
    PROSPECTIVE = "prospective"


class _MemoryState(str, Enum):
    ACTIVE = "active"
    CONSOLIDATING = "consolidating"
    ARCHIVED = "archived"
    DECAYING = "decaying"
    CORRUPTED = "corrupted"
    DELETED = "deleted"


@dataclass
class _MockEntry:
    """Mimics core.living_memory.core.MemoryEntry shape."""

    id: str
    content: str
    memory_type: _MemoryType = _MemoryType.SEMANTIC
    embedding: Optional[np.ndarray] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    reinforcement_count: int = 1
    ihsan_score: float = 0.95
    snr_score: float = 0.92
    confidence: float = 1.0
    state: _MemoryState = _MemoryState.ACTIVE
    source: str = "test"
    related_ids: Set[str] = field(default_factory=set)
    parent_id: Optional[str] = None
    importance: float = 0.8
    emotional_weight: float = 0.6


class _MockLivingMemoryCore:
    """Minimal duck-type of LivingMemoryCore for testing."""

    def __init__(self) -> None:
        self._memories: Dict[str, _MockEntry] = {}

    def add(self, entry: _MockEntry) -> None:
        self._memories[entry.id] = entry


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def lm_core() -> _MockLivingMemoryCore:
    core = _MockLivingMemoryCore()
    core.add(
        _MockEntry(
            id="ep-001",
            content="User asked about constitutional thresholds in the codebase",
            memory_type=_MemoryType.EPISODIC,
            embedding=np.random.rand(384).astype(np.float32),
            importance=0.9,
            ihsan_score=0.97,
            snr_score=0.95,
            source="session",
            related_ids={"sem-001"},
        )
    )
    core.add(
        _MockEntry(
            id="sem-001",
            content="Ihsan threshold is 0.95 for production, defined in constants.py",
            memory_type=_MemoryType.SEMANTIC,
            embedding=np.random.rand(384).astype(np.float32),
            importance=0.95,
            ihsan_score=0.99,
            confidence=1.0,
        )
    )
    core.add(
        _MockEntry(
            id="proc-001",
            content="To run tests use: pytest tests/ -m 'not slow'",
            memory_type=_MemoryType.PROCEDURAL,
            importance=0.7,
        )
    )
    core.add(
        _MockEntry(
            id="work-001",
            content="Currently analyzing the memory bridge architecture",
            memory_type=_MemoryType.WORKING,
            state=_MemoryState.ACTIVE,
        )
    )
    core.add(
        _MockEntry(
            id="del-001",
            content="This memory was deleted",
            memory_type=_MemoryType.SEMANTIC,
            state=_MemoryState.DELETED,
        )
    )
    return core


@pytest.fixture
def adapter(lm_core: _MockLivingMemoryCore) -> LivingMemoryAdapter:
    return LivingMemoryAdapter(lm_core)


@pytest.fixture
def agent_db(tmp_path: Path) -> AgentDB:
    db = AgentDB(MemoryConfig(data_dir=tmp_path, auto_embed=False))
    db.initialize()
    return db


@pytest.fixture
def bridge(lm_core: _MockLivingMemoryCore, agent_db: AgentDB) -> LivingMemoryBridge:
    return LivingMemoryBridge(lm_core, agent_db)


# ══════════════════════════════════════════════════════════════════════════
#  1. Forward Conversion: MemoryEntry → MemoryRecord
# ══════════════════════════════════════════════════════════════════════════


class TestForwardConversion:
    """entry_to_record() correctness."""

    def test_episodic_entry_maps_to_episodic_kind(
        self, adapter: LivingMemoryAdapter, lm_core: _MockLivingMemoryCore
    ) -> None:
        entry = lm_core._memories["ep-001"]
        record = adapter.entry_to_record(entry)
        assert record is not None
        assert record.kind == MemoryKind.EPISODIC
        assert record.state == RecordState.ACTIVE

    def test_semantic_entry_preserves_scores(
        self, adapter: LivingMemoryAdapter, lm_core: _MockLivingMemoryCore
    ) -> None:
        entry = lm_core._memories["sem-001"]
        record = adapter.entry_to_record(entry)
        assert record is not None
        assert record.ihsan_score == 0.99
        assert record.importance == 0.95

    def test_procedural_entry_no_embedding(
        self, adapter: LivingMemoryAdapter, lm_core: _MockLivingMemoryCore
    ) -> None:
        entry = lm_core._memories["proc-001"]
        record = adapter.entry_to_record(entry)
        assert record is not None
        assert record.embedding is None
        assert record.kind == MemoryKind.PROCEDURAL

    def test_working_entry_maps_correctly(
        self, adapter: LivingMemoryAdapter, lm_core: _MockLivingMemoryCore
    ) -> None:
        entry = lm_core._memories["work-001"]
        record = adapter.entry_to_record(entry)
        assert record is not None
        assert record.kind == MemoryKind.WORKING

    def test_deleted_entry_returns_none(
        self, adapter: LivingMemoryAdapter, lm_core: _MockLivingMemoryCore
    ) -> None:
        entry = lm_core._memories["del-001"]
        record = adapter.entry_to_record(entry)
        assert record is None

    def test_embedding_converted_to_list(
        self, adapter: LivingMemoryAdapter, lm_core: _MockLivingMemoryCore
    ) -> None:
        entry = lm_core._memories["ep-001"]
        record = adapter.entry_to_record(entry)
        assert record is not None
        assert isinstance(record.embedding, list)
        assert len(record.embedding) == 384

    def test_related_ids_preserved(
        self, adapter: LivingMemoryAdapter, lm_core: _MockLivingMemoryCore
    ) -> None:
        entry = lm_core._memories["ep-001"]
        record = adapter.entry_to_record(entry)
        assert record is not None
        assert "sem-001" in record.related_ids

    def test_metadata_includes_origin(
        self, adapter: LivingMemoryAdapter, lm_core: _MockLivingMemoryCore
    ) -> None:
        entry = lm_core._memories["sem-001"]
        record = adapter.entry_to_record(entry)
        assert record is not None
        assert record.metadata["origin"] == "living_memory"
        assert record.metadata["confidence"] == 1.0

    def test_source_prefixed(
        self, adapter: LivingMemoryAdapter, lm_core: _MockLivingMemoryCore
    ) -> None:
        entry = lm_core._memories["ep-001"]
        record = adapter.entry_to_record(entry)
        assert record is not None
        assert record.source.startswith("living_memory:")

    def test_tags_include_type(
        self, adapter: LivingMemoryAdapter, lm_core: _MockLivingMemoryCore
    ) -> None:
        entry = lm_core._memories["ep-001"]
        record = adapter.entry_to_record(entry)
        assert record is not None
        assert "lm_type:episodic" in record.tags


class TestExportAll:
    """Bulk export via export_all()."""

    def test_exports_active_entries_only(self, adapter: LivingMemoryAdapter) -> None:
        records = adapter.export_all()
        # 5 entries total, 1 deleted → 4 exported
        assert len(records) == 4

    def test_all_records_have_source_id(self, adapter: LivingMemoryAdapter) -> None:
        records = adapter.export_all()
        for record in records:
            assert record.source_id is not None


# ══════════════════════════════════════════════════════════════════════════
#  2. Reverse Conversion: MemoryRecord → entry dict
# ══════════════════════════════════════════════════════════════════════════


class TestReverseConversion:
    """record_to_entry_dict() fidelity."""

    def test_round_trip_preserves_content(
        self, adapter: LivingMemoryAdapter, lm_core: _MockLivingMemoryCore
    ) -> None:
        entry = lm_core._memories["sem-001"]
        record = adapter.entry_to_record(entry)
        assert record is not None
        entry_dict = adapter.record_to_entry_dict(record)
        assert entry_dict["content"] == entry.content

    def test_round_trip_preserves_type(
        self, adapter: LivingMemoryAdapter, lm_core: _MockLivingMemoryCore
    ) -> None:
        entry = lm_core._memories["proc-001"]
        record = adapter.entry_to_record(entry)
        assert record is not None
        entry_dict = adapter.record_to_entry_dict(record)
        assert entry_dict["memory_type"] == "procedural"

    def test_round_trip_preserves_scores(
        self, adapter: LivingMemoryAdapter, lm_core: _MockLivingMemoryCore
    ) -> None:
        entry = lm_core._memories["sem-001"]
        record = adapter.entry_to_record(entry)
        assert record is not None
        entry_dict = adapter.record_to_entry_dict(record)
        assert entry_dict["ihsan_score"] == entry.ihsan_score
        assert entry_dict["snr_score"] == entry.snr_score
        assert entry_dict["confidence"] == entry.confidence

    def test_round_trip_preserves_importance(
        self, adapter: LivingMemoryAdapter, lm_core: _MockLivingMemoryCore
    ) -> None:
        entry = lm_core._memories["ep-001"]
        record = adapter.entry_to_record(entry)
        assert record is not None
        entry_dict = adapter.record_to_entry_dict(record)
        assert entry_dict["importance"] == entry.importance
        assert entry_dict["emotional_weight"] == entry.emotional_weight

    def test_source_prefix_stripped(
        self, adapter: LivingMemoryAdapter, lm_core: _MockLivingMemoryCore
    ) -> None:
        entry = lm_core._memories["ep-001"]
        record = adapter.entry_to_record(entry)
        assert record is not None
        entry_dict = adapter.record_to_entry_dict(record)
        # "living_memory:session" → "session"
        assert entry_dict["source"] == "session"

    def test_external_record_source_preserved(
        self, adapter: LivingMemoryAdapter
    ) -> None:
        record = MemoryRecord(
            id="ext-001",
            content="External knowledge",
            source="external_api",
        )
        entry_dict = adapter.record_to_entry_dict(record)
        assert entry_dict["source"] == "external_api"


# ══════════════════════════════════════════════════════════════════════════
#  3. Bridge: sync_to_agentdb()
# ══════════════════════════════════════════════════════════════════════════


class TestBridgeSyncToAgentDB:
    """Bulk sync from LivingMemory to AgentDB."""

    def test_sync_stores_all_active(self, bridge: LivingMemoryBridge) -> None:
        count = bridge.sync_to_agentdb()
        assert count == 4  # 4 active entries

    def test_sync_is_idempotent(self, bridge: LivingMemoryBridge) -> None:
        first = bridge.sync_to_agentdb()
        second = bridge.sync_to_agentdb()
        assert first == second  # Content-addressable → no duplicates

    def test_synced_ids_tracked(self, bridge: LivingMemoryBridge) -> None:
        bridge.sync_to_agentdb()
        assert len(bridge._synced_ids) == 4


class TestBridgeSyncEntry:
    """Single-entry live sync."""

    def test_sync_single_entry(
        self, bridge: LivingMemoryBridge, lm_core: _MockLivingMemoryCore
    ) -> None:
        entry = lm_core._memories["sem-001"]
        record_id = bridge.sync_entry(entry)
        assert record_id is not None

    def test_sync_deleted_entry_returns_none(
        self, bridge: LivingMemoryBridge, lm_core: _MockLivingMemoryCore
    ) -> None:
        entry = lm_core._memories["del-001"]
        record_id = bridge.sync_entry(entry)
        assert record_id is None


# ══════════════════════════════════════════════════════════════════════════
#  4. Bridge: HNSW-accelerated search
# ══════════════════════════════════════════════════════════════════════════


class TestBridgeSearch:
    """Search through AgentDB's HNSW index via bridge."""

    def test_search_after_sync(self, bridge: LivingMemoryBridge) -> None:
        bridge.sync_to_agentdb()
        # Keyword search — matches content containing "constitutional" and "thresholds"
        results = bridge.search("constitutional thresholds", top_k=5)
        # Results may be empty without embedding fn, but call must not raise
        assert isinstance(results, list)
        assert all(isinstance(r, MemoryRecord) for r in results)

    def test_search_with_type_filter(self, bridge: LivingMemoryBridge) -> None:
        bridge.sync_to_agentdb()
        results = bridge.search(
            "Ihsan threshold constants", memory_type="semantic", top_k=5
        )
        assert isinstance(results, list)
        # If results found, they should be semantic kind
        for r in results:
            assert r.kind == MemoryKind.SEMANTIC

    def test_search_empty_db_returns_empty(self, bridge: LivingMemoryBridge) -> None:
        results = bridge.search("anything")
        assert results == []

    def test_search_with_mmr(self, bridge: LivingMemoryBridge) -> None:
        bridge.sync_to_agentdb()
        results = bridge.search(
            "memory architecture", top_k=5, use_mmr=True, mmr_lambda=0.3
        )
        assert isinstance(results, list)


# ══════════════════════════════════════════════════════════════════════════
#  5. Bridge: sync_from_agentdb()
# ══════════════════════════════════════════════════════════════════════════


class TestBridgeSyncFromAgentDB:
    """Pull external records from AgentDB into LivingMemory format."""

    def test_sync_from_returns_entry_dicts(
        self, bridge: LivingMemoryBridge, agent_db: AgentDB
    ) -> None:
        # Store a non-living-memory record directly via store_record (no embedding)
        from core.memory.types import MemoryRecord as MR

        agent_db.store_record(
            MR(
                id="ext-rag-001",
                content="External knowledge from RAG pipeline",
                source="rag_pipeline",
                importance=0.8,
            )
        )
        entry_dicts = bridge.sync_from_agentdb(source_filter="rag_pipeline")
        assert len(entry_dicts) == 1
        assert entry_dicts[0]["source"] == "rag_pipeline"

    def test_sync_from_skips_living_memory_origin(
        self, bridge: LivingMemoryBridge
    ) -> None:
        bridge.sync_to_agentdb()
        # Without source_filter, should skip living_memory-originated records
        entry_dicts = bridge.sync_from_agentdb()
        assert len(entry_dicts) == 0

    def test_sync_from_entry_dict_has_required_fields(
        self, bridge: LivingMemoryBridge, agent_db: AgentDB
    ) -> None:
        from core.memory.types import MemoryRecord as MR

        agent_db.store_record(
            MR(
                id="ext-fact-001",
                content="Test fact",
                source="external",
                importance=0.5,
            )
        )
        entry_dicts = bridge.sync_from_agentdb(source_filter="external")
        assert len(entry_dicts) == 1
        d = entry_dicts[0]
        assert "content" in d
        assert "memory_type" in d
        assert "ihsan_score" in d
        assert "state" in d


# ══════════════════════════════════════════════════════════════════════════
#  6. Bridge: stats()
# ══════════════════════════════════════════════════════════════════════════


class TestBridgeStats:
    """Cross-system statistics."""

    def test_stats_before_sync(self, bridge: LivingMemoryBridge) -> None:
        stats = bridge.stats()
        assert stats["living_memory_entries"] == 5  # includes deleted
        assert stats["synced_ids"] == 0
        assert stats["bridge_active"] is True

    def test_stats_after_sync(self, bridge: LivingMemoryBridge) -> None:
        bridge.sync_to_agentdb()
        stats = bridge.stats()
        assert stats["synced_ids"] == 4
        assert stats["agentdb_total"] >= 4


# ══════════════════════════════════════════════════════════════════════════
#  7. Edge Cases
# ══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Boundary conditions and error resilience."""

    def test_empty_living_memory(self, agent_db: AgentDB) -> None:
        empty_lm = _MockLivingMemoryCore()
        bridge = LivingMemoryBridge(empty_lm, agent_db)
        assert bridge.sync_to_agentdb() == 0
        assert bridge.stats()["living_memory_entries"] == 0

    def test_adapter_backward_compat(
        self, adapter: LivingMemoryAdapter, lm_core: _MockLivingMemoryCore
    ) -> None:
        """_entry_to_record() still works (backward compat alias)."""
        entry = lm_core._memories["sem-001"]
        record = adapter._entry_to_record(entry)
        assert record is not None
        assert record.kind == MemoryKind.SEMANTIC

    def test_prospective_entry(self, adapter: LivingMemoryAdapter) -> None:
        lm = _MockLivingMemoryCore()
        lm.add(
            _MockEntry(
                id="prosp-001",
                content="Plan to implement federation gossip protocol",
                memory_type=_MemoryType.PROSPECTIVE,
            )
        )
        adapter_p = LivingMemoryAdapter(lm)
        records = adapter_p.export_all()
        assert len(records) == 1
        assert records[0].kind == MemoryKind.PROSPECTIVE

    def test_consolidating_state_maps_to_active(
        self, adapter: LivingMemoryAdapter
    ) -> None:
        lm = _MockLivingMemoryCore()
        lm.add(
            _MockEntry(
                id="cons-001",
                content="Consolidating memory",
                state=_MemoryState.CONSOLIDATING,
            )
        )
        adapter_c = LivingMemoryAdapter(lm)
        records = adapter_c.export_all()
        assert len(records) == 1
        assert records[0].state == RecordState.ACTIVE

    def test_decaying_state_maps_to_active(self, adapter: LivingMemoryAdapter) -> None:
        lm = _MockLivingMemoryCore()
        lm.add(
            _MockEntry(
                id="decay-001",
                content="Decaying memory",
                state=_MemoryState.DECAYING,
            )
        )
        adapter_d = LivingMemoryAdapter(lm)
        records = adapter_d.export_all()
        assert len(records) == 1
        assert records[0].state == RecordState.ACTIVE

    def test_bridge_adapter_accessor(self, bridge: LivingMemoryBridge) -> None:
        assert isinstance(bridge.adapter, LivingMemoryAdapter)
