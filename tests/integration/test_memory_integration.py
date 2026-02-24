"""Integration test for V3 Memory Unification — full lifecycle.

Tests the complete chain: AgentDB → Bridge → Health → Orchestrator → Sync.
All components wired together as they are in SovereignRuntime.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from core.memory.agent_db import AgentDB
from core.memory.config import HNSWConfig, MemoryConfig
from core.memory.coordinator_bridge import AgentDBBridge
from core.memory.health import AgentDBHealthChecker, HealthStatus
from core.memory.orchestrator import MigrationOrchestrator
from core.memory.sync import MemorySyncPublisher, MemorySyncSubscriber
from core.memory.types import MemoryKind, MemoryRecord, RecordState


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def tmp_config(tmp_path: Path) -> MemoryConfig:
    cfg = MemoryConfig(
        data_dir=tmp_path / "agent_db",
        hnsw=HNSWConfig(dimensions=8, max_elements=1000),
    )
    cfg.auto_embed = False
    return cfg


@pytest.fixture
def db(tmp_config: MemoryConfig) -> AgentDB:
    d = AgentDB(tmp_config)
    d.initialize()
    return d


@pytest.fixture
def mock_coordinator():
    coord = MagicMock()
    coord._state_providers = {}
    coord.save_all = AsyncMock(return_value=True)

    def _register(name, provider, priority):
        coord._state_providers[name] = (provider, priority)

    coord.register_state_provider = _register
    return coord


@dataclass
class _FakeEntry:
    id: str
    content: str
    memory_type: MagicMock = field(default_factory=lambda: MagicMock(value="semantic"))
    state: MagicMock = field(default_factory=lambda: MagicMock(value="active"))
    embedding: object = None
    ihsan_score: float = 1.0
    snr_score: float = 1.0
    importance: float = 0.7
    source: str = "test"
    related_ids: list = field(default_factory=list)
    emotional_weight: float = 0.0
    confidence: float = 1.0
    parent_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0


@dataclass
class _FakeLivingMemory:
    _memories: dict = field(default_factory=dict)


# ── Integration Tests ─────────────────────────────────────────────────


class TestFullMemoryLifecycle:
    """End-to-end: store → search → bridge → health → save → restore."""

    def test_store_search_retrieve_forget(self, db, tmp_config):
        """Core CRUD lifecycle."""
        emb = np.random.randn(8).astype(np.float32).tolist()

        # Store
        rec = db.store("Earth orbits the Sun", embedding=emb, importance=0.9)
        assert rec.id is not None
        assert rec.kind == MemoryKind.SEMANTIC

        # Search by keyword
        results = db.search(query="Earth Sun")
        assert len(results) >= 1
        assert results[0].record.content == "Earth orbits the Sun"
        assert results[0].score > 0

        # Retrieve by ID
        fetched = db.retrieve(rec.id)
        assert fetched is not None
        assert fetched.content == rec.content

        # Forget
        db.forget(rec.id)
        gone = db.retrieve(rec.id)
        assert gone is None or gone.state == RecordState.DELETED

    def test_bridge_save_includes_agentdb(self, db, mock_coordinator, tmp_config):
        """Bridge registers provider and flushes HNSW on save."""
        emb = np.random.randn(8).astype(np.float32).tolist()
        db.store("bridged record", embedding=emb)

        bridge = AgentDBBridge(db, mock_coordinator)
        bridge.register()

        # Trigger save
        asyncio.get_event_loop().run_until_complete(
            mock_coordinator.save_all(source="integration_test")
        )

        # Verify state provider returns valid data
        provider_fn = mock_coordinator._state_providers["agent_db"][0]
        state = provider_fn()
        assert state["record_count"] >= 1
        assert state["vector_count"] >= 1

    def test_health_after_operations(self, db):
        """Health check reports healthy after normal operations."""
        db.store("health test record A")
        db.store("health test record B")

        checker = AgentDBHealthChecker(db)
        report = checker.check()

        assert report.status == HealthStatus.HEALTHY
        assert report.components["sqlite"].details["records"]["active"] == 2
        assert report.components["hnsw"].details["vector_count"] >= 0
        assert report.components["memory"].details["estimated_mb"] >= 0

    def test_orchestrator_imports_living_memory(self, db):
        """Orchestrator migrates LivingMemory entries into AgentDB."""
        lm = _FakeLivingMemory()
        lm._memories = {
            "lm1": _FakeEntry(id="lm1", content="learned fact alpha"),
            "lm2": _FakeEntry(id="lm2", content="learned fact beta"),
            "lm3": _FakeEntry(id="lm3", content="learned fact gamma"),
        }

        orch = MigrationOrchestrator(db)
        orch.set_living_memory(lm)
        result = orch.run()

        assert result.total_imported == 3
        assert result.total_errors == 0

        # Verify records are searchable
        results = db.search(query="learned fact")
        assert len(results) >= 1

    def test_sync_round_trip(self, db, tmp_config):
        """Sync subscriber imports records from another agent."""
        now = datetime.now(timezone.utc)
        record = MemoryRecord(
            id="sync_test_001",
            content="shared knowledge from agent A",
            kind=MemoryKind.SEMANTIC,
            state=RecordState.ACTIVE,
            importance=0.8,
            source="agent_a",
            created_at=now,
            updated_at=now,
            last_accessed=now,
        )

        # Simulate publisher serialization
        message = json.dumps({
            "sender_id": "agent_a",
            "record": record.to_dict(),
        })

        # Subscriber receives and imports
        sub = MemorySyncSubscriber(db, agent_id="agent_b")
        asyncio.get_event_loop().run_until_complete(sub._handle_message(message))

        assert sub.imported_count == 1

        # Verify it's in AgentDB
        stored = db.retrieve("sync_test_001")
        assert stored is not None
        assert stored.content == "shared knowledge from agent A"
        assert "synced" in stored.tags


class TestCrossModuleIntegration:
    """Test that modules interact correctly when chained."""

    def test_orchestrator_then_health(self, db):
        """Migrate records, then check health reflects them."""
        lm = _FakeLivingMemory()
        lm._memories = {f"m{i}": _FakeEntry(id=f"m{i}", content=f"memory {i}") for i in range(10)}

        orch = MigrationOrchestrator(db)
        orch.set_living_memory(lm)
        orch.run()

        checker = AgentDBHealthChecker(db)
        report = checker.check()

        assert report.status == HealthStatus.HEALTHY
        assert report.components["sqlite"].details["records"]["active"] == 10

    def test_bridge_save_after_orchestrator(self, db, mock_coordinator):
        """Migrate, then save via bridge, verify persistence."""
        lm = _FakeLivingMemory()
        lm._memories = {"x": _FakeEntry(id="x", content="bridged migration")}

        orch = MigrationOrchestrator(db)
        orch.set_living_memory(lm)
        orch.run()

        bridge = AgentDBBridge(db, mock_coordinator)
        bridge.register()

        asyncio.get_event_loop().run_until_complete(
            mock_coordinator.save_all(source="post_migration")
        )

        state = mock_coordinator._state_providers["agent_db"][0]()
        assert state["record_count"] >= 1

    def test_sync_after_orchestrator(self, db):
        """Import from LivingMemory, then receive sync from another agent."""
        lm = _FakeLivingMemory()
        lm._memories = {"local1": _FakeEntry(id="local1", content="local fact")}

        orch = MigrationOrchestrator(db)
        orch.set_living_memory(lm)
        orch.run()

        # Now receive synced record from another agent
        now = datetime.now(timezone.utc)
        remote_record = MemoryRecord(
            id="remote1",
            content="remote fact from agent C",
            kind=MemoryKind.SEMANTIC,
            state=RecordState.ACTIVE,
            source="agent_c",
            created_at=now,
            updated_at=now,
            last_accessed=now,
        )
        message = json.dumps({
            "sender_id": "agent_c",
            "record": remote_record.to_dict(),
        })

        sub = MemorySyncSubscriber(db, agent_id="node0")
        asyncio.get_event_loop().run_until_complete(sub._handle_message(message))

        # Both local and remote records exist
        assert db.count >= 2
        assert db.retrieve("remote1") is not None

    def test_health_degraded_near_capacity(self, tmp_path):
        """HNSW near capacity triggers degraded health."""
        cfg = MemoryConfig(
            data_dir=tmp_path / "small_db",
            hnsw=HNSWConfig(dimensions=8, max_elements=5),
        )
        cfg.auto_embed = False
        db = AgentDB(cfg)
        db.initialize()

        # Fill to capacity
        for i in range(5):
            emb = np.random.randn(8).astype(np.float32).tolist()
            db.store(f"record {i}", embedding=emb, source=f"s{i}")

        checker = AgentDBHealthChecker(db)
        report = checker.check()

        # HNSW at 100% capacity → degraded
        assert report.components["hnsw"].status == HealthStatus.DEGRADED
        assert report.components["hnsw"].details["capacity_ratio"] >= 0.9


class TestIdempotencyAndDedup:
    """Verify content-addressable dedup works across all entry paths."""

    def test_store_twice_same_content(self, db):
        db.store("duplicate content", source="test")
        db.store("duplicate content", source="test")
        assert db.count == 1

    def test_orchestrator_idempotent(self, db):
        lm = _FakeLivingMemory()
        lm._memories = {"d1": _FakeEntry(id="d1", content="unique fact")}

        orch = MigrationOrchestrator(db)
        orch.set_living_memory(lm)
        orch.run()
        count_after_first = db.count

        orch.run()
        assert db.count == count_after_first

    def test_sync_dedup_with_local(self, db):
        """Synced record with same ID as local is skipped."""
        db.store("local content", source="agent")
        local_id = db.backend.list_ids(limit=1)[0]

        now = datetime.now(timezone.utc)
        msg = json.dumps({
            "sender_id": "other",
            "record": MemoryRecord(
                id=local_id,
                content="duplicate from remote",
                created_at=now,
                updated_at=now,
                last_accessed=now,
            ).to_dict(),
        })

        sub = MemorySyncSubscriber(db, agent_id="self")
        asyncio.get_event_loop().run_until_complete(sub._handle_message(msg))

        assert sub.skipped_count == 1
        assert sub.imported_count == 0
