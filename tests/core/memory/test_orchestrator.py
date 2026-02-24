"""Tests for FR-03: Migration Orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest

from core.memory.agent_db import AgentDB
from core.memory.config import MemoryConfig
from core.memory.orchestrator import MigrationOrchestrator, OrchestratorResult
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


def _make_record(content: str, kind: MemoryKind = MemoryKind.SEMANTIC) -> MemoryRecord:
    from core.proof_engine.canonical import hex_digest

    record_id = hex_digest((content + "test").encode())[:16]
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


@dataclass
class _FakeLivingMemory:
    """Minimal stub for LivingMemoryCore."""

    _memories: dict = field(default_factory=dict)


@dataclass
class _FakeEntry:
    id: str
    content: str
    memory_type: MagicMock = field(default_factory=lambda: MagicMock(value="semantic"))
    state: MagicMock = field(default_factory=lambda: MagicMock(value="active"))
    embedding: object = None
    ihsan_score: float = 1.0
    snr_score: float = 1.0
    importance: float = 0.5
    source: str = "test"
    related_ids: list = field(default_factory=list)
    emotional_weight: float = 0.0
    confidence: float = 1.0
    parent_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0


class TestOrchestratorBasics:
    def test_no_sources_returns_empty(self, db):
        orch = MigrationOrchestrator(db)
        result = orch.run()
        assert len(result.phases) == 0
        assert result.total_imported == 0

    def test_summary_format(self, db):
        orch = MigrationOrchestrator(db)
        result = orch.run()
        summary = result.summary()
        assert "complete" in summary.lower()


class TestLivingMemoryMigration:
    def test_migrates_living_memory_entries(self, db):
        lm = _FakeLivingMemory()
        lm._memories = {
            "a1": _FakeEntry(id="a1", content="first memory"),
            "b2": _FakeEntry(id="b2", content="second memory"),
        }

        orch = MigrationOrchestrator(db)
        orch.set_living_memory(lm)
        result = orch.run()

        assert result.total_imported == 2
        assert result.total_errors == 0
        assert len(result.phases) == 1
        assert result.phases[0].source == "living_memory"

    def test_dry_run_no_writes(self, db):
        lm = _FakeLivingMemory()
        lm._memories = {
            "a1": _FakeEntry(id="a1", content="data"),
        }

        initial_count = db.count

        orch = MigrationOrchestrator(db)
        orch.set_living_memory(lm)
        result = orch.run(dry_run=True)

        assert result.dry_run
        assert result.phases[0].records_found == 1
        assert result.phases[0].records_imported == 0
        assert db.count == initial_count


class TestIdempotency:
    def test_double_run_no_duplication(self, db):
        lm = _FakeLivingMemory()
        lm._memories = {
            "x1": _FakeEntry(id="x1", content="idempotent test"),
        }

        orch = MigrationOrchestrator(db)
        orch.set_living_memory(lm)

        orch.run()
        count1 = db.count

        orch.run()
        count2 = db.count

        # Upsert semantics: same IDs, no duplication
        assert count1 == count2


class TestProgressCallback:
    def test_progress_fires(self, db):
        lm = _FakeLivingMemory()
        # Create 200+ entries to trigger progress callback
        lm._memories = {
            f"e{i}": _FakeEntry(id=f"e{i}", content=f"entry {i}")
            for i in range(150)
        }

        calls = []
        orch = MigrationOrchestrator(
            db, on_progress=lambda src, done, total: calls.append((src, done, total))
        )
        orch.set_living_memory(lm)
        orch.run()

        assert len(calls) >= 1  # At least one callback at 100th record


class TestOrchestratorResult:
    def test_to_summary_includes_phases(self):
        from core.memory.orchestrator import MigrationPhaseResult

        result = OrchestratorResult(
            phases=[
                MigrationPhaseResult("living_memory", 50, 48, 1, 1, 100.0),
                MigrationPhaseResult("experience_ledger", 20, 20, 0, 0, 50.0),
            ],
            total_imported=68,
            total_errors=1,
        )
        summary = result.summary()
        assert "living_memory" in summary
        assert "experience_ledger" in summary
        assert "68" in summary

    def test_fluent_api(self, db):
        lm = _FakeLivingMemory()
        orch = (
            MigrationOrchestrator(db)
            .set_living_memory(lm)
        )
        assert orch._living_memory is lm
