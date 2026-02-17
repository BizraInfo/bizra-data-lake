"""Tests for Memory Migrator (v1 -> v2)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from core.memory.agent_db import AgentDB
from core.memory.config import MemoryConfig
from core.memory.migrator import MemoryMigrator


def _create_v1_db(db_path: Path, num_entries: int = 5) -> None:
    """Create a mock v1 SQLite database matching LivingMemory schema."""
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY
        );
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            memory_type TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_accessed TEXT NOT NULL,
            access_count INTEGER DEFAULT 0,
            ihsan_score REAL DEFAULT 1.0,
            snr_score REAL DEFAULT 1.0,
            confidence REAL DEFAULT 1.0,
            state TEXT DEFAULT 'active',
            source TEXT DEFAULT 'test',
            importance REAL DEFAULT 0.5,
            emotional_weight REAL DEFAULT 0.5,
            related_ids TEXT DEFAULT '[]',
            parent_id TEXT,
            embedding BLOB
        );
        INSERT OR IGNORE INTO schema_version (version) VALUES (1);
    """)

    for i in range(num_entries):
        emb = np.random.randn(8).astype(np.float32).tobytes()
        conn.execute(
            """INSERT INTO memories (
                id, content, memory_type, created_at, last_accessed,
                access_count, ihsan_score, snr_score, confidence,
                state, source, importance, emotional_weight,
                related_ids, parent_id, embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                f"v1_entry_{i}",
                f"Test content number {i}",
                "semantic",
                "2026-01-01T00:00:00+00:00",
                "2026-02-01T00:00:00+00:00",
                i,
                0.95,
                0.90,
                0.85,
                "active",
                "test",
                0.5 + i * 0.1,
                0.5,
                json.dumps([]),
                None,
                emb,
            ),
        )

    conn.commit()
    conn.close()


@pytest.fixture
def v1_db_path(tmp_path) -> Path:
    db_path = tmp_path / "v1_memory.db"
    _create_v1_db(db_path, num_entries=5)
    return db_path


@pytest.fixture
def agent_db(tmp_path) -> AgentDB:
    config = MemoryConfig(
        data_dir=tmp_path / "agent_db",
        hnsw=__import__("core.memory.config", fromlist=["HNSWConfig"]).HNSWConfig(
            dimensions=8, max_elements=100
        ),
    )
    db = AgentDB(config)
    db.initialize()
    return db


class TestMigrator:
    def test_migrate_v1_to_v2(self, agent_db, v1_db_path):
        migrator = MemoryMigrator(agent_db, source_path=v1_db_path)
        result = migrator.migrate()
        assert result.migrated == 5
        assert result.errors == 0
        assert agent_db.count == 5

    def test_migrate_creates_backup(self, agent_db, v1_db_path):
        migrator = MemoryMigrator(agent_db, source_path=v1_db_path)
        migrator.migrate(backup=True)
        bak = v1_db_path.with_suffix(".db.bak")
        assert bak.exists()

    def test_migrate_preserves_content(self, agent_db, v1_db_path):
        migrator = MemoryMigrator(agent_db, source_path=v1_db_path)
        migrator.migrate()
        rec = agent_db.retrieve("v1_entry_0")
        assert rec is not None
        assert rec.content == "Test content number 0"

    def test_migrate_preserves_embeddings(self, agent_db, v1_db_path):
        migrator = MemoryMigrator(agent_db, source_path=v1_db_path)
        migrator.migrate()
        assert agent_db.hnsw.count == 5

    def test_migrate_preserves_scores(self, agent_db, v1_db_path):
        migrator = MemoryMigrator(agent_db, source_path=v1_db_path)
        migrator.migrate()
        rec = agent_db.retrieve("v1_entry_0")
        assert rec is not None
        assert rec.ihsan_score == 0.95
        assert rec.snr_score == 0.90

    def test_migrate_nonexistent_source(self, agent_db, tmp_path):
        migrator = MemoryMigrator(agent_db, source_path=tmp_path / "nonexistent.db")
        result = migrator.migrate()
        assert result.migrated == 0

    def test_migrate_idempotent(self, agent_db, v1_db_path):
        migrator = MemoryMigrator(agent_db, source_path=v1_db_path)
        migrator.migrate()
        migrator.migrate()  # Second run should upsert same records
        assert agent_db.count == 5
