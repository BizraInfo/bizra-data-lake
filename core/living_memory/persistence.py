"""
Living Memory Persistence — SQLite Backend

Replaces JSONL flat-file with a proper SQLite backend for:
- Atomic writes (no half-written state on crash)
- Indexed retrieval by type, score, recency
- WAL mode for concurrent read/write
- Compact storage with JSON for complex fields
- Cross-platform (Windows/Linux/macOS)

Standing on Giants:
- SQLite (Hipp, 2000): Most deployed database in the world
- WAL Mode: Write-Ahead Logging for concurrent access

Created: 2026-02-10 | Phase 18 — Making the system work
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .core import MemoryEntry, MemoryState, MemoryType

logger = logging.getLogger(__name__)

# Schema version for migrations
SCHEMA_VERSION = 1


class SQLiteMemoryStore:
    """
    SQLite-backed persistent memory store.

    Provides durable, indexed storage for LivingMemoryCore entries
    with atomic operations and crash safety via WAL mode.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def initialize(self) -> None:
        """Open connection and ensure schema exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_schema()
        logger.info(f"SQLite memory store opened: {self.db_path}")

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def _create_schema(self) -> None:
        """Create tables and indexes if they don't exist."""
        if self._conn is None:
            raise RuntimeError(
                "Database connection is closed — call initialize() first"
            )
        self._conn.executescript("""
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
                source TEXT DEFAULT 'unknown',
                importance REAL DEFAULT 1.0,
                emotional_weight REAL DEFAULT 0.5,
                related_ids TEXT DEFAULT '[]',
                parent_id TEXT,
                embedding BLOB
            );

            CREATE INDEX IF NOT EXISTS idx_memories_type
                ON memories(memory_type);
            CREATE INDEX IF NOT EXISTS idx_memories_state
                ON memories(state);
            CREATE INDEX IF NOT EXISTS idx_memories_accessed
                ON memories(last_accessed DESC);
            CREATE INDEX IF NOT EXISTS idx_memories_ihsan
                ON memories(ihsan_score DESC);
            CREATE INDEX IF NOT EXISTS idx_memories_importance
                ON memories(importance DESC);

            INSERT OR IGNORE INTO schema_version (version) VALUES (1);
        """)
        self._conn.commit()

    # ── CRUD Operations ─────────────────────────────────────────────────

    def save_entry(self, entry: MemoryEntry) -> None:
        """Insert or update a memory entry."""
        if self._conn is None:
            raise RuntimeError(
                "Database connection is closed — call initialize() first"
            )
        embedding_blob = (
            entry.embedding.tobytes() if entry.embedding is not None else None
        )
        self._conn.execute(
            """
            INSERT OR REPLACE INTO memories (
                id, content, memory_type, created_at, last_accessed,
                access_count, ihsan_score, snr_score, confidence,
                state, source, importance, emotional_weight,
                related_ids, parent_id, embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.id,
                entry.content,
                entry.memory_type.value,
                entry.created_at.isoformat(),
                entry.last_accessed.isoformat(),
                entry.access_count,
                entry.ihsan_score,
                entry.snr_score,
                entry.confidence,
                entry.state.value,
                entry.source,
                entry.importance,
                entry.emotional_weight,
                json.dumps(list(entry.related_ids)),
                entry.parent_id,
                embedding_blob,
            ),
        )
        self._conn.commit()

    def save_batch(self, entries: list[MemoryEntry]) -> int:
        """Batch-save multiple entries in a single transaction."""
        if self._conn is None:
            raise RuntimeError(
                "Database connection is closed — call initialize() first"
            )
        saved = 0
        with self._conn:
            for entry in entries:
                embedding_blob = (
                    entry.embedding.tobytes() if entry.embedding is not None else None
                )
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO memories (
                        id, content, memory_type, created_at, last_accessed,
                        access_count, ihsan_score, snr_score, confidence,
                        state, source, importance, emotional_weight,
                        related_ids, parent_id, embedding
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entry.id,
                        entry.content,
                        entry.memory_type.value,
                        entry.created_at.isoformat(),
                        entry.last_accessed.isoformat(),
                        entry.access_count,
                        entry.ihsan_score,
                        entry.snr_score,
                        entry.confidence,
                        entry.state.value,
                        entry.source,
                        entry.importance,
                        entry.emotional_weight,
                        json.dumps(list(entry.related_ids)),
                        entry.parent_id,
                        embedding_blob,
                    ),
                )
                saved += 1
        return saved

    def load_all(self) -> dict[str, MemoryEntry]:
        """Load all non-deleted memories from database."""
        if self._conn is None:
            raise RuntimeError(
                "Database connection is closed — call initialize() first"
            )
        memories: dict[str, MemoryEntry] = {}
        cursor = self._conn.execute(
            "SELECT * FROM memories WHERE state != ?",
            (MemoryState.DELETED.value,),
        )
        for row in cursor:
            entry = self._row_to_entry(row)
            memories[entry.id] = entry
        logger.info(f"Loaded {len(memories)} memories from SQLite")
        return memories

    def load_by_type(self, memory_type: MemoryType) -> list[MemoryEntry]:
        """Load memories of a specific type."""
        if self._conn is None:
            raise RuntimeError(
                "Database connection is closed — call initialize() first"
            )
        cursor = self._conn.execute(
            "SELECT * FROM memories WHERE memory_type = ? AND state != ?",
            (memory_type.value, MemoryState.DELETED.value),
        )
        return [self._row_to_entry(row) for row in cursor]

    def delete_entry(self, entry_id: str, hard: bool = False) -> bool:
        """Delete a memory entry (soft or hard)."""
        if self._conn is None:
            raise RuntimeError(
                "Database connection is closed — call initialize() first"
            )
        if hard:
            self._conn.execute("DELETE FROM memories WHERE id = ?", (entry_id,))
        else:
            self._conn.execute(
                "UPDATE memories SET state = ? WHERE id = ?",
                (MemoryState.DELETED.value, entry_id),
            )
        self._conn.commit()
        return True

    def count(self, state: Optional[MemoryState] = None) -> int:
        """Count entries, optionally filtered by state."""
        if self._conn is None:
            raise RuntimeError(
                "Database connection is closed — call initialize() first"
            )
        if state:
            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM memories WHERE state = ?", (state.value,)
            )
        else:
            cursor = self._conn.execute("SELECT COUNT(*) FROM memories")
        return cursor.fetchone()[0]

    def get_lowest_scored(self, limit: int = 100) -> list[str]:
        """Get IDs of lowest-scored active entries (for cleanup)."""
        if self._conn is None:
            raise RuntimeError(
                "Database connection is closed — call initialize() first"
            )
        cursor = self._conn.execute(
            """
            SELECT id FROM memories
            WHERE state = ?
            ORDER BY (0.3 * importance + 0.3 * ihsan_score + 0.2 * snr_score
                     + 0.2 * (access_count * 1.0 / (1 + access_count))) ASC
            LIMIT ?
            """,
            (MemoryState.ACTIVE.value, limit),
        )
        return [row["id"] for row in cursor]

    # ── Migration support ───────────────────────────────────────────────

    def migrate_from_jsonl(self, jsonl_path: Path) -> int:
        """Import memories from legacy JSONL file into SQLite."""
        if not jsonl_path.exists():
            return 0

        imported = 0
        entries: list[MemoryEntry] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        entry = MemoryEntry.from_dict(data)
                        entries.append(entry)
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Skipping corrupt JSONL entry: {e}")

        if entries:
            imported = self.save_batch(entries)
            logger.info(f"Migrated {imported} memories from JSONL to SQLite")
            # Rename old file as backup
            backup = jsonl_path.with_suffix(".jsonl.bak")
            jsonl_path.rename(backup)
            logger.info(f"Original JSONL backed up to {backup}")

        return imported

    # ── Internal helpers ────────────────────────────────────────────────

    def _row_to_entry(self, row: sqlite3.Row) -> MemoryEntry:
        """Convert a database row to a MemoryEntry."""
        entry = MemoryEntry(
            id=row["id"],
            content=row["content"],
            memory_type=MemoryType(row["memory_type"]),
        )
        entry.created_at = datetime.fromisoformat(row["created_at"])
        entry.last_accessed = datetime.fromisoformat(row["last_accessed"])
        entry.access_count = row["access_count"]
        entry.ihsan_score = row["ihsan_score"]
        entry.snr_score = row["snr_score"]
        entry.confidence = row["confidence"]
        entry.state = MemoryState(row["state"])
        entry.source = row["source"]
        entry.importance = row["importance"]
        entry.emotional_weight = row["emotional_weight"]
        entry.related_ids = set(json.loads(row["related_ids"] or "[]"))
        entry.parent_id = row["parent_id"]

        # Restore embedding from blob
        if row["embedding"] is not None:
            entry.embedding = np.frombuffer(row["embedding"], dtype=np.float32)

        return entry
