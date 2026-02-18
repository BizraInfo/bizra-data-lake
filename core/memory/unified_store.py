"""
Unified Store — SQLite v2 with FTS5 keyword search.

Extends the Phase 18 SQLite v1 schema (core/living_memory/persistence.py)
with:
- FTS5 full-text search on content
- Schema version 2 tracking
- Content-addressable IDs via hex_digest
- WAL mode + busy_timeout for concurrent access

Standing on Giants:
- SQLite (Hipp, 2000) — most deployed database
- FTS5 (2015) — full-text search extension built into SQLite
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .config import MemoryConfig
from .types import MemoryKind, MemoryRecord, RecordState

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2


class UnifiedStore:
    """SQLite v2 store with FTS5 full-text search.

    Provides CRUD operations for MemoryRecord plus keyword search
    via the FTS5 virtual table.
    """

    def __init__(self, config: Optional[MemoryConfig] = None) -> None:
        self._config = config or MemoryConfig()
        self._conn: Optional[sqlite3.Connection] = None

    @property
    def db_path(self) -> Path:
        return self._config.sqlite_path

    def initialize(self) -> None:
        """Open connection and create/migrate schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self.db_path),
            timeout=self._config.sqlite_busy_timeout_ms / 1000.0,
        )
        self._conn.row_factory = sqlite3.Row

        if self._config.sqlite_wal_mode:
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute(f"PRAGMA busy_timeout={self._config.sqlite_busy_timeout_ms}")

        self._create_schema()
        logger.info(f"UnifiedStore opened: {self.db_path}")

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Store not initialized — call initialize() first")
        return self._conn

    def _create_schema(self) -> None:
        conn = self._ensure_conn()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY
            );

            CREATE TABLE IF NOT EXISTS records (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'semantic',
                state TEXT NOT NULL DEFAULT 'active',

                -- Embedding stored as blob (float32 array)
                embedding BLOB,

                -- Quality scores
                ihsan_score REAL DEFAULT 1.0,
                snr_score REAL DEFAULT 1.0,
                importance REAL DEFAULT 0.5,

                -- Provenance
                source TEXT DEFAULT 'unknown',
                source_id TEXT,

                -- JSON arrays
                related_ids TEXT DEFAULT '[]',
                tags TEXT DEFAULT '[]',

                -- Timestamps (ISO-8601)
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_accessed TEXT NOT NULL,
                access_count INTEGER DEFAULT 0,

                -- Extensible metadata (JSON object)
                metadata TEXT DEFAULT '{}'
            );

            -- Indexes for common queries
            CREATE INDEX IF NOT EXISTS idx_records_kind ON records(kind);
            CREATE INDEX IF NOT EXISTS idx_records_state ON records(state);
            CREATE INDEX IF NOT EXISTS idx_records_source ON records(source);
            CREATE INDEX IF NOT EXISTS idx_records_accessed ON records(last_accessed DESC);
            CREATE INDEX IF NOT EXISTS idx_records_importance ON records(importance DESC);
            CREATE INDEX IF NOT EXISTS idx_records_ihsan ON records(ihsan_score DESC);
            CREATE INDEX IF NOT EXISTS idx_records_created ON records(created_at DESC);

            -- FTS5 virtual table for keyword search
            CREATE VIRTUAL TABLE IF NOT EXISTS records_fts USING fts5(
                id UNINDEXED,
                content,
                tags,
                tokenize='porter unicode61'
            );

            INSERT OR IGNORE INTO schema_version (version) VALUES (2);
        """
        )
        conn.commit()

    # ── CRUD ─────────────────────────────────────────────────────────────

    def upsert(self, record: MemoryRecord) -> None:
        """Insert or replace a record (and update FTS)."""
        conn = self._ensure_conn()

        embedding_blob = None
        if record.embedding is not None:
            import numpy as np

            embedding_blob = np.asarray(record.embedding, dtype=np.float32).tobytes()

        now = datetime.now(timezone.utc).isoformat()
        record.updated_at = datetime.now(timezone.utc)

        conn.execute(
            """
            INSERT OR REPLACE INTO records (
                id, content, kind, state, embedding,
                ihsan_score, snr_score, importance,
                source, source_id, related_ids, tags,
                created_at, updated_at, last_accessed, access_count,
                metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.content,
                record.kind.value,
                record.state.value,
                embedding_blob,
                record.ihsan_score,
                record.snr_score,
                record.importance,
                record.source,
                record.source_id,
                json.dumps(record.related_ids),
                json.dumps(record.tags),
                record.created_at.isoformat(),
                now,
                record.last_accessed.isoformat(),
                record.access_count,
                json.dumps(record.metadata),
            ),
        )

        # Update FTS index
        conn.execute(
            "INSERT OR REPLACE INTO records_fts (id, content, tags) VALUES (?, ?, ?)",
            (record.id, record.content, " ".join(record.tags)),
        )
        conn.commit()

    def upsert_batch(self, records: List[MemoryRecord]) -> int:
        """Batch upsert within a single transaction."""
        conn = self._ensure_conn()
        count = 0
        with conn:
            for record in records:
                import numpy as np

                embedding_blob = None
                if record.embedding is not None:
                    embedding_blob = np.asarray(
                        record.embedding, dtype=np.float32
                    ).tobytes()

                now = datetime.now(timezone.utc).isoformat()

                conn.execute(
                    """
                    INSERT OR REPLACE INTO records (
                        id, content, kind, state, embedding,
                        ihsan_score, snr_score, importance,
                        source, source_id, related_ids, tags,
                        created_at, updated_at, last_accessed, access_count,
                        metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.id,
                        record.content,
                        record.kind.value,
                        record.state.value,
                        embedding_blob,
                        record.ihsan_score,
                        record.snr_score,
                        record.importance,
                        record.source,
                        record.source_id,
                        json.dumps(record.related_ids),
                        json.dumps(record.tags),
                        record.created_at.isoformat(),
                        now,
                        record.last_accessed.isoformat(),
                        record.access_count,
                        json.dumps(record.metadata),
                    ),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO records_fts (id, content, tags) VALUES (?, ?, ?)",
                    (record.id, record.content, " ".join(record.tags)),
                )
                count += 1
        return count

    def get(self, record_id: str) -> Optional[MemoryRecord]:
        """Get a record by ID."""
        conn = self._ensure_conn()
        cursor = conn.execute("SELECT * FROM records WHERE id = ?", (record_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def delete(self, record_id: str, hard: bool = False) -> bool:
        """Soft-delete (set state=deleted) or hard-delete a record."""
        conn = self._ensure_conn()
        if hard:
            conn.execute("DELETE FROM records WHERE id = ?", (record_id,))
            conn.execute("DELETE FROM records_fts WHERE id = ?", (record_id,))
        else:
            conn.execute(
                "UPDATE records SET state = ? WHERE id = ?",
                (RecordState.DELETED.value, record_id),
            )
        conn.commit()
        return True

    def count(self, state: Optional[RecordState] = None) -> int:
        conn = self._ensure_conn()
        if state:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM records WHERE state = ?", (state.value,)
            )
        else:
            cursor = conn.execute("SELECT COUNT(*) FROM records")
        return cursor.fetchone()[0]

    def list_ids(
        self,
        state: Optional[RecordState] = None,
        kind: Optional[MemoryKind] = None,
        limit: int = 1000,
    ) -> List[str]:
        conn = self._ensure_conn()
        conditions = []
        params: list = []
        if state:
            conditions.append("state = ?")
            params.append(state.value)
        if kind:
            conditions.append("kind = ?")
            params.append(kind.value)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        cursor = conn.execute(
            f"SELECT id FROM records {where} LIMIT ?", params + [limit]
        )
        return [row["id"] for row in cursor]

    # ── FTS5 Keyword Search ──────────────────────────────────────────────

    def keyword_search(self, query: str, top_k: int = 10) -> List[tuple[str, float]]:
        """Full-text search using FTS5.

        Returns list of (record_id, bm25_score) sorted by relevance.
        BM25 scores are negative (more negative = more relevant).
        We negate them so higher = better.
        """
        conn = self._ensure_conn()
        cursor = conn.execute(
            """
            SELECT id, rank
            FROM records_fts
            WHERE records_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (query, top_k),
        )
        results = []
        for row in cursor:
            # FTS5 rank is negative BM25 (more negative = more relevant)
            # Normalize to 0-1 range: we use -rank as raw relevance
            raw_score = -float(row["rank"])
            results.append((row["id"], raw_score))

        # Normalize scores to 0-1 if we have results
        if results:
            max_score = max(s for _, s in results) or 1.0
            results = [(rid, s / max_score) for rid, s in results]

        return results

    # ── Bulk Operations ──────────────────────────────────────────────────

    def load_all_active(self) -> List[MemoryRecord]:
        """Load all active records (for HNSW index rebuilds)."""
        conn = self._ensure_conn()
        cursor = conn.execute(
            "SELECT * FROM records WHERE state = ?", (RecordState.ACTIVE.value,)
        )
        return [self._row_to_record(row) for row in cursor]

    def load_with_embeddings(self) -> List[tuple[str, bytes]]:
        """Load just IDs and embedding blobs (for HNSW rebuild)."""
        conn = self._ensure_conn()
        cursor = conn.execute(
            "SELECT id, embedding FROM records WHERE state = ? AND embedding IS NOT NULL",
            (RecordState.ACTIVE.value,),
        )
        return [(row["id"], row["embedding"]) for row in cursor]

    def touch_access(self, record_id: str) -> None:
        """Update last_accessed and increment access_count."""
        conn = self._ensure_conn()
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE records SET last_accessed = ?, access_count = access_count + 1 WHERE id = ?",
            (now, record_id),
        )
        conn.commit()

    # ── Internal ─────────────────────────────────────────────────────────

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        import numpy as np

        embedding = None
        if row["embedding"] is not None:
            embedding = list(np.frombuffer(row["embedding"], dtype=np.float32))

        return MemoryRecord(
            id=row["id"],
            content=row["content"],
            kind=MemoryKind(row["kind"]),
            state=RecordState(row["state"]),
            embedding=embedding,
            ihsan_score=row["ihsan_score"],
            snr_score=row["snr_score"],
            importance=row["importance"],
            source=row["source"],
            source_id=row["source_id"],
            related_ids=json.loads(row["related_ids"] or "[]"),
            tags=json.loads(row["tags"] or "[]"),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            last_accessed=datetime.fromisoformat(row["last_accessed"]),
            access_count=row["access_count"],
            metadata=json.loads(row["metadata"] or "{}"),
        )
