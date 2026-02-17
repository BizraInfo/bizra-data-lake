# Phase 3: Unified SQLite Store (v2) with FTS5

> ADR-006 | Unified Memory Service — Persistence Layer
> Standing on Giants: Hipp (SQLite, 2000) · Porter (FTS5 stemming)

## 3.1 — `core/memory/unified_store.py`

### Requirements
- SQLite v2 schema extending existing v1 (core/living_memory/persistence.py)
- FTS5 virtual table for full-text keyword search
- WAL mode + busy_timeout=5000ms (SAPE gap fix)
- Content-addressable IDs (hex_digest)
- Embeddings stored as BLOB (raw float32 bytes)
- Schema migration from v1 to v2 (non-destructive)
- adapter_source column to track provenance

### Schema v2

```sql
-- Metadata
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

-- Main memory table
CREATE TABLE IF NOT EXISTS memories_v2 (
    id TEXT PRIMARY KEY,                    -- hex_digest(content)
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL,              -- MemoryType enum value
    embedding BLOB,                         -- float32 raw bytes (768 * 4 = 3072 bytes)
    created_at TEXT NOT NULL,               -- ISO 8601
    last_accessed TEXT NOT NULL,
    access_count INTEGER DEFAULT 0,
    ihsan_score REAL DEFAULT 1.0,
    snr_score REAL DEFAULT 1.0,
    confidence REAL DEFAULT 1.0,
    state TEXT DEFAULT 'active',
    source TEXT DEFAULT 'unknown',
    related_ids TEXT DEFAULT '[]',          -- JSON array of strings
    parent_id TEXT,
    importance REAL DEFAULT 1.0,
    emotional_weight REAL DEFAULT 0.5,
    adapter_source TEXT                     -- 'living_memory' | 'sel' | 'pattern' | 'direct'
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_v2_type ON memories_v2(memory_type);
CREATE INDEX IF NOT EXISTS idx_v2_state ON memories_v2(state);
CREATE INDEX IF NOT EXISTS idx_v2_accessed ON memories_v2(last_accessed DESC);
CREATE INDEX IF NOT EXISTS idx_v2_ihsan ON memories_v2(ihsan_score DESC);
CREATE INDEX IF NOT EXISTS idx_v2_importance ON memories_v2(importance DESC);
CREATE INDEX IF NOT EXISTS idx_v2_adapter ON memories_v2(adapter_source);

-- FTS5 virtual table for keyword search
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    id UNINDEXED,
    content,
    source,
    tokenize='porter unicode61'
);
```

### Pseudocode

```
IMPORT sqlite3, json, logging FROM stdlib
IMPORT numpy as np
IMPORT Path FROM pathlib
IMPORT MemoryRecord, MemoryType, MemoryState FROM core.memory.types
IMPORT StoreConfig FROM core.memory.config
IMPORT hex_digest FROM core.proof_engine.canonical

logger = GET_LOGGER(__name__)

SCHEMA_VERSION_V2 = 2

CLASS UnifiedSQLiteStore:
    """
    SQLite v2 persistent store with FTS5 keyword search.

    Extends the v1 schema (core/living_memory/persistence.py) with:
    - FTS5 full-text search virtual table
    - adapter_source provenance tracking
    - Content-addressable deduplication
    """

    FUNCTION __init__(config: StoreConfig = StoreConfig()):
        self._config = config
        self._conn: Optional[sqlite3.Connection] = None

    FUNCTION initialize() -> None:
        """Open connection, set PRAGMAs, ensure schema."""
        self._config.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._config.db_path),
            timeout=self._config.busy_timeout_ms / 1000,
        )
        self._conn.row_factory = sqlite3.Row

        # PRAGMAs (match existing persistence.py pattern)
        IF self._config.wal_mode:
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute(f"PRAGMA busy_timeout={self._config.busy_timeout_ms}")

        self._ensure_schema()
        logger.info(f"UnifiedSQLiteStore opened: {self._config.db_path}")

    FUNCTION close() -> None:
        IF self._conn:
            self._conn.close()
            self._conn = None

    # ── CRUD ──

    FUNCTION save(record: MemoryRecord) -> str:
        """Insert or replace a memory record. Returns the record ID."""
        # Content-addressable ID if not already set
        IF record.id IS None OR record.id == "":
            record.id = hex_digest(record.content)

        embedding_blob = None
        IF record.embedding IS NOT None:
            embedding_blob = record.embedding.astype(np.float32).tobytes()

        self._conn.execute("""
            INSERT OR REPLACE INTO memories_v2
            (id, content, memory_type, embedding, created_at, last_accessed,
             access_count, ihsan_score, snr_score, confidence, state, source,
             related_ids, parent_id, importance, emotional_weight, adapter_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.id, record.content, record.memory_type.value,
            embedding_blob, record.created_at.isoformat(),
            record.last_accessed.isoformat(), record.access_count,
            record.ihsan_score, record.snr_score, record.confidence,
            record.state.value, record.source,
            json.dumps(list(record.related_ids)), record.parent_id,
            record.importance, record.emotional_weight, record.adapter_source,
        ))

        # Update FTS5 index
        self._conn.execute("""
            INSERT OR REPLACE INTO memories_fts(id, content, source)
            VALUES (?, ?, ?)
        """, (record.id, record.content, record.source))

        self._conn.commit()
        RETURN record.id

    FUNCTION get(id: str) -> Optional[MemoryRecord]:
        """Retrieve a single record by ID."""
        row = self._conn.execute(
            "SELECT * FROM memories_v2 WHERE id = ?", (id,)
        ).fetchone()
        IF row IS None:
            RETURN None
        RETURN self._row_to_record(row)

    FUNCTION delete(id: str) -> bool:
        """Delete a record by ID. Returns True if found."""
        cursor = self._conn.execute("DELETE FROM memories_v2 WHERE id = ?", (id,))
        self._conn.execute("DELETE FROM memories_fts WHERE id = ?", (id,))
        self._conn.commit()
        RETURN cursor.rowcount > 0

    FUNCTION touch(id: str) -> None:
        """Update last_accessed and increment access_count."""
        self._conn.execute("""
            UPDATE memories_v2
            SET last_accessed = ?, access_count = access_count + 1
            WHERE id = ?
        """, (datetime.now(timezone.utc).isoformat(), id))
        self._conn.commit()

    # ── Queries ──

    FUNCTION keyword_search(query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """FTS5 full-text search. Returns [(id, rank_score), ...]."""
        rows = self._conn.execute("""
            SELECT id, rank FROM memories_fts
            WHERE memories_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, top_k)).fetchall()
        # FTS5 rank is negative (lower = better), normalize to 0..1
        IF NOT rows:
            RETURN []
        max_rank = abs(rows[-1]["rank"]) + 1e-10
        RETURN [(row["id"], 1.0 - abs(row["rank"]) / max_rank) FOR row IN rows]

    FUNCTION query_by_type(memory_type: MemoryType, limit: int = 100) -> List[MemoryRecord]:
        """Retrieve records by type, ordered by importance."""
        rows = self._conn.execute("""
            SELECT * FROM memories_v2
            WHERE memory_type = ? AND state != 'deleted'
            ORDER BY importance DESC, last_accessed DESC
            LIMIT ?
        """, (memory_type.value, limit)).fetchall()
        RETURN [self._row_to_record(row) FOR row IN rows]

    FUNCTION count() -> int:
        RETURN self._conn.execute("SELECT COUNT(*) FROM memories_v2").fetchone()[0]

    FUNCTION list_ids(state: Optional[str] = None) -> List[str]:
        """List all record IDs, optionally filtered by state."""
        IF state:
            rows = self._conn.execute(
                "SELECT id FROM memories_v2 WHERE state = ?", (state,)
            ).fetchall()
        ELSE:
            rows = self._conn.execute("SELECT id FROM memories_v2").fetchall()
        RETURN [row["id"] FOR row IN rows]

    # ── Internal ──

    FUNCTION _ensure_schema() -> None:
        """Create tables/indexes if missing. Migrate from v1 if needed."""
        version = self._get_schema_version()
        IF version < SCHEMA_VERSION_V2:
            self._conn.executescript(V2_SCHEMA_SQL)
            self._set_schema_version(SCHEMA_VERSION_V2)
            logger.info(f"Schema migrated to v{SCHEMA_VERSION_V2}")

    FUNCTION _row_to_record(row: sqlite3.Row) -> MemoryRecord:
        """Convert a database row to a MemoryRecord."""
        embedding = None
        IF row["embedding"] IS NOT None:
            embedding = np.frombuffer(row["embedding"], dtype=np.float32)

        RETURN MemoryRecord(
            id=row["id"],
            content=row["content"],
            memory_type=MemoryType(row["memory_type"]),
            embedding=embedding,
            created_at=datetime.fromisoformat(row["created_at"]),
            last_accessed=datetime.fromisoformat(row["last_accessed"]),
            access_count=row["access_count"],
            ihsan_score=row["ihsan_score"],
            snr_score=row["snr_score"],
            confidence=row["confidence"],
            state=MemoryState(row["state"]),
            source=row["source"],
            related_ids=set(json.loads(row["related_ids"])),
            parent_id=row["parent_id"],
            importance=row["importance"],
            emotional_weight=row["emotional_weight"],
            adapter_source=row["adapter_source"],
        )

    FUNCTION _get_schema_version() -> int:
        TRY:
            row = self._conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
            RETURN row[0] OR 0
        EXCEPT sqlite3.OperationalError:
            RETURN 0

    FUNCTION _set_schema_version(version: int) -> None:
        self._conn.execute("INSERT OR REPLACE INTO schema_version VALUES (?)", (version,))
        self._conn.commit()
```

### TDD Anchors

```
TEST test_save_and_get(tmp_path):
    store = UnifiedSQLiteStore(StoreConfig(db_path=tmp_path/"test.db"))
    store.initialize()
    record = MemoryRecord(id="abc", content="hello world", memory_type=MemoryType.SEMANTIC)
    store.save(record)
    retrieved = store.get("abc")
    ASSERT retrieved IS NOT None
    ASSERT retrieved.content == "hello world"
    store.close()

TEST test_content_addressable_id(tmp_path):
    store = UnifiedSQLiteStore(StoreConfig(db_path=tmp_path/"test.db"))
    store.initialize()
    record = MemoryRecord(id="", content="unique content", memory_type=MemoryType.EPISODIC)
    id = store.save(record)
    ASSERT id == hex_digest("unique content")
    store.close()

TEST test_fts5_keyword_search(tmp_path):
    store = UnifiedSQLiteStore(StoreConfig(db_path=tmp_path/"test.db"))
    store.initialize()
    store.save(MemoryRecord(id="1", content="quantum computing breakthrough", memory_type=MemoryType.SEMANTIC))
    store.save(MemoryRecord(id="2", content="classical music theory", memory_type=MemoryType.SEMANTIC))
    results = store.keyword_search("quantum")
    ASSERT len(results) >= 1
    ASSERT results[0][0] == "1"
    store.close()

TEST test_delete(tmp_path):
    store = UnifiedSQLiteStore(StoreConfig(db_path=tmp_path/"test.db"))
    store.initialize()
    store.save(MemoryRecord(id="del", content="to delete", memory_type=MemoryType.WORKING))
    ASSERT store.delete("del") IS True
    ASSERT store.get("del") IS None
    store.close()

TEST test_touch_updates_access(tmp_path):
    store = UnifiedSQLiteStore(StoreConfig(db_path=tmp_path/"test.db"))
    store.initialize()
    store.save(MemoryRecord(id="t", content="test", memory_type=MemoryType.WORKING))
    store.touch("t")
    record = store.get("t")
    ASSERT record.access_count == 1
    store.close()

TEST test_embedding_roundtrip(tmp_path):
    store = UnifiedSQLiteStore(StoreConfig(db_path=tmp_path/"test.db"))
    store.initialize()
    emb = np.random.randn(768).astype(np.float32)
    store.save(MemoryRecord(id="e", content="with embedding", memory_type=MemoryType.SEMANTIC, embedding=emb))
    record = store.get("e")
    np.testing.assert_array_almost_equal(record.embedding, emb)
    store.close()
```

### Edge Cases

1. **Concurrent writers** — WAL + busy_timeout handles; SQLite serializes writes
2. **FTS5 special characters** — porter tokenizer handles unicode; escape quotes in query
3. **Empty content** — hex_digest("") is valid but useless; guard at AgentDB level
4. **Embedding size mismatch** — store as raw bytes; consumer validates dim on load
5. **Schema migration from v1** — separate migrator module handles the v1 -> v2 ETL
6. **Disk full** — SQLite raises OperationalError; catch and log at AgentDB level
