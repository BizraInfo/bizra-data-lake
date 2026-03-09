# Phase 7: Migration and API Integration

> ADR-006 / ADR-009 | Final Integration Phase
> Standing on Giants: Deming (PDCA migration) · Lamport (state transitions)

## 7.1 — `core/memory/migrator.py`

### Requirements
- Migrate SQLite v1 (core/living_memory/persistence.py) to v2 (unified_store)
- Build HNSW index from existing embeddings in v1 database
- NON-DESTRUCTIVE: copies data, preserves original as `.bak`
- Reports progress for large datasets (795K+ Gold Layer rows)
- Idempotent: re-running migration skips already-migrated records

### Pseudocode

```
IMPORT sqlite3, shutil, logging FROM stdlib
IMPORT numpy as np
IMPORT Path FROM pathlib
IMPORT MemoryRecord, MemoryType, MemoryState FROM core.memory.types
IMPORT AgentDBConfig FROM core.memory.config
IMPORT HNSWVectorIndex FROM core.memory.hnsw_index
IMPORT UnifiedSQLiteStore FROM core.memory.unified_store

logger = GET_LOGGER(__name__)

CLASS MemoryMigrator:
    """
    Non-destructive migration from SQLite v1 to AgentDB v2.

    Steps:
    1. Copy v1 database to .bak (safety)
    2. Read all entries from v1
    3. Write to v2 store (content-addressable dedup)
    4. Build HNSW index from embeddings
    5. Report migration summary
    """

    FUNCTION __init__(config: AgentDBConfig = AgentDBConfig()):
        self._config = config

    FUNCTION migrate_from_v1(
        v1_db_path: Path,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Migrate a v1 SQLite database to AgentDB v2.

        Returns migration report: {migrated, skipped, errors, elapsed}.
        """
        IF NOT v1_db_path.exists():
            RAISE FileNotFoundError(f"v1 database not found: {v1_db_path}")

        # Step 1: Safety backup
        backup_path = v1_db_path.with_suffix(".v1.bak")
        IF NOT backup_path.exists():
            shutil.copy2(v1_db_path, backup_path)
            logger.info(f"v1 backup created: {backup_path}")

        # Step 2: Open v1 database (read-only)
        v1_conn = sqlite3.connect(f"file:{v1_db_path}?mode=ro", uri=True)
        v1_conn.row_factory = sqlite3.Row

        # Step 3: Initialize v2 store and HNSW index
        store = UnifiedSQLiteStore(self._config.store)
        store.initialize()
        hnsw = HNSWVectorIndex(self._config.hnsw)

        # Step 4: Read and migrate entries
        cursor = v1_conn.execute("SELECT COUNT(*) FROM memories")
        total = cursor.fetchone()[0]
        logger.info(f"Migrating {total} entries from v1 to v2")

        migrated = 0
        skipped = 0
        errors = 0

        FOR row IN v1_conn.execute("SELECT * FROM memories"):
            TRY:
                record = self._v1_row_to_record(row)

                # Dedup check: skip if already exists in v2
                IF store.get(record.id) IS NOT None:
                    skipped += 1
                    CONTINUE

                # Save to v2 store
                store.save(record)

                # Index embedding if present
                IF record.embedding IS NOT None:
                    hnsw.add(record.id, record.embedding)

                migrated += 1

                # Progress callback
                IF progress_callback AND migrated % 1000 == 0:
                    progress_callback(migrated, total)

            EXCEPT Exception as e:
                errors += 1
                logger.warning(f"Migration error for row {row['id']}: {e}")

        # Step 5: Save HNSW index
        hnsw.save()
        store.close()
        v1_conn.close()

        report = {
            "total": total,
            "migrated": migrated,
            "skipped": skipped,
            "errors": errors,
            "v1_path": str(v1_db_path),
            "v2_path": str(self._config.store.db_path),
            "hnsw_path": str(self._config.hnsw.index_path),
            "backup_path": str(backup_path),
        }

        logger.info(f"Migration complete: {migrated} migrated, {skipped} skipped, {errors} errors")
        RETURN report

    FUNCTION _v1_row_to_record(row: sqlite3.Row) -> MemoryRecord:
        """Convert v1 schema row to MemoryRecord."""
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
            parent_id=row.get("parent_id"),
            importance=row["importance"],
            emotional_weight=row["emotional_weight"],
            adapter_source="living_memory",   # Migrated from v1
        )
```

---

## 7.2 — Runtime Integration: `core/sovereign/runtime_core.py` (Modification)

### Requirements
- Add AgentDB init alongside existing LivingMemory in `_init_memory_coordinator()`
- Register all 3 adapters
- ~15 lines of new code

### Pseudocode (patch)

```
# In _init_memory_coordinator() method, after LivingMemoryCore init:

# ── AgentDB Integration (V3 Memory Unification) ──
TRY:
    FROM core.memory IMPORT AgentDB, AgentDBConfig
    FROM core.memory.adapters IMPORT (
        LivingMemoryAdapter,
        ExperienceLedgerAdapter,
        PatternMemoryAdapter,
    )

    self._agent_db = AgentDB(AgentDBConfig())

    # Register adapters for backward-compatible search
    self._agent_db.register_adapter(
        "living_memory",
        LivingMemoryAdapter(self._living_memory),
    )
    self._agent_db.register_adapter(
        "experience_ledger",
        ExperienceLedgerAdapter(self._evidence_path),
    )
    self._agent_db.register_adapter(
        "pattern_memory",
        PatternMemoryAdapter(),
    )

    logger.info("AgentDB initialized with 3 adapters")
EXCEPT ImportError:
    self._agent_db = None
    logger.info("AgentDB not available (hnswlib not installed?)")
```

---

## 7.3 — Memory Coordinator Registration (Modification)

### Requirements
- Register AgentDB as an additional state provider in `MemoryCoordinator.__init__`
- ~10 lines change

### Pseudocode (patch)

```
# In MemoryCoordinator.__init__, after existing state_providers setup:

FUNCTION register_agent_db(self, agent_db) -> None:
    """Register AgentDB for checkpoint/restore."""
    self._agent_db = agent_db
    logger.info("MemoryCoordinator: AgentDB registered")

# In save_state():
    IF self._agent_db IS NOT None:
        self._agent_db.save()

# In restore_state():
    # AgentDB auto-loads on init — no explicit restore needed
```

---

## 7.4 — API Endpoint: `core/sovereign/api.py` (Modification)

### Requirements
- Add `/memory/search` POST endpoint
- Accept query_text and optional embedding
- Return SearchResult list as JSON
- ~25 lines

### Pseudocode (patch)

```
@app.post("/memory/search")
ASYNC FUNCTION memory_search(request: MemorySearchRequest) -> MemorySearchResponse:
    """Unified memory search via AgentDB."""
    IF runtime._agent_db IS None:
        RAISE HTTPException(503, "AgentDB not initialized")

    embedding = None
    IF request.embedding IS NOT None:
        embedding = np.array(request.embedding, dtype=np.float32)

    results = runtime._agent_db.search(
        query_text=request.query,
        query_embedding=embedding,
        options=QueryOptions(
            top_k=request.top_k OR 10,
            min_ihsan=request.min_ihsan OR 0.0,
        ),
    )

    RETURN MemorySearchResponse(
        results=[
            {
                "id": r.record.id,
                "content": r.record.content,
                "score": r.score,
                "memory_type": r.record.memory_type.value,
                "adapter_source": r.record.adapter_source,
            }
            FOR r IN results
        ],
        count=len(results),
    )
```

### TDD Anchors — Migration

```
TEST test_migrate_v1_to_v2(tmp_path):
    # Create a v1 database with test data
    v1_path = tmp_path / "v1.db"
    create_v1_test_db(v1_path, num_entries=50)

    migrator = MemoryMigrator(AgentDBConfig(
        hnsw=HNSWConfig(index_path=tmp_path/"test.index"),
        store=StoreConfig(db_path=tmp_path/"v2.db"),
    ))
    report = migrator.migrate_from_v1(v1_path)

    ASSERT report["migrated"] == 50
    ASSERT report["errors"] == 0
    ASSERT (tmp_path / "v1.v1.bak").exists()  # Backup created

TEST test_migrate_idempotent(tmp_path):
    # Run migration twice — second time should skip all
    v1_path = tmp_path / "v1.db"
    create_v1_test_db(v1_path, num_entries=10)
    config = AgentDBConfig(...)
    migrator = MemoryMigrator(config)

    report1 = migrator.migrate_from_v1(v1_path)
    ASSERT report1["migrated"] == 10

    report2 = migrator.migrate_from_v1(v1_path)
    ASSERT report2["migrated"] == 0
    ASSERT report2["skipped"] == 10

TEST test_migrate_preserves_embeddings(tmp_path):
    v1_path = tmp_path / "v1.db"
    create_v1_test_db(v1_path, num_entries=5, with_embeddings=True)
    migrator = MemoryMigrator(config)
    migrator.migrate_from_v1(v1_path)

    store = UnifiedSQLiteStore(config.store)
    store.initialize()
    ids = store.list_ids()
    FOR id IN ids:
        record = store.get(id)
        ASSERT record.embedding IS NOT None
        ASSERT record.embedding.shape == (768,)
    store.close()

TEST test_migrate_creates_backup(tmp_path):
    v1_path = tmp_path / "v1.db"
    create_v1_test_db(v1_path, num_entries=1)
    migrator = MemoryMigrator(config)
    migrator.migrate_from_v1(v1_path)
    ASSERT (tmp_path / "v1.v1.bak").exists()
    # Original still intact
    ASSERT v1_path.exists()

@pytest.mark.slow
TEST test_performance_linear_vs_hnsw(tmp_path):
    """Benchmark: prove HNSW is faster than linear scan at scale."""
    N = 10_000
    dim = 768
    config = AgentDBConfig(...)
    db = AgentDB(config)

    # Insert N random vectors
    FOR i IN range(N):
        emb = np.random.randn(dim).astype(np.float32)
        db.store(f"entry_{i}", MemoryType.SEMANTIC, embedding=emb)

    query = np.random.randn(dim).astype(np.float32)

    # Benchmark HNSW search
    start = time.perf_counter()
    FOR _ IN range(100):
        results = db.search(query_embedding=query, options=QueryOptions(top_k=10))
    hnsw_elapsed = (time.perf_counter() - start) / 100

    logger.info(f"HNSW search: {hnsw_elapsed*1000:.3f}ms (N={N})")
    ASSERT hnsw_elapsed < 0.001  # Target: < 1ms
    ASSERT len(results) == 10
    db.close()
```

### Dependency Addition: `pyproject.toml`

```toml
# Add to [project.dependencies]:
"hnswlib>=0.8.0",
```

---

## Summary — All Files

| Phase | File | Lines | Status |
|-------|------|-------|--------|
| 1 | `core/memory/types.py` | ~100 | Spec complete |
| 1 | `core/memory/config.py` | ~80 | Spec complete |
| 2 | `core/memory/hnsw_index.py` | ~250 | Spec complete |
| 3 | `core/memory/unified_store.py` | ~400 | Spec complete |
| 4 | `core/memory/hybrid_query.py` | ~300 | Spec complete |
| 5 | `core/memory/agent_db.py` | ~350 | Spec complete |
| 5 | `core/memory/__init__.py` | ~30 | Spec complete |
| 6 | `core/memory/adapters/living_memory.py` | ~150 | Spec complete |
| 6 | `core/memory/adapters/experience_ledger.py` | ~120 | Spec complete |
| 6 | `core/memory/adapters/pattern_memory.py` | ~100 | Spec complete |
| 6 | `core/memory/adapters/__init__.py` | ~10 | Spec complete |
| 7 | `core/memory/migrator.py` | ~250 | Spec complete |
| 7 | `core/sovereign/runtime_core.py` (patch) | +15 | Spec complete |
| 7 | `core/sovereign/memory_coordinator.py` (patch) | +10 | Spec complete |
| 7 | `core/sovereign/api.py` (patch) | +25 | Spec complete |
| 7 | `pyproject.toml` (patch) | +1 | Spec complete |
| **TOTAL** | **12 new + 4 modified** | **~2,400** | |
