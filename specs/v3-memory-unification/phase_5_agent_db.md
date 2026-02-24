# Phase 5: AgentDB Facade

> ADR-006 | Unified Memory Service — Single Entry Point
> Standing on Giants: Gamma et al. (Facade pattern, GoF 1994) · Lamport (distributed state)

## 5.1 — `core/memory/agent_db.py`

### Requirements
- Single entry point for ALL memory operations: store, search, retrieve, forget
- Wires together: HNSWVectorIndex + UnifiedSQLiteStore + HybridQueryEngine
- Auto-initializes on first use (lazy init)
- Registers adapters for backward-compatible access to legacy systems
- Content-addressable dedup at storage layer
- Ihsan quality gate on store (reject below threshold)
- Persistence: save/load index + DB together

### Public API

```
AgentDB.store(content, memory_type, embedding, ...)  -> str (record ID)
AgentDB.search(query_text, query_embedding, ...)      -> List[SearchResult]
AgentDB.get(id)                                        -> Optional[MemoryRecord]
AgentDB.delete(id)                                     -> bool
AgentDB.count()                                        -> int
AgentDB.save()                                         -> None
AgentDB.close()                                        -> None
AgentDB.register_adapter(name, adapter)                -> None
AgentDB.search_adapter(adapter_name, query, ...)       -> List[SearchResult]
```

### Pseudocode

```
IMPORT logging FROM stdlib
IMPORT numpy as np
IMPORT Optional, List, Dict, Callable FROM typing
IMPORT MemoryRecord, MemoryType, MemoryState, SearchResult, QueryOptions FROM core.memory.types
IMPORT AgentDBConfig, QualityConfig FROM core.memory.config
IMPORT HNSWVectorIndex FROM core.memory.hnsw_index
IMPORT UnifiedSQLiteStore FROM core.memory.unified_store
IMPORT HybridQueryEngine FROM core.memory.hybrid_query
IMPORT hex_digest FROM core.proof_engine.canonical

logger = GET_LOGGER(__name__)

# ── Adapter Protocol ──

CLASS MemoryAdapter(Protocol):
    """Interface for legacy memory system adapters."""
    FUNCTION search(query: str, top_k: int = 10) -> List[MemoryRecord]: ...
    FUNCTION get(id: str) -> Optional[MemoryRecord]: ...
    PROPERTY name -> str: ...
    PROPERTY read_only -> bool: ...

# ── AgentDB Facade ──

CLASS AgentDB:
    """
    Unified memory facade — single entry point for all BIZRA memory operations.

    Wraps HNSW vector index, SQLite v2 store, and hybrid query engine.
    Supports registered adapters for backward-compatible access to
    LivingMemoryCore, ExperienceLedger, and PatternMemory.

    Usage:
        db = AgentDB()
        db.store("Important fact about BIZRA", MemoryType.SEMANTIC, embedding=vec)
        results = db.search(query_text="BIZRA facts", query_embedding=vec)
    """

    FUNCTION __init__(config: AgentDBConfig = AgentDBConfig()):
        self._config = config
        self._initialized = False

        # Core components (lazy init)
        self._vector_index: Optional[HNSWVectorIndex] = None
        self._store: Optional[UnifiedSQLiteStore] = None
        self._query_engine: Optional[HybridQueryEngine] = None

        # Adapters
        self._adapters: Dict[str, MemoryAdapter] = {}

    FUNCTION _ensure_init() -> None:
        """Lazy initialization — create components on first use."""
        IF self._initialized:
            RETURN

        # Initialize HNSW index
        self._vector_index = HNSWVectorIndex(self._config.hnsw)
        IF self._config.hnsw.index_path.exists():
            self._vector_index.load()

        # Initialize SQLite store
        self._store = UnifiedSQLiteStore(self._config.store)
        self._store.initialize()

        # Initialize query engine
        self._query_engine = HybridQueryEngine(
            vector_index=self._vector_index,
            store=self._store,
            weights=self._config.weights,
            quality=self._config.quality,
        )

        self._initialized = True
        logger.info(f"AgentDB initialized: {self._store.count()} records, "
                     f"{len(self._vector_index)} vectors, "
                     f"{len(self._adapters)} adapters")

    # ── Store ──

    FUNCTION store(
        content: str,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        embedding: Optional[np.ndarray] = None,
        ihsan_score: float = 1.0,
        snr_score: float = 1.0,
        importance: float = 1.0,
        source: str = "direct",
        related_ids: Optional[Set[str]] = None,
        parent_id: Optional[str] = None,
    ) -> str:
        """Store a memory. Returns content-addressable ID.

        Quality gate: rejects records with ihsan_score below threshold.
        Dedup: if content already exists (same hex_digest), overwrites.
        """
        self._ensure_init()

        # Quality gate (from constants.py)
        IF ihsan_score < self._config.quality.ihsan_threshold:
            logger.warning(f"Rejected: ihsan {ihsan_score} < {self._config.quality.ihsan_threshold}")
            RAISE ValueError(f"Ihsan score {ihsan_score} below threshold")

        record_id = hex_digest(content)

        record = MemoryRecord(
            id=record_id,
            content=content,
            memory_type=memory_type,
            embedding=embedding,
            ihsan_score=ihsan_score,
            snr_score=snr_score,
            importance=importance,
            source=source,
            related_ids=related_ids OR set(),
            parent_id=parent_id,
            adapter_source="direct",
        )

        # Persist to SQLite
        self._store.save(record)

        # Index embedding in HNSW
        IF embedding IS NOT None:
            self._vector_index.add(record_id, embedding)

        RETURN record_id

    # ── Search ──

    FUNCTION search(
        query_text: Optional[str] = None,
        query_embedding: Optional[np.ndarray] = None,
        options: QueryOptions = QueryOptions(),
    ) -> List[SearchResult]:
        """Hybrid search across all memory tiers.

        Fuses vector similarity + keyword match + recency + importance + graph
        using configurable weights.
        """
        self._ensure_init()
        RETURN self._query_engine.search(query_text, query_embedding, options)

    # ── Get / Delete ──

    FUNCTION get(id: str) -> Optional[MemoryRecord]:
        """Retrieve a single record by ID. Updates access tracking."""
        self._ensure_init()
        record = self._store.get(id)
        IF record IS NOT None:
            self._store.touch(id)
        RETURN record

    FUNCTION delete(id: str) -> bool:
        """Delete a record from store and index."""
        self._ensure_init()
        self._vector_index.remove(id)
        RETURN self._store.delete(id)

    FUNCTION count() -> int:
        self._ensure_init()
        RETURN self._store.count()

    # ── Adapters ──

    FUNCTION register_adapter(name: str, adapter: MemoryAdapter) -> None:
        """Register a legacy memory system adapter."""
        self._adapters[name] = adapter
        logger.info(f"Adapter registered: {name} (read_only={adapter.read_only})")

    FUNCTION search_adapter(
        adapter_name: str,
        query: str,
        top_k: int = 10,
    ) -> List[MemoryRecord]:
        """Search through a specific adapter."""
        IF adapter_name NOT IN self._adapters:
            RAISE KeyError(f"Unknown adapter: {adapter_name}")
        RETURN self._adapters[adapter_name].search(query, top_k)

    # ── Persistence ──

    FUNCTION save() -> None:
        """Persist HNSW index and commit any pending store operations."""
        self._ensure_init()
        self._vector_index.save()
        logger.info(f"AgentDB saved: {self.count()} records, {len(self._vector_index)} vectors")

    FUNCTION close() -> None:
        """Save and close all resources."""
        IF self._initialized:
            self.save()
            self._store.close()
            self._initialized = False
            logger.info("AgentDB closed")

    # ── Context Manager ──

    FUNCTION __enter__() -> AgentDB:
        self._ensure_init()
        RETURN self

    FUNCTION __exit__() -> None:
        self.close()
```

## 5.2 — `core/memory/__init__.py`

### Pseudocode

```
"""
BIZRA AgentDB — Unified Memory Service (ADR-006)

Usage:
    from core.memory import AgentDB, MemoryRecord, MemoryType, SearchResult

    db = AgentDB()
    db.store("Important fact", MemoryType.SEMANTIC, embedding=vec)
    results = db.search(query_text="fact", query_embedding=vec)
"""

FROM core.memory.agent_db IMPORT AgentDB
FROM core.memory.types IMPORT (
    MemoryRecord, MemoryType, MemoryState,
    SearchResult, QueryOptions,
)
FROM core.memory.config IMPORT AgentDBConfig, HNSWConfig, StoreConfig, QueryWeights

__all__ = [
    "AgentDB", "AgentDBConfig",
    "MemoryRecord", "MemoryType", "MemoryState",
    "SearchResult", "QueryOptions",
    "HNSWConfig", "StoreConfig", "QueryWeights",
]
```

### TDD Anchors

```
TEST test_agent_db_store_and_search(tmp_path):
    config = AgentDBConfig(
        hnsw=HNSWConfig(dim=4, max_elements=100, index_path=tmp_path/"test.index"),
        store=StoreConfig(db_path=tmp_path/"test.db"),
    )
    db = AgentDB(config)
    id = db.store("hello world", MemoryType.SEMANTIC, embedding=np.array([1,0,0,0], dtype=np.float32))
    ASSERT id IS NOT None
    record = db.get(id)
    ASSERT record.content == "hello world"
    db.close()

TEST test_agent_db_dedup(tmp_path):
    db = AgentDB(config)
    id1 = db.store("same content", MemoryType.SEMANTIC)
    id2 = db.store("same content", MemoryType.SEMANTIC)
    ASSERT id1 == id2
    ASSERT db.count() == 1

TEST test_agent_db_quality_gate(tmp_path):
    db = AgentDB(config)
    WITH pytest.raises(ValueError):
        db.store("low quality", MemoryType.SEMANTIC, ihsan_score=0.50)

TEST test_agent_db_delete(tmp_path):
    db = AgentDB(config)
    id = db.store("to delete", MemoryType.WORKING)
    ASSERT db.delete(id) IS True
    ASSERT db.get(id) IS None
    ASSERT db.count() == 0

TEST test_agent_db_context_manager(tmp_path):
    WITH AgentDB(config) AS db:
        db.store("test", MemoryType.SEMANTIC)
        ASSERT db.count() == 1
    # After __exit__, db is closed

TEST test_agent_db_adapter_registration(tmp_path):
    db = AgentDB(config)
    mock_adapter = MockAdapter(name="living", read_only=False)
    db.register_adapter("living", mock_adapter)
    results = db.search_adapter("living", "test query")
    ASSERT mock_adapter.search_called IS True

TEST test_agent_db_save_reload(tmp_path):
    config = AgentDBConfig(...)
    db1 = AgentDB(config)
    db1.store("persistent", MemoryType.SEMANTIC, embedding=np.array([0.5]*4, dtype=np.float32))
    db1.close()
    # Reload
    db2 = AgentDB(config)
    ASSERT db2.count() == 1
    record = db2.get(hex_digest("persistent"))
    ASSERT record IS NOT None
    db2.close()
```

### Edge Cases

1. **Store without embedding** — record saved in SQLite, no HNSW entry (keyword-only searchable)
2. **Search with neither text nor embedding** — return empty list
3. **Double close** — idempotent, no error
4. **Adapter search for unknown name** — raises KeyError
5. **Ihsan below threshold** — raises ValueError, not stored
6. **Lazy init failure** — hnswlib not installed, falls back seamlessly
