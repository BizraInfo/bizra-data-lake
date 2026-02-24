# AgentDB V3 — Unified Memory with HNSW Indexing

> ADR-006 (Unified Memory Service) + ADR-009 (Hybrid Memory Backend)

## Quick Start

```python
from core.memory import AgentDB
from core.memory.config import MemoryConfig
from pathlib import Path

# Initialize
config = MemoryConfig(data_dir=Path("sovereign_state/memory"))
db = AgentDB(config)
db.initialize()

# Store
record = db.store("The OODA loop runs every 30 seconds", tags=["proactive", "ooda"])

# Search (hybrid: vector + keyword + recency + importance)
results = db.search("OODA cycle frequency", top_k=5)
for r in results:
    print(f"  [{r.score:.3f}] {r.record.content[:80]}")

# Retrieve by ID
rec = db.retrieve(record.id)

# Forget (soft delete by default, hard=True for permanent)
db.forget(record.id)
db.forget(record.id, hard=True)  # permanent removal

# Persist HNSW index to disk
db.save()

# Stats
print(db.stats())
# {'total_records': 42, 'total_vectors': 42, 'hnsw_backend': 'hnswlib'}
```

## Architecture

```
                    +-------------------------+
                    |       AgentDB           |  <-- Single facade
                    |  store() / search()     |
                    |  retrieve() / forget()  |
                    +------------+------------+
                                 |
              +------------------+------------------+
              |                  |                  |
     +--------v-----+  +--------v-------+  +-------v--------+
     | HNSW Index   |  | SQLite v2      |  |  Adapters      |
     | (hnswlib)    |  | + FTS5         |  |                |
     | cosine       |  | keyword search |  | LivingMemory   |
     | dim=768      |  | metadata       |  | SEL (read-only)|
     | M=16         |  | soft/hard del  |  | PatternMemory  |
     +--------------+  +----------------+  +----------------+
              |                  |                  |
              +------------------+------------------+
                                 |
                    +------------v------------+
                    |  Hybrid Query Engine    |
                    |  0.40 x vector          |
                    |  0.15 x keyword         |
                    |  0.20 x recency         |
                    |  0.15 x importance      |
                    |  0.10 x graph           |
                    +-------------------------+
```

## Package Structure

```
core/memory/
  __init__.py              # Public exports
  agent_db.py              # AgentDB facade (main entry point)
  hnsw_index.py            # HNSW vector index (hnswlib with numpy fallback)
  unified_store.py         # SQLite v2 with FTS5 keyword search
  hybrid_query.py          # Score fusion engine (5-signal)
  types.py                 # MemoryRecord, SearchResult, QueryOptions, MemoryKind
  config.py                # MemoryConfig, HNSWConfig
  migrator.py              # SQLite v1 -> v2 migration
  adapters/
    __init__.py
    living_memory.py       # Wraps existing LivingMemoryCore
    experience_ledger.py   # Read-only SEL adapter
    pattern_memory.py      # Rust PatternMemory bridge (optional PyO3)
```

## Configuration

```python
from core.memory.config import MemoryConfig, HNSWConfig

config = MemoryConfig(
    # Paths
    data_dir=Path("sovereign_state/memory"),  # All files go here
    sqlite_filename="agent_db.sqlite",        # SQLite database
    hnsw_filename="hnsw.index",               # HNSW index file

    # HNSW parameters (from .swarm/schema.sql defaults)
    hnsw=HNSWConfig(
        dimensions=768,           # Embedding dimension
        max_elements=1_000_000,   # Max vectors
        m=16,                     # HNSW connections per node
        ef_construction=200,      # Construction-time quality
        ef_search=100,            # Search-time quality
        metric="cosine",          # Distance metric
    ),

    # Quality gates (from core/integration/constants.py)
    ihsan_threshold=0.95,         # UNIFIED_IHSAN_THRESHOLD
    snr_threshold=0.85,           # UNIFIED_SNR_THRESHOLD

    # SQLite
    sqlite_busy_timeout_ms=5000,  # Busy retry timeout
    sqlite_wal_mode=True,         # WAL for concurrent reads

    # Hybrid search weights (must sum to 1.0)
    weight_vector=0.40,
    weight_keyword=0.15,
    weight_recency=0.20,
    weight_importance=0.15,
    weight_graph=0.10,

    # Migration (optional)
    living_memory_db=None,        # Path to LivingMemory SQLite for migration
)
```

## API Reference

### `AgentDB`

| Method | Signature | Description |
|--------|-----------|-------------|
| `initialize()` | `() -> None` | Initialize subsystems. **Must call before use.** |
| `store()` | `(content, kind?, embedding?, importance?, source?, tags?, metadata?) -> MemoryRecord` | Store content. Content-addressable (deduplicates). |
| `search()` | `(query, top_k?, embedding?, kind?, tags?, min_importance?) -> List[SearchResult]` | Hybrid search across vector + keyword + recency + importance. |
| `retrieve()` | `(record_id) -> Optional[MemoryRecord]` | Get by ID. Returns `None` if not found or hard-deleted. |
| `forget()` | `(record_id, hard=False) -> bool` | Soft-delete (default) or hard-delete. |
| `save()` | `() -> None` | Persist HNSW index to disk. SQLite auto-commits. |
| `stats()` | `() -> dict` | Total records, vectors, HNSW backend info. |

### `MemoryRecord`

```python
@dataclass
class MemoryRecord:
    id: str                    # Content-addressable hex digest
    content: str               # The stored text
    kind: MemoryKind           # SEMANTIC, EPISODIC, PROCEDURAL, META
    importance: float          # 0.0-1.0
    source: str                # "agent", "user", "system"
    tags: List[str]            # Searchable tags
    metadata: dict             # Arbitrary key-value pairs
    created_at: datetime
    updated_at: datetime
```

### `SearchResult`

```python
@dataclass
class SearchResult:
    record: MemoryRecord
    score: float               # 0.0-1.0 (higher = better match)
    components: dict           # Per-signal scores for debugging
```

## Performance

| Scale | Linear Scan | HNSW (hnswlib) | Speedup |
|-------|-------------|----------------|---------|
| 1,000 | ~5ms | ~0.09ms | **55x** |
| 10,000 | ~50ms | ~1.3ms | **38x** |
| 100,000 | ~500ms | ~1.75ms | **285x** |

*Benchmarks from `tests/core/memory/test_performance.py` on WSL2/RTX 4090.*

HNSW requires `hnswlib` (C++ bindings). Falls back to numpy cosine scan if unavailable.

```bash
pip install hnswlib>=0.8.0   # Required for production performance
```

## Integration with Existing Systems

### Living Memory Migration

```python
from core.memory.migrator import migrate_living_memory

migrate_living_memory(
    source_db=Path("04_GOLD/living_memory.db"),  # v1 SQLite
    target=agent_db,                              # AgentDB instance
    batch_size=1000,
)
```

### Runtime Core

AgentDB is initialized in `core/sovereign/runtime_core.py` alongside LivingMemory:

```python
# In RuntimeCore._init_memory_coordinator()
self.agent_db = AgentDB(MemoryConfig(data_dir=self.state_dir / "memory"))
self.agent_db.initialize()
```

### SEL Adapter (Read-Only)

The Experience Ledger adapter is strictly read-only to preserve hash-chain integrity:

```python
from core.memory.adapters.experience_ledger import ExperienceLedgerAdapter

adapter = ExperienceLedgerAdapter(sel_path=Path("sovereign_state/sel.db"))
# adapter.store() raises ReadOnlyError
entries = adapter.search("recent decisions", top_k=10)
```

## Tests

```bash
# All memory tests
pytest tests/core/memory/ -v

# Performance benchmarks (requires hnswlib)
source .venv-linux/bin/activate
pytest tests/core/memory/test_performance.py -v

# Full CI-safe suite
pytest tests/ -m "not requires_ollama and not requires_gpu and not slow"
```

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| `hnswlib` over FAISS/ChromaDB | Minimal dependency, local-first, incremental add/remove |
| Content-addressable IDs | `hex_digest(content)` for automatic deduplication |
| Soft delete by default | Audit trail preservation; `hard=True` for permanent |
| FTS5 for keyword search | Built into SQLite, zero extra dependencies |
| 5-signal hybrid scoring | Balances semantic similarity, keyword match, recency, importance, graph |
| SEL adapter read-only | Hash-chain integrity MUST NOT be broken by writes |
| WAL mode + busy timeout | Concurrent read access without locking issues |
