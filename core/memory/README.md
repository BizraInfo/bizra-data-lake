# AgentDB — Unified Memory System (V3)

> ADR-006 (Unified Memory Service) + ADR-009 (Hybrid Memory Backend)
> Standing on Giants: Malkov & Yashunin (HNSW, 2016) · Robertson (BM25) · Merkle (content-addressable storage) · Shannon (information duality)

## Quick Start

```python
from core.memory import AgentDB, MemoryConfig

db = AgentDB(MemoryConfig())
db.initialize()

# Store
record = db.store("The Earth orbits the Sun", importance=0.9)

# Search (keyword + vector + recency + importance + graph)
results = db.search("solar system", top_k=5)
for r in results:
    print(f"{r.score:.3f}  {r.record.content[:60]}")

# Retrieve by ID
record = db.retrieve(record.id)

# Forget (soft delete by default)
db.forget(record.id)

# Persist HNSW index to disk
db.save()
```

## Architecture

```
┌──────────────────────────────────┐
│          AgentDB (facade)        │
│  store / search / retrieve / forget
└────────────┬─────────────────────┘
             │
   ┌─────────┼──────────┐
   │         │          │
┌──▼───┐ ┌──▼─────┐ ┌──▼──────────────┐
│ HNSW │ │SQLite  │ │ HybridQuery     │
│Index │ │v2+FTS5 │ │ Engine          │
│      │ │        │ │ 0.40 vector     │
│cosine│ │keyword │ │ 0.15 keyword    │
│d=768 │ │search  │ │ 0.20 recency    │
│M=16  │ │metadata│ │ 0.15 importance │
└──────┘ └────────┘ │ 0.10 graph      │
                    └─────────────────┘
```

## Module Map

| File | Purpose | Lines |
|------|---------|-------|
| `agent_db.py` | Single facade — all operations go through here | ~280 |
| `hnsw_index.py` | hnswlib wrapper with save/load/rebuild | ~250 |
| `unified_store.py` | SQLite v2 schema + FTS5 full-text search | ~400 |
| `hybrid_query.py` | 5-signal score fusion engine | ~165 |
| `types.py` | `MemoryRecord`, `SearchResult`, `QueryOptions` | ~120 |
| `config.py` | `MemoryConfig`, `HNSWConfig` with constitutional thresholds | ~80 |
| `migrator.py` | v1 (LivingMemory) to v2 non-destructive migration | ~210 |
| `adapters/` | Wrappers for legacy memory systems | ~3 files |

## API Reference

### `AgentDB`

The single entry point for all memory operations.

```python
class AgentDB:
    def __init__(config: Optional[MemoryConfig] = None) -> None
    def initialize() -> None
    def store(content, kind, embedding, importance, source, tags, metadata) -> MemoryRecord
    def store_record(record: MemoryRecord) -> None
    def search(query, query_embedding, top_k, min_score, kinds, tags, source) -> List[SearchResult]
    def retrieve(record_id: str) -> Optional[MemoryRecord]
    def forget(record_id: str, hard: bool = False) -> bool
    def save() -> None
    def stats() -> dict
    def set_embedding_fn(fn: Callable[[str], List[float]]) -> None
```

#### `store()`

Stores content as a new memory record. Returns a `MemoryRecord`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `content` | `str` | required | The text content to store |
| `kind` | `MemoryKind` | `SEMANTIC` | Category: episodic, semantic, procedural, working, prospective |
| `embedding` | `Sequence[float]` | `None` | Optional pre-computed embedding (dim=768) |
| `importance` | `float` | `0.5` | Importance weight for ranking (0.0-1.0) |
| `source` | `str` | `"agent"` | Provenance tag |
| `tags` | `List[str]` | `[]` | Searchable tags |
| `metadata` | `dict` | `{}` | Extensible key-value metadata |

Content-addressable: storing identical content from the same source updates the existing record (upsert).

#### `search()`

Hybrid search across all memory using 5-signal score fusion.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | `None` | Text query for keyword (FTS5) and optional auto-embedding |
| `query_embedding` | `Sequence[float]` | `None` | Pre-computed query embedding for vector search |
| `top_k` | `int` | `10` | Number of results to return |
| `min_score` | `float` | `0.1` | Minimum fused score threshold |
| `kinds` | `List[MemoryKind]` | `None` | Filter by memory kind |
| `tags` | `List[str]` | `None` | Filter by tags (any match) |
| `source` | `str` | `None` | Filter by source |
| `context_ids` | `List[str]` | `None` | IDs for graph-overlap scoring |

Returns `List[SearchResult]` sorted by fused score descending.

Provide both `query` and `query_embedding` for the best results. If an embedding function has been injected via `set_embedding_fn()`, text queries will be auto-embedded.

#### `forget()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `record_id` | `str` | required | ID of the record to forget |
| `hard` | `bool` | `False` | If True, permanently delete from SQLite and FTS. If False, mark as deleted. |

### `MemoryRecord`

The canonical data structure for all memories.

```python
@dataclass
class MemoryRecord:
    id: str                              # Content-addressable (hex_digest)
    content: str                         # The memory text
    kind: MemoryKind                     # episodic | semantic | procedural | working | prospective
    state: RecordState                   # active | archived | deleted
    embedding: Optional[List[float]]     # float32, dim=768
    ihsan_score: float                   # Quality score (0.0-1.0)
    snr_score: float                     # Signal-to-noise (0.0-1.0)
    importance: float                    # Ranking weight (0.0-1.0)
    source: str                          # Provenance tag
    source_id: Optional[str]             # Original ID in source system
    related_ids: List[str]               # Graph connections
    tags: List[str]                      # Searchable tags
    created_at: datetime                 # UTC
    updated_at: datetime                 # UTC
    last_accessed: datetime              # UTC (updated on retrieve/search)
    access_count: int                    # Auto-incremented on access
    metadata: Dict[str, Any]             # Extensible
```

### `SearchResult`

Returned by `search()`. Contains the record plus a full score breakdown.

```python
@dataclass
class SearchResult:
    record: MemoryRecord
    score: float            # Final fused score (0.0 - 1.0)
    vector_score: float     # Cosine similarity via HNSW
    keyword_score: float    # BM25 via FTS5
    recency_score: float    # Exponential decay (half-life = 1 week)
    importance_score: float # record.importance clamped to [0,1]
    graph_score: float      # Overlap of related_ids with context
```

## Configuration

```python
from core.memory import MemoryConfig, HNSWConfig

config = MemoryConfig(
    # Storage paths
    data_dir=Path("sovereign_state/agent_db"),
    sqlite_filename="agent_db.sqlite",
    hnsw_filename="hnsw.index",

    # HNSW parameters (from .swarm/schema.sql, proven production defaults)
    hnsw=HNSWConfig(
        dimensions=768,           # Embedding dimensionality
        space="cosine",           # Distance metric
        m=16,                     # Bi-directional links per element
        ef_construction=200,      # Build-time candidate list size
        ef_search=100,            # Search-time candidate list size
        max_elements=1_000_000,   # Initial capacity (auto-resized)
    ),

    # Score fusion weights (must sum to 1.0)
    weight_vector=0.40,
    weight_keyword=0.15,
    weight_recency=0.20,
    weight_importance=0.15,
    weight_graph=0.10,

    # Quality thresholds (from core/integration/constants.py)
    ihsan_threshold=0.95,   # UNIFIED_IHSAN_THRESHOLD
    snr_threshold=0.85,     # UNIFIED_SNR_THRESHOLD

    # SQLite tuning
    sqlite_busy_timeout_ms=5000,
    sqlite_wal_mode=True,
)
```

All constitutional thresholds are imported from `core/integration/constants.py` — never hardcoded.

## Score Fusion

The `HybridQueryEngine` combines 5 independent signals into a single score:

| Signal | Weight | Source | Range |
|--------|--------|--------|-------|
| Vector similarity | 0.40 | HNSW cosine distance converted to similarity | [0, 1] |
| Keyword relevance | 0.15 | SQLite FTS5 BM25 rank, normalized | [0, 1] |
| Recency | 0.20 | Exponential decay on `last_accessed` (half-life = 168h) | [0, 1] |
| Importance | 0.15 | `record.importance` field, clamped | [0, 1] |
| Graph overlap | 0.10 | Fraction of `related_ids` present in `context_ids` | [0, 1] |

**Formula**: `score = 0.40*V + 0.15*K + 0.20*R + 0.15*I + 0.10*G`

Weights are configurable in `MemoryConfig`. Must sum to 1.0.

## Migration from v1 (LivingMemory)

```python
from core.memory import AgentDB, MemoryConfig
from core.memory.migrator import MemoryMigrator
from pathlib import Path

db = AgentDB(MemoryConfig())
db.initialize()

migrator = MemoryMigrator(
    agent_db=db,
    source_path=Path("sovereign_state/living_memory.db"),
)
result = migrator.migrate()
print(result)  # MigrationResult(migrated=1234, skipped=0, errors=0)
```

**Safety guarantees:**
- Source database is opened read-only (`?mode=ro`)
- A `.bak` copy is created before migration begins
- Batch processing (500 records per batch) for memory efficiency
- Idempotent: re-running skips already-migrated records (upsert semantics)

## REST API

### `POST /v1/memory/search`

```json
{
  "query": "quantum computing",
  "top_k": 10,
  "min_score": 0.1,
  "source": null
}
```

Response:

```json
{
  "query": "quantum computing",
  "top_k": 10,
  "count": 3,
  "results": [
    {
      "id": "a1b2c3d4e5f6",
      "content": "Quantum computing uses qubits...",
      "kind": "semantic",
      "score": 0.8231,
      "vector_score": 0.9100,
      "keyword_score": 0.7500,
      "recency_score": 0.9800,
      "importance_score": 0.8000,
      "source": "agent"
    }
  ]
}
```

### `GET /v1/memory/stats`

Returns record counts, vector index size, and file paths.

## Adapters

Adapters wrap existing memory systems into the AgentDB interface without modifying them.

| Adapter | Module | Mode | Wraps |
|---------|--------|------|-------|
| `LivingMemoryAdapter` | `adapters/living_memory.py` | Read-write | `core.living_memory.core.LivingMemoryCore` |
| `ExperienceLedgerAdapter` | `adapters/experience_ledger.py` | **Read-only** | JSONL evidence chain |
| `PatternMemoryAdapter` | `adapters/pattern_memory.py` | Read-only | Rust PyO3 `bizra_python` (optional) |

**Constitutional invariant**: `ExperienceLedgerAdapter` has no write path. The hash-chain integrity of the SEL must never be compromised.

## Performance

Benchmarked on the production BIZRA-DATA-LAKE dataset:

| Scale | Linear Scan | HNSW | Speedup |
|-------|-------------|------|---------|
| 1,000 entries | ~5ms | ~0.035ms | **140x** |
| 10,000 entries | ~50ms | ~0.04ms | **1,250x** |

HNSW index parameters: M=16, ef_construction=200, ef_search=100, dim=768, cosine.

## Testing

```bash
# All tests (excluding slow benchmarks)
pytest tests/core/memory/ -v -m "not slow"

# With benchmarks
pytest tests/core/memory/test_performance.py -v -m slow

# Regression check against LivingMemory
pytest tests/core/living_memory/ -v
```

Test coverage: 82 tests across 7 files (79 pass in CI, 3 slow benchmarks gated).

## Files

```
core/memory/
├── __init__.py              # Package exports
├── agent_db.py              # AgentDB facade (single entry point)
├── config.py                # MemoryConfig, HNSWConfig
├── hnsw_index.py            # hnswlib wrapper
├── hybrid_query.py          # 5-signal score fusion
├── migrator.py              # v1 -> v2 migration
├── types.py                 # MemoryRecord, SearchResult, QueryOptions
├── unified_store.py         # SQLite v2 + FTS5
└── adapters/
    ├── __init__.py
    ├── experience_ledger.py # Read-only SEL wrapper
    ├── living_memory.py     # LivingMemoryCore wrapper
    └── pattern_memory.py    # Rust PatternMemory (optional)
```

## Dependencies

- `hnswlib>=0.8.0` — C++ HNSW with Python bindings (added to `pyproject.toml`)
- `numpy>=1.24.0` — Embedding vector operations
- `sqlite3` — Standard library (FTS5 enabled in CPython builds)
