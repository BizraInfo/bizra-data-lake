# Phase 1: Foundation Types and Configuration

> ADR-006 / ADR-009 | V3 Memory Unification
> Standing on Giants: Hipp (SQLite) · Johnson (hnswlib) · Shannon (SNR thresholds)

## 1.1 — `core/memory/types.py`

### Requirements
- Define `MemoryRecord` as the unified data type across all memory tiers
- Map 1:1 from existing `MemoryEntry` (core/living_memory/core.py:62) but decoupled
- Define `SearchResult` wrapping a record with score metadata
- Define `QueryOptions` for configuring search behavior
- All IDs are content-addressable via `hex_digest(content)`

### Pseudocode

```
IMPORT dataclass, field FROM dataclasses
IMPORT datetime, timezone FROM datetime
IMPORT Enum FROM enum
IMPORT Optional, List, Set, Dict, Any FROM typing
IMPORT numpy as np

# ── Enums ──

CLASS MemoryType(str, Enum):
    EPISODIC   = "episodic"
    SEMANTIC   = "semantic"
    PROCEDURAL = "procedural"
    WORKING    = "working"
    PROSPECTIVE = "prospective"

CLASS MemoryState(str, Enum):
    ACTIVE       = "active"
    CONSOLIDATING = "consolidating"
    ARCHIVED     = "archived"
    DECAYING     = "decaying"
    DELETED      = "deleted"

# ── Core Record ──

@dataclass
CLASS MemoryRecord:
    """Unified memory record — single type across all AgentDB tiers."""

    id: str                          # hex_digest(content) — content-addressable
    content: str
    memory_type: MemoryType
    embedding: Optional[np.ndarray] = None   # 768-dim float32

    # Timestamps
    created_at: datetime   = NOW_UTC
    last_accessed: datetime = NOW_UTC
    access_count: int = 0

    # Quality scores (from constants.py)
    ihsan_score: float = 1.0
    snr_score: float   = 1.0
    confidence: float  = 1.0

    # State
    state: MemoryState = MemoryState.ACTIVE
    source: str = "unknown"

    # Graph relationships
    related_ids: Set[str] = EMPTY_SET
    parent_id: Optional[str] = None

    # Decay / salience
    importance: float       = 1.0
    emotional_weight: float = 0.5

    # Provenance (new in v2)
    adapter_source: Optional[str] = None   # "living_memory" | "sel" | "pattern" | "direct"

    FUNCTION to_dict() -> Dict[str, Any]:
        RETURN {all fields serialized, embedding excluded}

    @classmethod
    FUNCTION from_dict(data: Dict) -> MemoryRecord:
        RETURN MemoryRecord(**mapped fields with defaults)

# ── Search Types ──

@dataclass
CLASS SearchResult:
    """A memory record with search relevance metadata."""
    record: MemoryRecord
    score: float              # Composite score from hybrid query
    vector_score: float = 0.0 # HNSW cosine similarity component
    keyword_score: float = 0.0 # FTS5 match component
    recency_score: float = 0.0
    importance_score: float = 0.0
    graph_score: float = 0.0

@dataclass
CLASS QueryOptions:
    """Configuration for a search query."""
    top_k: int = 10
    min_score: float = 0.0
    memory_types: Optional[List[MemoryType]] = None  # Filter by type
    states: Optional[List[MemoryState]] = None        # Filter by state
    min_ihsan: float = 0.0                             # Quality gate
    include_embeddings: bool = False                    # Omit by default for speed
    source_filter: Optional[str] = None                # Filter by adapter source
```

### TDD Anchors

```
TEST test_memory_record_create():
    record = MemoryRecord(id="abc123", content="test", memory_type=MemoryType.SEMANTIC)
    ASSERT record.state == MemoryState.ACTIVE
    ASSERT record.ihsan_score == 1.0
    ASSERT record.adapter_source IS None

TEST test_memory_record_roundtrip():
    record = MemoryRecord(id="abc", content="hello", memory_type=MemoryType.EPISODIC)
    d = record.to_dict()
    restored = MemoryRecord.from_dict(d)
    ASSERT restored.id == record.id
    ASSERT restored.content == record.content
    ASSERT restored.memory_type == record.memory_type

TEST test_search_result_scores():
    record = MemoryRecord(id="x", content="y", memory_type=MemoryType.WORKING)
    result = SearchResult(record=record, score=0.92, vector_score=0.95)
    ASSERT result.score == 0.92
    ASSERT result.vector_score == 0.95

TEST test_query_options_defaults():
    opts = QueryOptions()
    ASSERT opts.top_k == 10
    ASSERT opts.min_score == 0.0
    ASSERT opts.include_embeddings IS False
```

---

## 1.2 — `core/memory/config.py`

### Requirements
- All thresholds imported from `core/integration/constants.py` (single source of truth)
- HNSW parameters from `.swarm/schema.sql` defaults (M=16, ef_construction=200, ef_search=100)
- Paths auto-resolve via `BIZRA_DATA_LAKE_ROOT` env var
- Embedding dimension = 768 (nomic-embed-text-v1.5)

### Pseudocode

```
IMPORT os FROM os
IMPORT Path FROM pathlib
IMPORT constants FROM core.integration.constants

# ── Path Resolution ──

FUNCTION _resolve_root() -> Path:
    root = os.getenv("BIZRA_DATA_LAKE_ROOT")
    IF root:
        RETURN Path(root)
    # Fallback: walk up from this file to find pyproject.toml
    RETURN Path(__file__).resolve().parent.parent.parent

PROJECT_ROOT = _resolve_root()

# ── HNSW Configuration ──

@dataclass(frozen=True)
CLASS HNSWConfig:
    dim: int = 768                    # nomic-embed-text-v1.5 output dimension
    space: str = "cosine"             # Distance metric
    M: int = 16                       # Max connections per layer
    ef_construction: int = 200        # Build-time search width
    ef_search: int = 100              # Query-time search width
    max_elements: int = 1_000_000     # Pre-allocated capacity
    index_path: Path = PROJECT_ROOT / ".agentdb" / "hnsw.index"

# ── SQLite v2 Configuration ──

@dataclass(frozen=True)
CLASS StoreConfig:
    db_path: Path = PROJECT_ROOT / ".agentdb" / "unified.db"
    schema_version: int = 2
    busy_timeout_ms: int = 5000       # PRAGMA busy_timeout (SAPE gap fix)
    wal_mode: bool = True

# ── Quality Thresholds (from constants.py — NEVER hardcode) ──

@dataclass(frozen=True)
CLASS QualityConfig:
    ihsan_threshold: float = constants.UNIFIED_IHSAN_THRESHOLD    # 0.95
    snr_threshold: float = constants.UNIFIED_SNR_THRESHOLD        # 0.85
    ihsan_strict: float = constants.STRICT_IHSAN_THRESHOLD        # 0.99
    snr_elite: float = constants.SNR_THRESHOLD_T0_ELITE           # 0.98

# ── Hybrid Query Weights ──

@dataclass(frozen=True)
CLASS QueryWeights:
    vector: float = 0.40
    keyword: float = 0.15
    recency: float = 0.20
    importance: float = 0.15
    graph: float = 0.10
    # Invariant: sum == 1.0

    FUNCTION __post_init__():
        total = self.vector + self.keyword + self.recency + self.importance + self.graph
        ASSERT abs(total - 1.0) < 1e-6, f"Weights must sum to 1.0, got {total}"

# ── Top-Level Config ──

@dataclass
CLASS AgentDBConfig:
    hnsw: HNSWConfig = HNSWConfig()
    store: StoreConfig = StoreConfig()
    quality: QualityConfig = QualityConfig()
    weights: QueryWeights = QueryWeights()
```

### TDD Anchors

```
TEST test_query_weights_sum_to_one():
    w = QueryWeights()
    ASSERT abs(w.vector + w.keyword + w.recency + w.importance + w.graph - 1.0) < 1e-6

TEST test_query_weights_reject_invalid():
    WITH pytest.raises(AssertionError):
        QueryWeights(vector=0.5, keyword=0.5, recency=0.5, importance=0.0, graph=0.0)

TEST test_quality_from_constants():
    q = QualityConfig()
    ASSERT q.ihsan_threshold == 0.95
    ASSERT q.snr_threshold == 0.85

TEST test_paths_resolve():
    cfg = AgentDBConfig()
    ASSERT cfg.hnsw.index_path.name == "hnsw.index"
    ASSERT cfg.store.db_path.name == "unified.db"
```

---

## Edge Cases

1. `BIZRA_DATA_LAKE_ROOT` not set — fallback to pyproject.toml discovery
2. `QueryWeights` with negative values — should raise ValueError
3. `MemoryRecord.from_dict` with missing fields — use defaults, never crash
4. Embedding as `None` — valid for text-only memories (no vector search)
5. `hex_digest` collision (astronomically unlikely) — content-addressable IDs are SHA-256

## Dependencies

- `numpy` (existing)
- `core.integration.constants` (existing)
- `core.proof_engine.canonical.hex_digest` (existing)
