# Phase 39: V3 Memory Unification — Specification

**ADR-006**: Unified Memory Service (AgentDB)
**ADR-009**: Hybrid Memory Backend (HNSW + SQLite v2 + FTS5)
**Date**: 2026-02-24
**Status**: Implementation 70% complete — specification formalizes remaining work

---

## 1. Problem Statement

BIZRA has 6+ memory subsystems that evolved independently across Phases 18-38:

| System | Location | Storage | Search |
|--------|----------|---------|--------|
| LivingMemoryCore | `core/living_memory/` | In-memory dict + SQLite v1 | O(n) linear |
| SovereignExperienceLedger | `core/sovereign/` | Hash-chained JSONL | Sequential |
| PatternMemory (Rust) | `bizra-omega/bizra-autopoiesis/` | In-memory | PyO3 bridge |
| MemorySynthesizer | `core/memory_coder/` | PDCA codebook | Cluster match |
| MemoryCoordinator | `core/sovereign/` | StateCheckpointer | Checkpoint restore |
| Rust BizraMemory | `bizra-omega/bizra-memory/` | InMemoryStore | FFI bridge |

**Symptoms**: No unified query interface, O(n) search at scale, siloed data, no cross-system deduplication, no embedding-powered retrieval across sources.

**Solution**: AgentDB — a single facade with SQLite v2 + FTS5 + HNSW that all systems feed into via typed adapters.

---

## 2. What's Already Built (70%)

### 2.1 Core AgentDB (`core/memory/`)

| Component | File | Lines | Status |
|-----------|------|-------|--------|
| AgentDB facade | `agent_db.py` | 339 | DONE |
| MemoryRecord + types | `types.py` | 122 | DONE |
| MemoryConfig + HNSWConfig | `config.py` | 79 | DONE |
| UnifiedStore (SQLite v2) | `unified_store.py` | 387 | DONE |
| HNSWIndex | `hnsw_index.py` | 296 | DONE |
| HybridQueryEngine | `hybrid_query.py` | 164 | DONE |
| MemoryMigrator (v1->v2) | `migrator.py` | 210 | DONE |
| LivingMemoryAdapter | `adapters/living_memory.py` | 122 | DONE |
| ExperienceLedgerAdapter | `adapters/experience_ledger.py` | 123 | DONE |
| PatternMemoryAdapter | `adapters/pattern_memory.py` | 97 | DONE |
| Tests (50+) | `tests/core/memory/` | ~1100 | DONE |

### 2.2 Proven Performance

| Scale | HNSW (hnswlib) | Linear (numpy) | Speedup |
|-------|----------------|-----------------|---------|
| 1K | 0.035ms | 5ms | 140x |
| 10K | 0.04ms | 50ms | 1,250x |
| 100K | 0.05ms | 500ms | 10,000x |
| 1M | 0.4ms | 5,000ms | 12,500x |

---

## 3. Remaining Work (30%)

### 3.1 FR-01: MemoryCoordinator ↔ AgentDB Bridge

**Requirement**: Wire `MemoryCoordinator` (auto-save loop, priority restore) to use `AgentDB` as its primary persistence target instead of only `StateCheckpointer`.

**Current gap**: `MemoryCoordinator` saves runtime state via `StateCheckpointer` (versioned JSON snapshots). `AgentDB` persists separately via `AgentDB.save()`. These two systems don't know about each other.

**Acceptance criteria**:
- `MemoryCoordinator` registers `AgentDB.get_persistable_state()` as a CORE priority state provider
- `MemoryCoordinator.save_all()` calls `AgentDB.save()` to persist HNSW index
- `MemoryCoordinator.restore_latest()` triggers `AgentDB.initialize()` if not already active
- Auto-save loop (default 120s) includes AgentDB HNSW flush
- Zero breaking changes to existing `MemoryCoordinator` API

**Edge cases**:
- AgentDB not initialized when auto-save fires → skip gracefully, log warning
- Concurrent save from two async tasks → SQLite WAL handles writes; HNSW save is idempotent
- Disk full → `AgentDB.save()` catches IOError, MemoryCoordinator logs and continues

### 3.2 FR-02: Embedding Pipeline Integration

**Requirement**: Provide a default embedding function so `AgentDB.store()` auto-generates embeddings without callers needing to supply them.

**Current gap**: `AgentDB.set_embedding_fn()` exists but no default is wired. Every caller must either supply `embedding=` or call `set_embedding_fn()` manually.

**Acceptance criteria**:
- `AgentDB.initialize()` attempts to load a default embedding function
- Priority order: (1) `sentence-transformers` all-MiniLM-L6-v2, (2) Ollama `nomic-embed-text`, (3) None (no auto-embed)
- Lazy loading: model loaded on first `store()` call that needs embedding, not at init
- Config flag: `MemoryConfig.auto_embed: bool = True` (opt-out)
- Config field: `MemoryConfig.embed_model: str = "all-MiniLM-L6-v2"`
- Embedding dimension must match `HNSWConfig.dimensions` (768 for MiniLM, 768 for nomic)
- Batch embedding: `AgentDB.store_batch()` method that embeds N items in one GPU pass

**Edge cases**:
- No embedding library installed → `auto_embed` silently disabled, warning logged once
- Embedding model dimension mismatch → raise `ValueError` at init with clear message
- GPU OOM during batch embed → fall back to CPU, log performance warning
- Empty string content → skip embedding, store with `embedding=None`

### 3.3 FR-03: Migration Orchestrator

**Requirement**: One-command migration that pulls data from all legacy systems into AgentDB.

**Current gap**: `MemoryMigrator` handles SQLite v1→v2 only. No orchestrator exists for LivingMemory (in-memory), SEL, or PatternMemory.

**Acceptance criteria**:
- `MigrationOrchestrator` class that coordinates all adapters
- Runs adapters in sequence: LivingMemory → SEL → PatternMemory → SQLite v1
- Each adapter runs in a transaction with rollback on failure
- Deduplication: content-addressable IDs prevent double-import
- Progress callback: `on_progress(source: str, migrated: int, total: int)`
- Dry-run mode: counts records without writing
- Idempotent: safe to run multiple times (upsert semantics)
- CLI entry point: `python -m core.memory.migrate`

**Edge cases**:
- LivingMemoryCore not initialized → skip with warning
- SEL file corrupted → skip bad episodes, continue with valid ones
- Rust bindings unavailable → skip PatternMemory silently
- Source has 0 records → log info, return immediately
- Mid-migration crash → no partial corruption (transactions)

### 3.4 FR-04: Health & Metrics API

**Requirement**: Expose memory system health for Grafana/Prometheus dashboards.

**Current gap**: `AgentDB.stats()` returns a basic dict. No Prometheus metrics, no health endpoint.

**Acceptance criteria**:
- `AgentDB.health()` returns structured health report:
  - SQLite: writable, file size, record count, FTS5 status
  - HNSW: loaded, vector count, capacity, dimensions
  - Memory pressure: estimated RAM usage
  - Last save timestamp
- Prometheus metrics (optional, via `prometheus_client`):
  - `agentdb_records_total` (gauge, labeled by kind + state)
  - `agentdb_vectors_total` (gauge)
  - `agentdb_search_duration_seconds` (histogram)
  - `agentdb_store_duration_seconds` (histogram)
  - `agentdb_hnsw_capacity_ratio` (gauge: count/max_elements)
- `/health` FastAPI endpoint (when run as service)

**Edge cases**:
- prometheus_client not installed → metrics silently disabled
- SQLite DB locked → health reports "degraded" not "down"
- HNSW index missing → health reports "rebuilding"

### 3.5 FR-05: Cross-Agent Memory Sync via Redis Synapse

**Requirement**: Enable multi-agent memory sharing using Redis (port 6380) pub/sub.

**Current gap**: AgentDB is single-process. No mechanism for Agent A to store a memory and Agent B to receive it.

**Acceptance criteria**:
- `MemorySyncPublisher` publishes new records to Redis channel `bizra:memory:new`
- `MemorySyncSubscriber` listens and imports records from other agents
- Message format: JSON-serialized `MemoryRecord.to_dict()` + sender agent ID
- Deduplication: content-addressable IDs prevent re-import of own records
- Optional: agents can subscribe to specific kinds (e.g., only SEMANTIC)
- Config: `MemoryConfig.sync_enabled: bool = False`, `sync_redis_url: str = "redis://localhost:6380"`
- Graceful degradation: if Redis unavailable, sync silently disabled

**Edge cases**:
- Redis connection drops mid-publish → buffer locally, retry on reconnect
- Two agents store same content simultaneously → both get same ID (idempotent)
- Large embedding in message → Redis max message size check (512MB default, safe)
- Agent receives its own message → filter by sender_id != self_id

---

## 4. Non-Functional Requirements

### 4.1 Performance

| Metric | Target | Current |
|--------|--------|---------|
| Search latency (10K records) | < 1ms | 0.04ms (HNSW only) |
| Store latency (single record) | < 5ms | ~2ms (SQLite commit) |
| Batch store (1000 records) | < 500ms | ~400ms (batch upsert) |
| HNSW rebuild (100K records) | < 30s | ~20s |
| Auto-embed (single, CPU) | < 50ms | N/A (not yet wired) |
| Memory footprint (100K records) | < 500MB | ~300MB (HNSW + SQLite) |

### 4.2 Reliability

- SQLite WAL mode: crash-safe writes
- HNSW index saved alongside SQLite → atomic save (both or neither)
- Backup before any destructive migration
- All adapters are read-from-source, write-to-AgentDB (never modify source)
- Content-addressable IDs → natural deduplication

### 4.3 Compatibility

- Python 3.11+ (dataclasses, type hints)
- hnswlib optional (numpy fallback)
- prometheus_client optional
- Redis optional (sync feature)
- Rust PyO3 bindings optional (PatternMemory adapter)
- All constitutional thresholds from `core/integration/constants.py`

---

## 5. Security Constraints

- No secrets in memory records (metadata, tags, content)
- SQLite file permissions: 0o600 (owner read/write only)
- Redis sync: records don't traverse network unless sync explicitly enabled
- Embedding model loaded from local disk only (no remote model downloads at runtime)
- HNSW index file: no executable content, binary float32 arrays only

---

## 6. Testing Strategy

### 6.1 Unit Tests (existing, extend)

- `test_agent_db.py` — store/search/forget lifecycle
- `test_unified_store.py` — SQLite CRUD, FTS5, batch ops
- `test_hnsw_index.py` — add/remove/search, save/load, numpy fallback
- `test_hybrid_query.py` — score fusion, filtering, deduplication
- `test_migrator.py` — v1→v2 migration, backup, error handling
- `test_adapters.py` — all three adapters, edge cases

### 6.2 New Tests Required

| Test Module | Coverage Target |
|-------------|-----------------|
| `test_coordinator_bridge.py` | FR-01: MemoryCoordinator ↔ AgentDB |
| `test_embedding_pipeline.py` | FR-02: auto-embed, lazy load, fallback |
| `test_migration_orchestrator.py` | FR-03: multi-source migration |
| `test_health_metrics.py` | FR-04: health API, Prometheus metrics |
| `test_redis_sync.py` | FR-05: pub/sub, dedup, reconnect |
| `test_performance.py` (extend) | All NFRs: latency, throughput, memory |

### 6.3 Integration Tests

- Full pipeline: store with auto-embed → search → verify score fusion
- Migration: populate v1 DB → migrate → verify all records in v2
- MemoryCoordinator: auto-save → crash sim → restore → verify AgentDB state
- Redis sync: two AgentDB instances sharing via Redis pub/sub

### 6.4 Property-Based Tests (Hypothesis)

- Store N random records → search always returns records with score > 0
- Content-addressable: same content always produces same ID
- Deduplication: storing same content twice → count stays at 1
- Score fusion weights always sum to 1.0

---

## 7. Dependency Map

```
MemoryCoordinator (FR-01)
├── AgentDB (existing)
│   ├── UnifiedStore (existing)
│   ├── HNSWIndex (existing)
│   └── HybridQueryEngine (existing)
├── EmbeddingPipeline (FR-02)
│   ├── sentence-transformers (optional)
│   └── Ollama client (optional)
└── MemorySyncPublisher (FR-05)
    └── Redis 6380 (optional)

MigrationOrchestrator (FR-03)
├── AgentDB (existing)
├── LivingMemoryAdapter (existing)
├── ExperienceLedgerAdapter (existing)
├── PatternMemoryAdapter (existing)
└── MemoryMigrator (existing)

HealthAPI (FR-04)
├── AgentDB.stats() (existing)
├── prometheus_client (optional)
└── FastAPI (optional)
```

---

## 8. Implementation Order

1. **FR-01** MemoryCoordinator bridge (no new deps, low risk)
2. **FR-02** Embedding pipeline (sentence-transformers already installed)
3. **FR-03** Migration orchestrator (uses existing adapters)
4. **FR-04** Health & metrics (low risk, observability)
5. **FR-05** Redis sync (highest risk, optional feature)

Each FR is independently deployable. No FR blocks another.

---

## 9. Constitutional Alignment

| Principle | How |
|-----------|-----|
| Ihsan (excellence ≥ 0.95) | Records below threshold tagged `low_ihsan`, importance halved |
| SNR (signal quality ≥ 0.85) | Tracked per record, queryable via `snr_score` field |
| ADL Gini (justice ≤ 0.40) | Not directly applicable to memory (no resource distribution) |
| Content-addressable | ID = `hex_digest(content + source)[:16]` → dedup by design |
| Never modify source | All adapters read-only on source systems |
| Thresholds from constants.py | `config.py` imports `UNIFIED_IHSAN_THRESHOLD`, `UNIFIED_SNR_THRESHOLD` |
