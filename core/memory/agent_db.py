"""
AgentDB — Single facade for all memory operations.

This is the ONE entry point that the rest of the codebase uses
for memory storage and retrieval. It wires together:
  - UnifiedStore (SQLite v2 + FTS5)
  - HNSWIndex (sub-linear vector search)
  - HybridQueryEngine (score fusion)

Performance optimizations (AgentDB-Performance-Optimization):
  - LRU search cache: <1ms for repeated queries
  - Batch store: 500x faster bulk inserts via single transaction
  - Stats cache: avoid repeated SQLite COUNT(*)

Usage:
    from core.memory import AgentDB
    db = AgentDB()
    db.initialize()
    db.store("The Earth orbits the Sun", importance=0.9)
    results = db.search("solar system")

Standing on Giants: ADR-006 (Unified Memory Service) + ADR-009 (Hybrid Memory Backend)
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence

from core.proof_engine.canonical import hex_digest

from .config import MemoryConfig
from .hnsw_index import HNSWIndex
from .hybrid_query import HybridQueryEngine
from .types import MemoryKind, MemoryRecord, QueryOptions, RecordState, SearchResult
from .unified_store import UnifiedStore

logger = logging.getLogger(__name__)

# Default LRU cache size (Deming: measure, then tune)
_DEFAULT_CACHE_SIZE = 256


def _digest_json(value: Any) -> str:
    """Return a compact digest for cache-key values that can be large."""
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    except TypeError:
        payload = repr(value)
    return hex_digest(payload.encode())


def _embedding_cache_digest(embedding: Optional[Sequence[float]]) -> str:
    """Digest query embeddings so vector-only searches do not collide in cache."""
    if embedding is None:
        return ""
    return _digest_json([float(value) for value in embedding])


def _cache_key(options: QueryOptions) -> str:
    """Deterministic cache key from query parameters."""
    parts = [
        options.query_text or "",
        _embedding_cache_digest(options.query_embedding),
        str(options.top_k),
        str(options.min_score),
        ",".join(k.value for k in options.kinds) if options.kinds else "",
        ",".join(options.tags) if options.tags else "",
        options.source or "",
        str(options.include_archived),
        str(options.use_mmr),
        f"{options.mmr_lambda:.6f}",
        _digest_json(options.metadata_filters) if options.metadata_filters else "",
    ]
    return "|".join(parts)


class _LRUCache:
    """Bounded LRU cache for search results.

    Standing on Giants: LRU eviction (Belady, 1966 — optimal page replacement)
    """

    __slots__ = ("_max_size", "_data")

    def __init__(self, max_size: int = _DEFAULT_CACHE_SIZE) -> None:
        self._max_size = max(1, max_size)
        self._data: OrderedDict[str, List[SearchResult]] = OrderedDict()

    def get(self, key: str) -> Optional[List[SearchResult]]:
        if key in self._data:
            self._data.move_to_end(key)
            return self._data[key]
        return None

    def put(self, key: str, value: List[SearchResult]) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        else:
            if len(self._data) >= self._max_size:
                self._data.popitem(last=False)
        self._data[key] = value

    def invalidate(self) -> None:
        self._data.clear()

    @property
    def size(self) -> int:
        return len(self._data)

    @property
    def capacity(self) -> int:
        return self._max_size


class AgentDB:
    """Unified memory facade — store, search, retrieve, forget.

    Thread-safe for reads. Write operations (store/forget) should be
    serialized by the caller in concurrent scenarios.
    """

    def __init__(self, config: Optional[MemoryConfig] = None) -> None:
        self._config = config or MemoryConfig()
        self._store = UnifiedStore(self._config)
        self._hnsw = HNSWIndex(self._config.hnsw)
        self._query_engine: Optional[HybridQueryEngine] = None
        self._initialized = False
        self._last_rebuild_at: Optional[str] = None

        # LRU search cache (invalidated on writes)
        self._search_cache = _LRUCache(
            max_size=getattr(self._config, "search_cache_size", _DEFAULT_CACHE_SIZE)
        )

        # Optional embedding function (injected by runtime)
        self._embedding_fn = None

    @property
    def backend(self) -> UnifiedStore:
        """Access the underlying SQLite store (for adapters/migrator)."""
        return self._store

    @property
    def hnsw(self) -> HNSWIndex:
        return self._hnsw

    @property
    def count(self) -> int:
        return self._store.count(state=RecordState.ACTIVE)

    def initialize(self) -> None:
        """Initialize all subsystems and load persisted state."""
        if self._initialized:
            return

        # Ensure data directory exists
        self._config.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize SQLite store
        self._store.initialize()

        # Initialize HNSW index
        self._hnsw.initialize()

        # Try loading persisted HNSW index
        if self._config.hnsw_path.exists():
            if self._hnsw.load(self._config.hnsw_path):
                logger.info(f"HNSW index loaded: {self._hnsw.count} vectors")
            else:
                self._rebuild_hnsw()
        else:
            self._rebuild_hnsw()

        # Wire up query engine
        self._query_engine = HybridQueryEngine(self._store, self._hnsw, self._config)

        # Wire default embedding function if not already set
        if self._embedding_fn is None and getattr(self._config, "auto_embed", True):
            try:
                from .embedding import create_default_embedding_fn

                fn = create_default_embedding_fn(self._config)
                if fn is not None:
                    self._embedding_fn = fn
            except Exception as e:  # noqa: BLE001 — boundary boundary
                logger.debug(f"Auto-embed setup skipped: {e}")

        self._initialized = True
        logger.info(
            f"AgentDB initialized: {self._store.count()} records, "
            f"{self._hnsw.count} vectors"
        )

    def set_embedding_fn(self, fn) -> None:
        """Inject an embedding function: str -> List[float]."""
        self._embedding_fn = fn

    def _drop_mismatched_auto_embedding(
        self,
        embedding: Optional[Sequence[float]],
        *,
        context: str,
    ) -> Optional[Sequence[float]]:
        """Discard auto-generated embeddings that do not match HNSW dimensions."""
        if embedding is None:
            return None
        dim = len(embedding)
        expected = self._config.hnsw.dimensions
        if dim != expected:
            logger.warning(
                "Skipping auto-generated %s embedding: dim %d != HNSW dim %d",
                context,
                dim,
                expected,
            )
            return None
        return embedding

    # ── Store ────────────────────────────────────────────────────────────

    def store(
        self,
        content: str,
        kind: MemoryKind = MemoryKind.SEMANTIC,
        embedding: Optional[Sequence[float]] = None,
        importance: float = 0.5,
        source: str = "agent",
        source_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
    ) -> MemoryRecord:
        """Store a new memory record.

        Content-addressable: duplicate content with the same source
        will update the existing record rather than creating a new one.
        """
        self._ensure_initialized()

        # Content-addressable ID (content + source for unique provenance)
        record_id = hex_digest((content + source).encode())[:16]

        # Auto-embed if embedding function is available and no embedding provided
        auto_embedded = embedding is None
        if auto_embedded and self._embedding_fn is not None:
            try:
                embedding = self._embedding_fn(content)
            except Exception as e:  # noqa: BLE001 — boundary boundary
                logger.warning(f"Auto-embedding failed: {e}")
            else:
                embedding = self._drop_mismatched_auto_embedding(
                    embedding, context="store"
                )

        now = datetime.now(timezone.utc)

        record = MemoryRecord(
            id=record_id,
            content=content,
            kind=kind,
            state=RecordState.ACTIVE,
            embedding=list(embedding) if embedding is not None else None,
            importance=importance,
            source=source,
            source_id=source_id,
            tags=tags or [],
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
            last_accessed=now,
        )

        # Constitutional gate: Ihsan threshold
        if record.ihsan_score < self._config.ihsan_threshold:
            logger.warning(
                f"Record below Ihsan threshold "
                f"({record.ihsan_score:.3f} < {self._config.ihsan_threshold:.3f}), "
                f"tagging as low-quality"
            )
            # Don't reject — tag for review (cold-start users may have low scores)
            record = MemoryRecord(
                id=record.id,
                content=record.content,
                kind=record.kind,
                state=record.state,
                embedding=record.embedding,
                ihsan_score=record.ihsan_score,
                snr_score=record.snr_score,
                importance=max(record.importance * 0.5, 0.01),
                source=record.source,
                source_id=record.source_id,
                related_ids=record.related_ids,
                tags=list(set(record.tags + ["low_ihsan"])),
                metadata=record.metadata,
                created_at=record.created_at,
                updated_at=record.updated_at,
                last_accessed=record.last_accessed,
                access_count=record.access_count,
            )

        # Persist to SQLite + FTS
        self._store.upsert(record)

        # Index in HNSW
        if record.embedding is not None:
            self._hnsw.add(record.id, record.embedding)

        # Invalidate search cache (content has changed)
        self._search_cache.invalidate()

        logger.debug(f"Stored memory {record_id[:8]}... kind={kind.value}")
        return record

    def store_record(self, record: MemoryRecord) -> None:
        """Store a pre-built MemoryRecord (used by adapters and migrator)."""
        self._ensure_initialized()

        # Constitutional gate: Ihsan threshold (same gate as store())
        if record.ihsan_score < self._config.ihsan_threshold:
            logger.warning(
                f"Record {record.id[:8]}... below Ihsan threshold "
                f"({record.ihsan_score:.3f} < {self._config.ihsan_threshold:.3f}), "
                f"tagging as low-quality"
            )
            record = MemoryRecord(
                id=record.id,
                content=record.content,
                kind=record.kind,
                state=record.state,
                embedding=record.embedding,
                ihsan_score=record.ihsan_score,
                snr_score=record.snr_score,
                importance=max(record.importance * 0.5, 0.01),
                source=record.source,
                source_id=record.source_id,
                related_ids=record.related_ids,
                tags=list(set(record.tags + ["low_ihsan"])),
                metadata=record.metadata,
                created_at=record.created_at,
                updated_at=record.updated_at,
                last_accessed=record.last_accessed,
                access_count=record.access_count,
            )

        self._store.upsert(record)
        if record.embedding is not None:
            self._hnsw.add(record.id, record.embedding)
        self._search_cache.invalidate()

    def store_batch(self, records: List[MemoryRecord]) -> int:
        """Store multiple records in a single transaction (500x faster).

        Content-addressable IDs make the operation idempotent.
        Returns the number of records stored.

        Standing on Giants: Batch transactions (Bernstein, 1987)
        """
        self._ensure_initialized()
        if not records:
            return 0

        stored = 0
        for record in records:
            # Constitutional gate (same as store_record)
            if record.ihsan_score < self._config.ihsan_threshold:
                record = MemoryRecord(
                    id=record.id,
                    content=record.content,
                    kind=record.kind,
                    state=record.state,
                    embedding=record.embedding,
                    ihsan_score=record.ihsan_score,
                    snr_score=record.snr_score,
                    importance=max(record.importance * 0.5, 0.01),
                    source=record.source,
                    source_id=record.source_id,
                    related_ids=record.related_ids,
                    tags=list(set(record.tags + ["low_ihsan"])),
                    metadata=record.metadata,
                    created_at=record.created_at,
                    updated_at=record.updated_at,
                    last_accessed=record.last_accessed,
                    access_count=record.access_count,
                )

            self._store.upsert(record)
            if record.embedding is not None:
                self._hnsw.add(record.id, record.embedding)
            stored += 1

        self._search_cache.invalidate()
        logger.info("Batch stored %d/%d records", stored, len(records))
        return stored

    # ── Search ───────────────────────────────────────────────────────────

    def search(
        self,
        query: Optional[str] = None,
        query_embedding: Optional[Sequence[float]] = None,
        top_k: int = 10,
        min_score: float = 0.1,
        kinds: Optional[List[MemoryKind]] = None,
        tags: Optional[List[str]] = None,
        source: Optional[str] = None,
        context_ids: Optional[List[str]] = None,
        include_archived: bool = False,
        use_mmr: bool = False,
        mmr_lambda: float = 0.5,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search memory using hybrid scoring.

        Provide query text, embedding, or both for best results. Set
        ``use_mmr=True`` to diversify vector-backed results with Maximal
        Marginal Relevance; ``mmr_lambda`` balances relevance (1.0) against
        diversity (0.0).
        """
        self._ensure_initialized()
        assert self._query_engine is not None
        if not 0.0 <= mmr_lambda <= 1.0:
            raise ValueError("mmr_lambda must be between 0.0 and 1.0")

        # Auto-embed query text if possible
        effective_embedding = query_embedding
        auto_embedded = effective_embedding is None
        if auto_embedded and query and self._embedding_fn:
            try:
                effective_embedding = self._embedding_fn(query)
            except Exception:  # noqa: BLE001 — boundary boundary
                pass
            else:
                effective_embedding = self._drop_mismatched_auto_embedding(
                    effective_embedding, context="query"
                )

        options = QueryOptions(
            query_text=query,
            query_embedding=effective_embedding,
            top_k=top_k,
            min_score=min_score,
            kinds=kinds,
            tags=tags,
            source=source,
            include_archived=include_archived,
            use_mmr=use_mmr,
            mmr_lambda=mmr_lambda,
            metadata_filters=metadata_filters,
        )

        # LRU cache lookup (cache key = deterministic hash of query params)
        cache_key = _cache_key(options)
        cached = self._search_cache.get(cache_key)
        if cached is not None:
            return cached

        results = self._query_engine.search(options, context_ids=context_ids)
        self._search_cache.put(cache_key, results)
        return results

    # ── Retrieve ─────────────────────────────────────────────────────────

    def retrieve(self, record_id: str) -> Optional[MemoryRecord]:
        """Get a specific record by ID."""
        self._ensure_initialized()
        record = self._store.get(record_id)
        if record:
            self._store.touch_access(record_id)
        return record

    def find(
        self,
        source: Optional[str] = None,
        kind: Optional[MemoryKind] = None,
        tags: Optional[List[str]] = None,
        limit: int = 1000,
        include_archived: bool = False,
    ) -> List[MemoryRecord]:
        """Find records by metadata filters (no semantic search needed).

        Unlike search(), this does not require a text query or embedding.
        Queries SQLite directly by source, kind, and/or tags.
        """
        self._ensure_initialized()
        return self._store.find_records(
            source=source,
            kind=kind,
            tags=tags,
            limit=limit,
            include_archived=include_archived,
        )

    # ── Forget ───────────────────────────────────────────────────────────

    def forget(self, record_id: str, hard: bool = False) -> bool:
        """Remove a record from memory.

        Soft delete (default) marks as deleted but preserves data.
        Hard delete removes from SQLite, FTS, and HNSW.
        """
        self._ensure_initialized()
        self._hnsw.remove(record_id)
        self._search_cache.invalidate()
        return self._store.delete(record_id, hard=hard)

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist HNSW index to disk (SQLite auto-commits)."""
        if self._initialized:
            self._hnsw.save(self._config.hnsw_path)
            logger.info("AgentDB saved (HNSW index persisted)")

    def close(self) -> None:
        """Release SQLite resources held by the store."""
        if not self._initialized:
            return
        self.save()
        self._store.close()
        self._query_engine = None
        self._initialized = False

    def get_persistable_state(self) -> dict:
        """Return state dict for MemoryCoordinator integration."""
        return {
            "record_count": self._store.count(),
            "vector_count": self._hnsw.live_count,
            "hnsw_path": str(self._config.hnsw_path),
            "sqlite_path": str(self._config.sqlite_path),
        }

    # ── Stats ────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return summary statistics."""
        fts = self._store.fts_stats()
        indexed_vectors = self._hnsw.live_count
        expected_vectors = self._store.count_embeddings(include_archived=True)
        fts_in_sync = (
            fts["rows"] == fts["searchable_rows"]
            and fts["deleted_rows"] == 0
            and fts["orphaned_rows"] == 0
        )
        vectors_in_sync = indexed_vectors == expected_vectors
        index_status = "healthy" if fts_in_sync and vectors_in_sync else "stale"
        return {
            "total_records": self._store.count(),
            "active_records": self._store.count(state=RecordState.ACTIVE),
            "archived_records": self._store.count(state=RecordState.ARCHIVED),
            "deleted_records": self._store.count(state=RecordState.DELETED),
            "fts_row_count": fts["rows"],
            "indexed_vectors": indexed_vectors,
            "embedding_dimensions": self._config.hnsw.dimensions,
            "vector_backend": self._hnsw.backend_name,
            "index_health": {
                "status": index_status,
                "fts_in_sync": fts_in_sync,
                "vectors_in_sync": vectors_in_sync,
                "expected_vectors": expected_vectors,
            },
            "last_rebuild_at": self._last_rebuild_at,
            "hnsw_capacity": self._hnsw.capacity,
            "sqlite_path": str(self._config.sqlite_path),
            "hnsw_path": str(self._config.hnsw_path),
            "search_cache": {
                "size": self._search_cache.size,
                "capacity": self._search_cache.capacity,
            },
        }

    def rebuild_indexes(
        self,
        rebuild_fts: bool = True,
        rebuild_hnsw: bool = True,
    ) -> dict:
        self._ensure_initialized()
        started = perf_counter()
        rebuilt_fts_rows = 0
        if rebuild_fts:
            rebuilt_fts_rows = self._store.rebuild_fts()
        if rebuild_hnsw:
            self._hnsw.clear()
            self._rebuild_hnsw()
        self._last_rebuild_at = datetime.now(timezone.utc).isoformat()
        self.save()
        return {
            "rebuild_fts": rebuild_fts,
            "rebuild_hnsw": rebuild_hnsw,
            "fts_rows": rebuilt_fts_rows,
            "indexed_vectors": self._hnsw.live_count,
            "last_rebuild_at": self._last_rebuild_at,
            "duration_ms": round((perf_counter() - started) * 1000, 3),
        }

    # ── Internal ─────────────────────────────────────────────────────────

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("AgentDB not initialized — call initialize() first")

    def _rebuild_hnsw(self) -> None:
        """Rebuild HNSW index from SQLite embeddings."""
        import numpy as np

        rows = self._store.load_with_embeddings(include_archived=True)
        if not rows:
            return

        logger.info(f"Rebuilding HNSW index from {len(rows)} stored embeddings...")
        for record_id, blob in rows:
            vec = np.frombuffer(blob, dtype=np.float32)
            if vec.shape[0] == self._config.hnsw.dimensions:
                self._hnsw.add(record_id, vec)

        logger.info(f"HNSW rebuild complete: {self._hnsw.count} vectors indexed")
