"""
AgentDB — Single facade for all memory operations.

This is the ONE entry point that the rest of the codebase uses
for memory storage and retrieval. It wires together:
  - UnifiedStore (SQLite v2 + FTS5)
  - HNSWIndex (sub-linear vector search)
  - HybridQueryEngine (score fusion)

Usage:
    from core.memory import AgentDB
    db = AgentDB()
    db.initialize()
    db.store("The Earth orbits the Sun", importance=0.9)
    results = db.search("solar system")

Standing on Giants: ADR-006 (Unified Memory Service) + ADR-009 (Hybrid Memory Backend)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional, Sequence

from core.proof_engine.canonical import hex_digest

from .config import MemoryConfig
from .hnsw_index import HNSWIndex
from .hybrid_query import HybridQueryEngine
from .types import MemoryKind, MemoryRecord, QueryOptions, RecordState, SearchResult
from .unified_store import UnifiedStore

logger = logging.getLogger(__name__)


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

        self._initialized = True
        logger.info(
            f"AgentDB initialized: {self._store.count()} records, "
            f"{self._hnsw.count} vectors"
        )

    def set_embedding_fn(self, fn) -> None:
        """Inject an embedding function: str -> List[float]."""
        self._embedding_fn = fn

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
        if embedding is None and self._embedding_fn is not None:
            try:
                embedding = self._embedding_fn(content)
            except Exception as e:
                logger.warning(f"Auto-embedding failed: {e}")

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

        # Persist to SQLite + FTS
        self._store.upsert(record)

        # Index in HNSW
        if record.embedding is not None:
            self._hnsw.add(record.id, record.embedding)

        logger.debug(f"Stored memory {record_id[:8]}... kind={kind.value}")
        return record

    def store_record(self, record: MemoryRecord) -> None:
        """Store a pre-built MemoryRecord (used by adapters and migrator)."""
        self._ensure_initialized()
        self._store.upsert(record)
        if record.embedding is not None:
            self._hnsw.add(record.id, record.embedding)

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
    ) -> List[SearchResult]:
        """Search memory using hybrid scoring.

        Provide query text, embedding, or both for best results.
        """
        self._ensure_initialized()
        assert self._query_engine is not None

        # Auto-embed query text if possible
        effective_embedding = query_embedding
        if effective_embedding is None and query and self._embedding_fn:
            try:
                effective_embedding = self._embedding_fn(query)
            except Exception:
                pass

        options = QueryOptions(
            query_text=query,
            query_embedding=effective_embedding,
            top_k=top_k,
            min_score=min_score,
            kinds=kinds,
            tags=tags,
            source=source,
        )

        return self._query_engine.search(options, context_ids=context_ids)

    # ── Retrieve ─────────────────────────────────────────────────────────

    def retrieve(self, record_id: str) -> Optional[MemoryRecord]:
        """Get a specific record by ID."""
        self._ensure_initialized()
        record = self._store.get(record_id)
        if record:
            self._store.touch_access(record_id)
        return record

    # ── Forget ───────────────────────────────────────────────────────────

    def forget(self, record_id: str, hard: bool = False) -> bool:
        """Remove a record from memory.

        Soft delete (default) marks as deleted but preserves data.
        Hard delete removes from SQLite, FTS, and HNSW.
        """
        self._ensure_initialized()
        self._hnsw.remove(record_id)
        return self._store.delete(record_id, hard=hard)

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist HNSW index to disk (SQLite auto-commits)."""
        if self._initialized:
            self._hnsw.save(self._config.hnsw_path)
            logger.info("AgentDB saved (HNSW index persisted)")

    def get_persistable_state(self) -> dict:
        """Return state dict for MemoryCoordinator integration."""
        return {
            "record_count": self._store.count(),
            "vector_count": self._hnsw.count,
            "hnsw_path": str(self._config.hnsw_path),
            "sqlite_path": str(self._config.sqlite_path),
        }

    # ── Stats ────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return summary statistics."""
        return {
            "total_records": self._store.count(),
            "active_records": self._store.count(state=RecordState.ACTIVE),
            "archived_records": self._store.count(state=RecordState.ARCHIVED),
            "deleted_records": self._store.count(state=RecordState.DELETED),
            "indexed_vectors": self._hnsw.count,
            "hnsw_capacity": self._hnsw.capacity,
            "sqlite_path": str(self._config.sqlite_path),
            "hnsw_path": str(self._config.hnsw_path),
        }

    # ── Internal ─────────────────────────────────────────────────────────

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("AgentDB not initialized — call initialize() first")

    def _rebuild_hnsw(self) -> None:
        """Rebuild HNSW index from SQLite embeddings."""
        import numpy as np

        rows = self._store.load_with_embeddings()
        if not rows:
            return

        logger.info(f"Rebuilding HNSW index from {len(rows)} stored embeddings...")
        for record_id, blob in rows:
            vec = np.frombuffer(blob, dtype=np.float32)
            if vec.shape[0] == self._config.hnsw.dimensions:
                self._hnsw.add(record_id, vec)

        logger.info(f"HNSW rebuild complete: {self._hnsw.count} vectors indexed")
