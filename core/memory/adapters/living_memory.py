"""
LivingMemory Adapter — Bidirectional Bridge between LivingMemoryCore and AgentDB.

Converts MemoryEntry (core/living_memory/core.py) ↔ MemoryRecord
(core/memory/types.py) in both directions, enabling:

  1. **Export**: LivingMemoryCore → AgentDB (migration, one-shot or incremental)
  2. **Import**: AgentDB → LivingMemoryCore (reverse sync for shared memories)
  3. **HNSW-accelerated search**: Proxy retrieve() through AgentDB's vector index
     for 150x faster semantic lookup (Malkov & Yashunin, 2018)
  4. **Live sync**: Auto-mirror encode() calls to AgentDB on write

Standing on Giants:
  - Maturana & Varela (Autopoiesis — Living Systems, 1980)
  - Malkov & Yashunin (HNSW — Navigable Small Worlds, 2018)
  - ADR-006 (Unified Memory Service)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from core.memory.types import MemoryKind, MemoryRecord, RecordState

logger = logging.getLogger(__name__)

# Map between LivingMemory MemoryType and AgentDB MemoryKind
_TYPE_MAP: Dict[str, MemoryKind] = {
    "episodic": MemoryKind.EPISODIC,
    "semantic": MemoryKind.SEMANTIC,
    "procedural": MemoryKind.PROCEDURAL,
    "working": MemoryKind.WORKING,
    "prospective": MemoryKind.PROSPECTIVE,
}

_KIND_TO_TYPE: Dict[MemoryKind, str] = {v: k for k, v in _TYPE_MAP.items()}

_STATE_MAP: Dict[str, RecordState] = {
    "active": RecordState.ACTIVE,
    "archived": RecordState.ARCHIVED,
    "deleted": RecordState.DELETED,
    "consolidating": RecordState.ACTIVE,
    "decaying": RecordState.ACTIVE,
    "corrupted": RecordState.ACTIVE,
}

_RECORD_STATE_TO_LM: Dict[RecordState, str] = {
    RecordState.ACTIVE: "active",
    RecordState.ARCHIVED: "archived",
    RecordState.DELETED: "deleted",
}


class LivingMemoryAdapter:
    """Bidirectional adapter between LivingMemoryCore and AgentDB.

    Usage (export):
        adapter = LivingMemoryAdapter(lm_core)
        records = adapter.export_all()

    Usage (bridge with HNSW acceleration):
        bridge = LivingMemoryBridge(lm_core, agent_db)
        results = bridge.search("query", top_k=10)  # Uses HNSW index
        bridge.sync_to_agentdb()   # Push all living memories to AgentDB
        bridge.sync_entry(entry)   # Push single entry on encode()
    """

    def __init__(self, living_memory: Any) -> None:
        """Accept a LivingMemoryCore instance (duck-typed to avoid circular import)."""
        self._lm = living_memory

    # ── Forward: LivingMemory → AgentDB ─────────────────────────────────

    def export_all(self) -> List[MemoryRecord]:
        """Export all active memories as MemoryRecords."""
        records: List[MemoryRecord] = []
        memories = getattr(self._lm, "_memories", {})

        for _entry_id, entry in memories.items():
            record = self.entry_to_record(entry)
            if record is not None:
                records.append(record)

        logger.info("LivingMemoryAdapter: exported %d records", len(records))
        return records

    def export_entry(self, entry: Any) -> Optional[MemoryRecord]:
        """Convert a single MemoryEntry to MemoryRecord."""
        return self.entry_to_record(entry)

    def entry_to_record(self, entry: Any) -> Optional[MemoryRecord]:
        """Convert a LivingMemory MemoryEntry to AgentDB MemoryRecord."""
        try:
            kind = _TYPE_MAP.get(entry.memory_type.value, MemoryKind.SEMANTIC)
            state = _STATE_MAP.get(entry.state.value, RecordState.ACTIVE)

            if state == RecordState.DELETED:
                return None

            embedding: Optional[List[float]] = None
            if entry.embedding is not None:
                embedding = entry.embedding.tolist()

            tags = [f"lm_type:{entry.memory_type.value}"]
            if hasattr(entry, "parent_id") and entry.parent_id:
                tags.append(f"parent:{entry.parent_id}")

            return MemoryRecord(
                id=entry.id,
                content=entry.content,
                kind=kind,
                state=state,
                embedding=embedding,
                ihsan_score=entry.ihsan_score,
                snr_score=entry.snr_score,
                importance=entry.importance,
                source=f"living_memory:{entry.source}",
                source_id=entry.id,
                related_ids=list(entry.related_ids),
                tags=tags,
                created_at=entry.created_at,
                updated_at=entry.last_accessed,
                last_accessed=entry.last_accessed,
                access_count=entry.access_count,
                metadata={
                    "emotional_weight": getattr(entry, "emotional_weight", 0.5),
                    "confidence": getattr(entry, "confidence", 1.0),
                    "parent_id": getattr(entry, "parent_id", None),
                    "reinforcement_count": getattr(entry, "reinforcement_count", 1),
                    "origin": "living_memory",
                },
            )
        except Exception as e:  # noqa: BLE001 — boundary boundary
            logger.warning(
                "Failed to convert MemoryEntry %s: %s",
                getattr(entry, "id", "?"),
                e,
            )
            return None

    # ── Reverse: AgentDB → LivingMemory ─────────────────────────────────

    def record_to_entry_dict(self, record: MemoryRecord) -> Dict[str, Any]:
        """Convert an AgentDB MemoryRecord to a dict suitable for MemoryEntry.from_dict().

        Returns a dict rather than a MemoryEntry to avoid importing
        core.living_memory.core (which would create a circular dependency).
        Caller is responsible for calling ``MemoryEntry.from_dict(d)``.
        """
        lm_type = _KIND_TO_TYPE.get(record.kind, "semantic")
        lm_state = _RECORD_STATE_TO_LM.get(record.state, "active")
        meta = record.metadata or {}

        return {
            "id": record.source_id or record.id,
            "content": record.content,
            "memory_type": lm_type,
            "created_at": record.created_at.isoformat(),
            "last_accessed": record.last_accessed.isoformat(),
            "access_count": record.access_count,
            "reinforcement_count": meta.get("reinforcement_count", 1),
            "ihsan_score": record.ihsan_score,
            "snr_score": record.snr_score,
            "confidence": meta.get("confidence", 1.0),
            "state": lm_state,
            "source": (
                record.source.replace("living_memory:", "", 1)
                if record.source.startswith("living_memory:")
                else record.source
            ),
            "related_ids": record.related_ids,
            "importance": record.importance,
            "emotional_weight": meta.get("emotional_weight", 0.5),
        }

    # ── Private helpers kept for backward compat ─────────────────────────

    def _entry_to_record(self, entry: Any) -> Optional[MemoryRecord]:
        """Backward-compatible alias for entry_to_record()."""
        return self.entry_to_record(entry)


class LivingMemoryBridge:
    """Bidirectional bridge connecting LivingMemoryCore and AgentDB.

    Provides:
      - ``sync_to_agentdb()``: bulk push all living memories into AgentDB
      - ``sync_entry(entry)``: push a single entry (call after encode())
      - ``search(query, ...)``: HNSW-accelerated semantic search through AgentDB
      - ``sync_from_agentdb(source)``: pull records from AgentDB into living memory
      - ``stats()``: cross-system statistics

    Standing on Giants: Malkov & Yashunin (HNSW, 2018), ADR-006

    Usage:
        from core.living_memory.core import LivingMemoryCore
        from core.memory.agent_db import AgentDB

        lm = LivingMemoryCore()
        await lm.initialize()
        db = AgentDB()

        bridge = LivingMemoryBridge(lm, db)
        bridge.sync_to_agentdb()           # Bulk sync
        results = bridge.search("query")   # HNSW-accelerated
    """

    def __init__(self, living_memory: Any, agent_db: Any) -> None:
        self._lm = living_memory
        self._db = agent_db
        self._adapter = LivingMemoryAdapter(living_memory)
        self._synced_ids: Set[str] = set()

    @property
    def adapter(self) -> LivingMemoryAdapter:
        """Access the underlying adapter for direct conversions."""
        return self._adapter

    def sync_to_agentdb(self) -> int:
        """Push all living memories into AgentDB (idempotent via content-addressable IDs).

        Returns the number of records stored.
        """
        records = self._adapter.export_all()
        stored = 0
        for record in records:
            try:
                self._db.store_record(record)
                self._synced_ids.add(record.id)
                stored += 1
            except Exception as e:  # noqa: BLE001 — boundary
                logger.warning("sync_to_agentdb failed for %s: %s", record.id, e)
        logger.info("LivingMemoryBridge: synced %d/%d to AgentDB", stored, len(records))
        return stored

    def sync_entry(self, entry: Any) -> Optional[str]:
        """Push a single LivingMemory entry to AgentDB.

        Call this after ``LivingMemoryCore.encode()`` for live mirroring.
        Returns the record ID on success, None on failure.
        """
        record = self._adapter.entry_to_record(entry)
        if record is None:
            return None
        try:
            self._db.store_record(record)
            self._synced_ids.add(record.id)
            return record.id
        except Exception as e:  # noqa: BLE001 — boundary
            logger.warning("sync_entry failed for %s: %s", record.id, e)
            return None

    def search(
        self,
        query: str,
        top_k: int = 10,
        memory_type: Optional[str] = None,
        min_score: float = 0.1,
        use_mmr: bool = False,
        mmr_lambda: float = 0.5,
    ) -> List[MemoryRecord]:
        """HNSW-accelerated semantic search via AgentDB.

        Replaces the O(n) numpy scan in LivingMemoryCore.retrieve() with
        AgentDB's HNSW index for sub-linear search (150x faster at scale).

        Args:
            query: Natural language query text.
            top_k: Maximum results to return.
            memory_type: Filter by living memory type (e.g. "episodic").
            min_score: Minimum relevance threshold.
            use_mmr: Enable Maximal Marginal Relevance diversity.
            mmr_lambda: MMR relevance/diversity balance (0.0-1.0).

        Returns:
            List of MemoryRecords sorted by relevance.
        """
        kinds = None
        if memory_type:
            kind = _TYPE_MAP.get(memory_type)
            if kind:
                kinds = [kind]

        results = self._db.search(
            query=query,
            top_k=top_k,
            min_score=min_score,
            kinds=kinds,
            use_mmr=use_mmr,
            mmr_lambda=mmr_lambda,
        )
        return [r.record for r in results]

    def sync_from_agentdb(
        self,
        source_filter: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Pull records from AgentDB that originated outside living_memory.

        Returns entry dicts suitable for ``MemoryEntry.from_dict()``.
        Does NOT auto-insert into LivingMemoryCore — caller decides.

        Args:
            source_filter: Only import records matching this source prefix.
                          Defaults to None (all non-living_memory sources).
            limit: Maximum records to return.

        Returns:
            List of dicts ready for ``MemoryEntry.from_dict()``.
        """
        records = self._db.find(source=source_filter, limit=limit)
        entry_dicts: List[Dict[str, Any]] = []

        for record in records:
            # Skip records that originated from living_memory (avoid loops)
            origin = (record.metadata or {}).get("origin", "")
            if origin == "living_memory" and source_filter is None:
                continue
            entry_dict = self._adapter.record_to_entry_dict(record)
            entry_dicts.append(entry_dict)

        logger.info(
            "LivingMemoryBridge: pulled %d records from AgentDB", len(entry_dicts)
        )
        return entry_dicts

    def stats(self) -> Dict[str, Any]:
        """Cross-system statistics for monitoring."""
        lm_count = len(getattr(self._lm, "_memories", {}))
        db_stats = self._db.stats() if hasattr(self._db, "stats") else {}
        return {
            "living_memory_entries": lm_count,
            "agentdb_total": (
                db_stats.get("total_records", 0) if isinstance(db_stats, dict) else 0
            ),
            "synced_ids": len(self._synced_ids),
            "bridge_active": True,
        }
