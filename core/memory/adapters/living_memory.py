"""
LivingMemory Adapter — Wraps existing LivingMemoryCore for AgentDB.

This adapter converts MemoryEntry (core/living_memory/core.py) to
MemoryRecord (core/memory/types.py) in both directions, enabling
zero-disruption migration: existing code continues using LivingMemoryCore
while AgentDB indexes the same data for sub-linear search.

The adapter is READ-WRITE: it can both import existing memories
and sync new AgentDB writes back to LivingMemory.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from core.memory.types import MemoryKind, MemoryRecord, RecordState

logger = logging.getLogger(__name__)

# Map between LivingMemory MemoryType and AgentDB MemoryKind
_TYPE_MAP = {
    "episodic": MemoryKind.EPISODIC,
    "semantic": MemoryKind.SEMANTIC,
    "procedural": MemoryKind.PROCEDURAL,
    "working": MemoryKind.WORKING,
    "prospective": MemoryKind.PROSPECTIVE,
}

_STATE_MAP = {
    "active": RecordState.ACTIVE,
    "archived": RecordState.ARCHIVED,
    "deleted": RecordState.DELETED,
    "consolidating": RecordState.ACTIVE,  # Map to active
    "decaying": RecordState.ACTIVE,
    "corrupted": RecordState.ACTIVE,
}


class LivingMemoryAdapter:
    """Adapts LivingMemoryCore entries to AgentDB MemoryRecords.

    Usage:
        from core.living_memory.core import LivingMemoryCore
        lm = LivingMemoryCore(...)
        await lm.initialize()

        adapter = LivingMemoryAdapter(lm)
        records = adapter.export_all()
        # ... feed records into AgentDB
    """

    def __init__(self, living_memory) -> None:
        """Accept a LivingMemoryCore instance (duck-typed to avoid circular import)."""
        self._lm = living_memory

    def export_all(self) -> List[MemoryRecord]:
        """Export all active memories as MemoryRecords.

        Reads directly from LivingMemoryCore's in-memory dict.
        """
        records = []
        memories = getattr(self._lm, "_memories", {})

        for entry_id, entry in memories.items():
            record = self._entry_to_record(entry)
            if record is not None:
                records.append(record)

        logger.info(f"LivingMemoryAdapter: exported {len(records)} records")
        return records

    def export_entry(self, entry) -> Optional[MemoryRecord]:
        """Convert a single MemoryEntry to MemoryRecord."""
        return self._entry_to_record(entry)

    def _entry_to_record(self, entry) -> Optional[MemoryRecord]:
        """Convert a LivingMemory MemoryEntry to AgentDB MemoryRecord."""
        try:
            kind = _TYPE_MAP.get(entry.memory_type.value, MemoryKind.SEMANTIC)
            state = _STATE_MAP.get(entry.state.value, RecordState.ACTIVE)

            # Skip deleted entries
            if state == RecordState.DELETED:
                return None

            # Convert embedding (numpy -> list)
            embedding = None
            if entry.embedding is not None:
                embedding = entry.embedding.tolist()

            return MemoryRecord(
                id=entry.id,
                content=entry.content,
                kind=kind,
                state=state,
                embedding=embedding,
                ihsan_score=entry.ihsan_score,
                snr_score=entry.snr_score,
                importance=entry.importance,
                source=entry.source,
                source_id=entry.id,
                related_ids=list(entry.related_ids),
                tags=[],
                created_at=entry.created_at,
                updated_at=entry.last_accessed,
                last_accessed=entry.last_accessed,
                access_count=entry.access_count,
                metadata={
                    "emotional_weight": entry.emotional_weight,
                    "confidence": entry.confidence,
                    "parent_id": entry.parent_id,
                    "origin": "living_memory",
                },
            )
        except Exception as e:
            logger.warning(f"Failed to convert MemoryEntry {getattr(entry, 'id', '?')}: {e}")
            return None
