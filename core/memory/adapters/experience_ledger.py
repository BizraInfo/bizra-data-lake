"""
Experience Ledger Adapter — READ-ONLY bridge from SEL to AgentDB.

The SovereignExperienceLedger (SEL) uses a hash-chained episodic store
with content-addressable integrity. Writing to it from outside would
break the chain — so this adapter is STRICTLY READ-ONLY.

Episodes are exported as MemoryRecords with kind=EPISODIC and
source="experience_ledger".

Constitutional invariant: NO write path exists. This is compiled ethics.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional

from core.memory.types import MemoryKind, MemoryRecord, RecordState

logger = logging.getLogger(__name__)


class ExperienceLedgerAdapter:
    """Read-only adapter wrapping SovereignExperienceLedger for AgentDB.

    Usage:
        from core.sovereign.experience_ledger import SovereignExperienceLedger
        sel = SovereignExperienceLedger()

        adapter = ExperienceLedgerAdapter(sel)
        records = adapter.export_all()
    """

    def __init__(self, sel) -> None:
        """Accept a SovereignExperienceLedger instance (duck-typed)."""
        self._sel = sel

    def export_all(self) -> List[MemoryRecord]:
        """Export all episodes as MemoryRecords.

        This is a snapshot — the SEL may grow after export.
        """
        records = []
        episodes = getattr(self._sel, "_episodes", [])

        for episode in episodes:
            record = self._episode_to_record(episode)
            if record is not None:
                records.append(record)

        logger.info(f"ExperienceLedgerAdapter: exported {len(records)} episodes")
        return records

    def export_recent(self, limit: int = 100) -> List[MemoryRecord]:
        """Export the most recent episodes."""
        episodes = getattr(self._sel, "_episodes", [])
        recent = list(episodes)[-limit:]
        records = []
        for ep in recent:
            record = self._episode_to_record(ep)
            if record is not None:
                records.append(record)
        return records

    def _episode_to_record(self, episode) -> Optional[MemoryRecord]:
        """Convert an Episode to a MemoryRecord."""
        try:
            content_hash = getattr(episode, "content_hash", "")
            query_text = getattr(episode, "query_text", "")
            response_text = getattr(episode, "response_text", "")
            snr = getattr(episode, "snr_score", 0.0)
            ihsan = getattr(episode, "ihsan_score", 0.0)
            importance = getattr(episode, "importance", 0.5)
            timestamp = getattr(episode, "timestamp_secs", 0)

            # Build content from query + response
            content = f"Q: {query_text}\nA: {response_text}" if query_text else str(episode)

            # Convert epoch seconds to datetime
            if timestamp > 0:
                created = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            else:
                created = datetime.now(timezone.utc)

            # Get embedding if available
            embedding = None
            raw_embedding = getattr(episode, "embedding", None)
            if raw_embedding is not None:
                if hasattr(raw_embedding, "tolist"):
                    embedding = raw_embedding.tolist()
                elif isinstance(raw_embedding, list):
                    embedding = raw_embedding

            return MemoryRecord(
                id=content_hash[:16] if content_hash else f"sel_{timestamp}",
                content=content,
                kind=MemoryKind.EPISODIC,
                state=RecordState.ACTIVE,
                embedding=embedding,
                ihsan_score=ihsan,
                snr_score=snr,
                importance=importance,
                source="experience_ledger",
                source_id=content_hash,
                tags=["episode", "sel"],
                created_at=created,
                updated_at=created,
                last_accessed=created,
                metadata={
                    "chain_prev": getattr(episode, "prev_hash", None),
                    "sequence": getattr(episode, "sequence_number", None),
                    "verdict": getattr(episode, "verdict", None),
                    "origin": "experience_ledger",
                },
            )
        except Exception as e:
            logger.warning(f"Failed to convert Episode: {e}")
            return None
