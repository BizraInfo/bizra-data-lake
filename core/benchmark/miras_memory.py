"""
MIRAS Memory — Cross-Cycle Memory Substrate (GEM #2 + v9 spec §Phase 02)
═══════════════════════════════════════════════════════════════════════════════

Cross-cycle memory so the dominance loop doesn't rebuild knowledge from
scratch on every iteration. Solves the "Memory Paradox" identified in the
TRUE SPEARPOINT hidden-architecture analysis — currently the system wastes
3–5× compute on redundant ablations because each cycle starts cold.

Architecture — three tiers (Atkinson-Shiffrin multi-store model):
  short_term  — LRU, bounded (capacity=100). New knowledge enters here.
  long_term   — LRU, bounded (capacity=10000). Promoted from short_term.
  episodic    — Unbounded. Action/result log; never evicted.

Quality gate:
  Every store() call is gated on snr_score >= UNIFIED_SNR_THRESHOLD (0.85).
  Low-quality content is rejected with None return.

Retrieval formula (from sovereign_spearpoint.py::MemorySystem.retrieve):
  score = 0.4 × keyword_overlap
        + 0.3 × importance
        + 0.2 × snr_score
        + 0.1 × recency

Standing on Giants:
  Atkinson & Shiffrin (1968) — Multi-store memory model
  Tulving (1972) — Episodic memory
  Shannon (1948) — SNR as quality signal
"""

from __future__ import annotations

import hashlib
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from core.integration.constants import UNIFIED_SNR_THRESHOLD


@dataclass
class MIRASMemoryEntry:
    """A single memory entry across any tier."""

    key: str
    content: str
    tier: Literal["short_term", "long_term", "episodic"]
    timestamp: float
    snr_score: float  # Quality gate: must be >= UNIFIED_SNR_THRESHOLD
    access_count: int = 0
    importance: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Result of a memory retrieval query."""

    entries: List[MIRASMemoryEntry]
    sources: Dict[str, int]  # tier → count
    total_retrieved: int
    dedup_removed: int


class MIRASMemory:
    """
    Three-tier quality-gated cross-cycle memory.

    Usage:
        memory = MIRASMemory()
        key = memory.store("attention head ablation improved MMLU by 2%", snr_score=0.91)
        result = memory.retrieve("MMLU attention", k=5)
        promoted = memory.consolidate()
    """

    def __init__(
        self,
        short_term_capacity: int = 100,
        long_term_capacity: int = 10_000,
        compression_threshold: int = 50,
    ) -> None:
        """
        Args:
            short_term_capacity: Max entries in short-term LRU store.
            long_term_capacity: Max entries in long-term LRU store.
            compression_threshold: access_count at which a short-term entry
                is promoted to long-term during consolidate().
        """
        self._short_term: OrderedDict[str, MIRASMemoryEntry] = OrderedDict()
        self._long_term: OrderedDict[str, MIRASMemoryEntry] = OrderedDict()
        self._episodic: List[MIRASMemoryEntry] = []
        self._short_term_capacity = short_term_capacity
        self._long_term_capacity = long_term_capacity
        self._compression_threshold = compression_threshold

    # ─── Public API ────────────────────────────────────────────────────────────

    def store(
        self,
        content: str,
        snr_score: float,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
    ) -> Optional[str]:
        """
        Quality-gated store of new knowledge.

        Args:
            content: Text content to store.
            snr_score: SNR quality score — rejected if < UNIFIED_SNR_THRESHOLD.
            metadata: Optional structured metadata dict.
            importance: Relevance weight 0–1 (default 0.5).

        Returns:
            Content key (hex string) or None if rejected by quality gate.
        """
        if snr_score < UNIFIED_SNR_THRESHOLD:
            return None

        key = hashlib.sha256(content.encode()).hexdigest()[:16]

        # If already stored in any tier, update importance and return key.
        if key in self._short_term:
            self._short_term[key].importance = max(
                self._short_term[key].importance, importance
            )
            self._short_term.move_to_end(key)
            return key
        if key in self._long_term:
            return key

        # Evict oldest short-term entry if at capacity (LRU policy).
        if len(self._short_term) >= self._short_term_capacity:
            self._short_term.popitem(last=False)

        entry = MIRASMemoryEntry(
            key=key,
            content=content,
            tier="short_term",
            timestamp=time.time(),
            snr_score=snr_score,
            importance=importance,
            metadata=metadata or {},
        )
        self._short_term[key] = entry
        self._short_term.move_to_end(key)
        return key

    def store_episodic(
        self,
        action: str,
        result: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store an action/result pair in the episodic log (unbounded).

        Args:
            action: Description of the action taken.
            result: Observed outcome.
            context: Optional context dict (benchmark name, cost, etc.).
        """
        entry = MIRASMemoryEntry(
            key=str(uuid.uuid4()),
            content=f"ACTION: {action}\nRESULT: {result}",
            tier="episodic",
            timestamp=time.time(),
            snr_score=1.0,  # Episodic log is always accepted
            importance=0.5,
            metadata=context or {},
        )
        self._episodic.append(entry)

    def retrieve(self, query: str, k: int = 10) -> RetrievalResult:
        """
        Retrieve the top-k most relevant memories across all tiers.

        Scoring: 0.4×keyword_overlap + 0.3×importance + 0.2×snr_score + 0.1×recency.

        Args:
            query: Natural-language query string.
            k: Maximum number of results.

        Returns:
            RetrievalResult with sorted entries and tier-source breakdown.
        """
        query_words = set(query.lower().split()) if query.strip() else set()
        now = time.time()

        all_entries: List[MIRASMemoryEntry] = (
            list(self._short_term.values())
            + list(self._long_term.values())
            + self._episodic
        )

        scored: List[tuple[float, MIRASMemoryEntry]] = [
            (self._score_entry(e, query_words, now), e) for e in all_entries
        ]
        scored.sort(key=lambda x: x[0], reverse=True)

        seen_keys: set[str] = set()
        dedup_removed = 0
        result_entries: List[MIRASMemoryEntry] = []
        sources: Dict[str, int] = {"short_term": 0, "long_term": 0, "episodic": 0}

        for _score, entry in scored:
            if entry.key in seen_keys:
                dedup_removed += 1
                continue
            seen_keys.add(entry.key)
            entry.access_count += 1
            result_entries.append(entry)
            sources[entry.tier] = sources.get(entry.tier, 0) + 1
            if len(result_entries) >= k:
                break

        return RetrievalResult(
            entries=result_entries,
            sources=sources,
            total_retrieved=len(result_entries),
            dedup_removed=dedup_removed,
        )

    def consolidate(self) -> int:
        """
        Promote hot short-term entries to long-term storage.

        Entries with access_count >= compression_threshold are promoted.
        Evicts oldest long-term entries if long-term is at capacity.

        Returns:
            Number of entries promoted.
        """
        keys_to_promote = [
            key
            for key, entry in self._short_term.items()
            if entry.access_count >= self._compression_threshold
        ]
        promoted = 0
        for key in keys_to_promote:
            entry = self._short_term.pop(key)
            entry.tier = "long_term"
            if len(self._long_term) >= self._long_term_capacity:
                self._long_term.popitem(last=False)
            self._long_term[key] = entry
            promoted += 1
        return promoted

    def get_stats(self) -> dict:
        """Return current memory statistics."""
        return {
            "short_term_count": len(self._short_term),
            "long_term_count": len(self._long_term),
            "episodic_count": len(self._episodic),
            "total_count": (
                len(self._short_term) + len(self._long_term) + len(self._episodic)
            ),
            "short_term_capacity": self._short_term_capacity,
            "long_term_capacity": self._long_term_capacity,
            "snr_threshold": UNIFIED_SNR_THRESHOLD,
        }

    # ─── Private ───────────────────────────────────────────────────────────────

    def _score_entry(
        self,
        entry: MIRASMemoryEntry,
        query_words: set,
        now: float,
    ) -> float:
        """Compute relevance score for retrieval ranking."""
        entry_words = set(entry.content.lower().split())
        if query_words:
            overlap = len(query_words & entry_words) / len(query_words)
        else:
            overlap = 0.0

        # Recency decays linearly over 24 hours.
        age_hours = (now - entry.timestamp) / 3600.0
        recency = max(0.0, 1.0 - age_hours / 24.0)

        return (
            0.4 * overlap
            + 0.3 * entry.importance
            + 0.2 * entry.snr_score
            + 0.1 * recency
        )
