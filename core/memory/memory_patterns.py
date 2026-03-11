"""
AgentDB Memory Patterns — High-level memory management for AI agents.

Provides five pattern classes built on the AgentDB facade:

  SessionMemory        — Conversation-scoped turn tracking with synthesis
  FactStore            — Category/key persistent facts with confidence
  HierarchicalMemory   — 4-tier organization (immediate/short/long/semantic)
  MemoryConsolidator   — Importance-based pruning, dedup, compaction
  ContextSynthesizer   — Multi-memory coherent context generation

Maps to BIZRA DDAGI OS architecture (Spine §2–§3):
  SessionMemory      → P7 DEMA conversation history
  FactStore          → P2 Researcher persistent knowledge
  HierarchicalMemory → Triple Helix tiers (Reactive / Deliberative / Evolutionary)
  MemoryConsolidator → Helix 3 evolutionary heartbeat (60s consolidation)
  ContextSynthesizer → Mission context preparation for PAT ensemble

Standing on Giants:
  Atkinson & Shiffrin (1968) — Multi-store memory model
  Tulving (1972) — Episodic vs semantic memory distinction
  Baddeley (2000) — Working memory model
  Anderson (1983) — ACT-R decay and strengthening
  Carbonell & Goldstein (1998) — MMR diversity (used in retrieval)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

from .agent_db import AgentDB
from .types import MemoryKind, MemoryRecord, RecordState, SearchResult

logger = logging.getLogger(__name__)


# ── Session Memory ───────────────────────────────────────────────────────


class SessionMemory:
    """Session-scoped conversation memory.

    Each turn is stored as an EPISODIC MemoryRecord with session_id
    in metadata. Supports retrieval of conversation history and
    synthesis of recent context for prompt injection.

    Maps to P7 DEMA → tracks user↔agent dialogue per session.
    """

    def __init__(self, db: AgentDB, session_id: str) -> None:
        self._db = db
        self._session_id = session_id
        self._turn_counter = 0

    @property
    def session_id(self) -> str:
        return self._session_id

    def store_turn(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
    ) -> MemoryRecord:
        """Store a single conversation turn.

        Args:
            role: Speaker role ("user", "assistant", "system").
            content: Turn content text.
            metadata: Additional turn metadata.
            importance: Importance weight (default 0.5).

        Returns:
            The stored MemoryRecord.
        """
        self._turn_counter += 1
        turn_meta = {
            "session_id": self._session_id,
            "role": role,
            "turn_index": self._turn_counter,
            **(metadata or {}),
        }

        return self._db.store(
            content=content,
            kind=MemoryKind.EPISODIC,
            importance=importance,
            source=f"session:{self._session_id}",
            tags=["session", f"role:{role}", f"session:{self._session_id}"],
            metadata=turn_meta,
        )

    def get_history(self, limit: int = 20) -> List[MemoryRecord]:
        """Retrieve conversation history for this session, ordered by turn index.

        Args:
            limit: Maximum turns to return.

        Returns:
            List of MemoryRecord in chronological order.
        """
        records = self._db.find(
            source=f"session:{self._session_id}",
            kind=MemoryKind.EPISODIC,
            tags=["session"],
            limit=limit,
        )
        records.sort(key=lambda r: r.metadata.get("turn_index", 0))
        return records

    def get_recent_context(self, limit: int = 5) -> str:
        """Synthesize recent turns into a context string.

        Args:
            limit: Number of recent turns to include.

        Returns:
            Formatted conversation context string.
        """
        records = self.get_history(limit=limit)
        # Take the last `limit` records
        recent = records[-limit:] if len(records) > limit else records
        lines = []
        for r in recent:
            role = r.metadata.get("role", "unknown")
            lines.append(f"[{role}]: {r.content}")
        return "\n".join(lines)

    def clear_session(self, archive: bool = True) -> int:
        """Clear all turns for this session.

        Args:
            archive: If True, soft-delete (archive). If False, hard-delete.

        Returns:
            Number of records cleared.
        """
        records = self._db.find(
            source=f"session:{self._session_id}",
            kind=MemoryKind.EPISODIC,
            tags=["session"],
            limit=10000,
        )
        count = 0
        for r in records:
            self._db.forget(r.id, hard=not archive)
            count += 1
        return count


# ── Fact Store ───────────────────────────────────────────────────────────


class FactStore:
    """Category-based persistent fact storage.

    Facts are stored as SEMANTIC MemoryRecords with structured
    metadata (category, key, confidence, source). Supports
    category-scoped retrieval and confidence updates.

    Maps to P2 Researcher → persistent knowledge base.

    Standing on Giants: Anderson's ACT-R (1983) — activation-based retrieval
    """

    SOURCE_PREFIX = "fact_store"

    def __init__(self, db: AgentDB) -> None:
        self._db = db

    def _fact_id(self, category: str, key: str) -> str:
        """Deterministic ID for a category/key pair."""
        raw = f"fact:{category}:{key}"
        return hashlib.blake2b(raw.encode(), digest_size=8).hexdigest()

    def store_fact(
        self,
        category: str,
        key: str,
        value: str,
        confidence: float = 1.0,
        source: str = "explicit",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryRecord:
        """Store or update a fact.

        Args:
            category: Fact category (e.g., "user_preference", "system_config").
            key: Fact key within category.
            value: Fact value (text content).
            confidence: Confidence level [0.0, 1.0].
            source: Origin of the fact.
            metadata: Additional metadata.

        Returns:
            The stored MemoryRecord.
        """
        fact_meta = {
            "category": category,
            "key": key,
            "confidence": confidence,
            "fact_source": source,
            **(metadata or {}),
        }

        return self._db.store(
            content=value,
            kind=MemoryKind.SEMANTIC,
            importance=confidence,
            source=f"{self.SOURCE_PREFIX}:{category}",
            tags=["fact", f"category:{category}", f"key:{key}"],
            metadata=fact_meta,
        )

    def get_fact(self, category: str, key: str) -> Optional[MemoryRecord]:
        """Retrieve a specific fact by category and key.

        Returns:
            The MemoryRecord or None if not found.
        """
        records = self._db.find(
            source=f"{self.SOURCE_PREFIX}:{category}",
            tags=[f"key:{key}"],
            limit=1,
        )
        return records[0] if records else None

    def get_facts(
        self, category: str, min_confidence: float = 0.0
    ) -> List[MemoryRecord]:
        """Retrieve all facts in a category.

        Args:
            category: Fact category to retrieve.
            min_confidence: Minimum confidence threshold.

        Returns:
            List of MemoryRecord matching the category.
        """
        records = self._db.find(
            source=f"{self.SOURCE_PREFIX}:{category}",
            tags=["fact"],
            limit=1000,
        )
        if min_confidence > 0.0:
            records = [
                r
                for r in records
                if r.metadata.get("confidence", 0.0) >= min_confidence
            ]
        return records

    def update_confidence(self, category: str, key: str, new_confidence: float) -> bool:
        """Update a fact's confidence score.

        Returns:
            True if the fact was found and updated.
        """
        record = self.get_fact(category, key)
        if record is None:
            return False

        # Re-store with updated confidence (upsert by content-addressable ID)
        record.metadata["confidence"] = new_confidence
        record.importance = new_confidence
        self._db.store_record(record)
        return True

    def forget_fact(self, category: str, key: str, hard: bool = False) -> bool:
        """Remove a fact.

        Returns:
            True if the fact was found and removed.
        """
        record = self.get_fact(category, key)
        if record is None:
            return False
        return self._db.forget(record.id, hard=hard)


# ── Hierarchical Memory ─────────────────────────────────────────────────


class MemoryTier:
    """Memory tier constants mapping to Triple Helix (Spine §3)."""

    IMMEDIATE = "immediate"  # Reactive (System-1): working memory, <50ms
    SHORT_TERM = "short_term"  # Session-scoped episodic memory
    LONG_TERM = "long_term"  # Cross-session semantic + procedural
    SEMANTIC = "semantic"  # Deep vector search across all tiers

    _TAG_MAP = {
        IMMEDIATE: "tier:immediate",
        SHORT_TERM: "tier:short_term",
        LONG_TERM: "tier:long_term",
    }

    _KIND_MAP = {
        IMMEDIATE: MemoryKind.WORKING,
        SHORT_TERM: MemoryKind.EPISODIC,
        LONG_TERM: MemoryKind.SEMANTIC,
    }


class HierarchicalMemory:
    """4-tier memory organization.

    Standing on Giants: Atkinson & Shiffrin (1968) — multi-store model

    Tiers:
      IMMEDIATE  — Working memory for current task (high recency weight)
      SHORT_TERM — Session-scoped episodic memories
      LONG_TERM  — Cross-session facts, patterns, knowledge
      SEMANTIC   — Deep vector search across all tiers (virtual tier)
    """

    # Capacity limits per tier (records)
    IMMEDIATE_CAPACITY = 50
    SHORT_TERM_CAPACITY = 500

    def __init__(self, db: AgentDB) -> None:
        self._db = db

    def store(
        self,
        content: str,
        tier: str = MemoryTier.SHORT_TERM,
        importance: float = 0.5,
        source: str = "hierarchical",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[Sequence[float]] = None,
    ) -> MemoryRecord:
        """Store a memory in a specific tier.

        Args:
            content: Memory content.
            tier: One of MemoryTier constants.
            importance: Importance weight.
            source: Memory source.
            tags: Additional tags.
            metadata: Additional metadata.
            embedding: Optional pre-computed embedding.

        Returns:
            The stored MemoryRecord.
        """
        tier_tag = MemoryTier._TAG_MAP.get(tier, f"tier:{tier}")
        kind = MemoryTier._KIND_MAP.get(tier, MemoryKind.SEMANTIC)
        tier_meta = {"memory_tier": tier, **(metadata or {})}

        return self._db.store(
            content=content,
            kind=kind,
            embedding=embedding,
            importance=importance,
            source=source,
            tags=[tier_tag, "hierarchical"] + (tags or []),
            metadata=tier_meta,
        )

    def retrieve(
        self,
        query: Optional[str] = None,
        query_embedding: Optional[Sequence[float]] = None,
        tier: Optional[str] = None,
        top_k: int = 10,
        use_mmr: bool = False,
    ) -> List[SearchResult]:
        """Retrieve memories, optionally scoped to a tier.

        For semantic queries (with text or embedding), uses hybrid search.
        For tier-only queries, uses direct metadata filtering.

        Args:
            query: Text query for semantic + keyword search.
            query_embedding: Pre-computed query embedding.
            tier: Tier to search (None = all tiers / SEMANTIC).
            top_k: Max results.
            use_mmr: Enable MMR diversity re-ranking.

        Returns:
            List of SearchResult.
        """
        # If we have a semantic query, use hybrid search
        if query or query_embedding:
            tags = None
            kinds = None
            if tier and tier != MemoryTier.SEMANTIC:
                tier_tag = MemoryTier._TAG_MAP.get(tier, f"tier:{tier}")
                tags = [tier_tag]
                kind = MemoryTier._KIND_MAP.get(tier)
                if kind:
                    kinds = [kind]
            return self._db.search(
                query=query,
                query_embedding=list(query_embedding) if query_embedding else None,
                top_k=top_k,
                kinds=kinds,
                tags=tags,
                min_score=0.0,
            )

        # Metadata-only retrieval by tier
        tags = ["hierarchical"]
        kind = None
        if tier and tier != MemoryTier.SEMANTIC:
            tier_tag = MemoryTier._TAG_MAP.get(tier, f"tier:{tier}")
            tags = [tier_tag]
            kind = MemoryTier._KIND_MAP.get(tier)

        records = self._db.find(tags=tags, kind=kind, limit=top_k)
        return [SearchResult(record=r, score=r.importance) for r in records]

    def promote(self, record_id: str, to_tier: str) -> bool:
        """Promote a memory to a higher tier.

        Args:
            record_id: Record to promote.
            to_tier: Target tier.

        Returns:
            True if promotion succeeded.
        """
        record = self._db.retrieve(record_id)
        if record is None:
            return False

        # Update tier tag and kind
        new_tags = [t for t in record.tags if not t.startswith("tier:")]
        new_tier_tag = MemoryTier._TAG_MAP.get(to_tier, f"tier:{to_tier}")
        new_tags.append(new_tier_tag)

        new_kind = MemoryTier._KIND_MAP.get(to_tier, record.kind)
        record.metadata["memory_tier"] = to_tier

        promoted = MemoryRecord(
            id=record.id,
            content=record.content,
            kind=new_kind,
            state=record.state,
            embedding=record.embedding,
            ihsan_score=record.ihsan_score,
            snr_score=record.snr_score,
            importance=min(record.importance * 1.2, 1.0),  # Boost on promotion
            source=record.source,
            source_id=record.source_id,
            related_ids=record.related_ids,
            tags=new_tags,
            metadata=record.metadata,
            created_at=record.created_at,
            updated_at=datetime.now(timezone.utc),
            last_accessed=record.last_accessed,
            access_count=record.access_count,
        )
        self._db.store_record(promoted)
        return True

    def tier_stats(self) -> Dict[str, int]:
        """Count records per tier."""
        stats = {}
        for tier_name, tier_tag in MemoryTier._TAG_MAP.items():
            records = self._db.find(tags=[tier_tag], limit=10000)
            stats[tier_name] = len(records)
        stats["total"] = self._db.count
        return stats


# ── Memory Consolidator ─────────────────────────────────────────────────


@dataclass
class ConsolidationResult:
    """Result of a memory consolidation pass."""

    pruned: int = 0
    deduplicated: int = 0
    promoted: int = 0
    total_before: int = 0
    total_after: int = 0
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pruned": self.pruned,
            "deduplicated": self.deduplicated,
            "promoted": self.promoted,
            "total_before": self.total_before,
            "total_after": self.total_after,
            "duration_ms": self.duration_ms,
        }


class MemoryConsolidator:
    """Importance-based memory pruning and consolidation.

    Maps to Helix 3 — Evolutionary cycle (60-second heartbeat).
    Runs periodically to maintain memory health.

    Standing on Giants:
      Anderson (1983) — ACT-R memory decay/strengthening
      Deming (1950) — PDCA continuous improvement cycle
    """

    def __init__(self, db: AgentDB) -> None:
        self._db = db

    def consolidate(
        self,
        max_records: int = 10000,
        min_importance: float = 0.1,
        max_age_days: Optional[int] = None,
    ) -> ConsolidationResult:
        """Run a full consolidation pass.

        Strategy: prune low-importance records, then prune expired records,
        keeping the most important and most recent memories.

        Args:
            max_records: Maximum records to retain (prune excess by importance).
            min_importance: Records below this importance are pruned.
            max_age_days: Records older than this are pruned (None = no age limit).

        Returns:
            ConsolidationResult with statistics.
        """
        from time import perf_counter

        started = perf_counter()
        total_before = self._db.count
        pruned = 0

        # Phase 1: Prune by minimum importance
        pruned += self._prune_low_importance(min_importance)

        # Phase 2: Prune by age
        if max_age_days is not None:
            pruned += self._prune_expired(max_age_days)

        # Phase 3: Prune excess if over capacity
        current_count = self._db.count
        if current_count > max_records:
            excess = current_count - max_records
            pruned += self._prune_least_important(excess)

        total_after = self._db.count
        duration_ms = (perf_counter() - started) * 1000

        result = ConsolidationResult(
            pruned=pruned,
            total_before=total_before,
            total_after=total_after,
            duration_ms=round(duration_ms, 3),
        )
        logger.info(
            f"Consolidation complete: pruned={pruned}, "
            f"{total_before}→{total_after} records, "
            f"{duration_ms:.1f}ms"
        )
        return result

    def prune_expired(self, max_age_days: int, min_importance: float = 0.3) -> int:
        """Prune records older than max_age_days with low importance.

        High-importance records are preserved regardless of age.

        Args:
            max_age_days: Age threshold in days.
            min_importance: Records above this importance survive regardless.

        Returns:
            Number of records pruned.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        store = self._db.backend
        conn = store._ensure_conn()

        cursor = conn.execute(
            """
            SELECT id, importance FROM records
            WHERE state = ? AND last_accessed < ? AND importance < ?
            ORDER BY importance ASC
            """,
            (RecordState.ACTIVE.value, cutoff.isoformat(), min_importance),
        )
        pruned = 0
        for row in cursor.fetchall():
            self._db.forget(row["id"])
            pruned += 1
        return pruned

    def deduplicate(self, similarity_threshold: float = 0.95) -> int:
        """Remove near-duplicate records based on content hash similarity.

        For exact content duplicates, keeps the one with higher importance.

        Args:
            similarity_threshold: Not used for exact dedup (reserved for vector dedup).

        Returns:
            Number of duplicates removed.
        """
        store = self._db.backend
        conn = store._ensure_conn()

        # Find exact content duplicates via content hash
        cursor = conn.execute(
            """
            SELECT content, COUNT(*) as cnt
            FROM records
            WHERE state = ?
            GROUP BY content
            HAVING cnt > 1
            """,
            (RecordState.ACTIVE.value,),
        )
        duplicated = 0
        for row in cursor.fetchall():
            dupes = conn.execute(
                """
                SELECT id, importance FROM records
                WHERE state = ? AND content = ?
                ORDER BY importance DESC, access_count DESC
                """,
                (RecordState.ACTIVE.value, row["content"]),
            ).fetchall()
            # Keep the first (highest importance), forget the rest
            for dupe in dupes[1:]:
                self._db.forget(dupe["id"])
                duplicated += 1

        return duplicated

    def compact(self) -> Dict[str, Any]:
        """Run index rebuild and return health stats.

        Returns:
            Dict with rebuild info and current stats.
        """
        rebuild = self._db.rebuild_indexes()
        stats = self._db.stats()
        return {
            "rebuild": rebuild,
            "stats": stats,
        }

    # ── Internal ─────────────────────────────────────────────────────────

    def _prune_low_importance(self, min_importance: float) -> int:
        store = self._db.backend
        conn = store._ensure_conn()
        cursor = conn.execute(
            "SELECT id FROM records WHERE state = ? AND importance < ?",
            (RecordState.ACTIVE.value, min_importance),
        )
        pruned = 0
        for row in cursor.fetchall():
            self._db.forget(row["id"])
            pruned += 1
        return pruned

    def _prune_expired(self, max_age_days: int) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        store = self._db.backend
        conn = store._ensure_conn()
        cursor = conn.execute(
            "SELECT id FROM records WHERE state = ? AND last_accessed < ?",
            (RecordState.ACTIVE.value, cutoff.isoformat()),
        )
        pruned = 0
        for row in cursor.fetchall():
            self._db.forget(row["id"])
            pruned += 1
        return pruned

    def _prune_least_important(self, excess: int) -> int:
        store = self._db.backend
        conn = store._ensure_conn()
        cursor = conn.execute(
            """
            SELECT id FROM records
            WHERE state = ?
            ORDER BY importance ASC, access_count ASC
            LIMIT ?
            """,
            (RecordState.ACTIVE.value, excess),
        )
        pruned = 0
        for row in cursor.fetchall():
            self._db.forget(row["id"])
            pruned += 1
        return pruned


# ── Context Synthesizer ─────────────────────────────────────────────────


@dataclass
class SynthesizedContext:
    """Result of context synthesis from multiple memories."""

    context: str
    sources: List[MemoryRecord] = field(default_factory=list)
    fact_count: int = 0
    episodic_count: int = 0
    procedural_count: int = 0
    total_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context": self.context,
            "source_count": len(self.sources),
            "fact_count": self.fact_count,
            "episodic_count": self.episodic_count,
            "procedural_count": self.procedural_count,
            "total_score": self.total_score,
        }


class ContextSynthesizer:
    """Generate coherent context from multiple retrieved memories.

    Combines facts, episodes, and procedures into a structured
    context suitable for prompt injection or mission briefing.

    Maps to P2 Researcher + P4 Evaluator → context preparation
    for PAT ensemble mission execution.
    """

    def __init__(self, db: AgentDB, fact_store: Optional[FactStore] = None) -> None:
        self._db = db
        self._facts = fact_store or FactStore(db)

    def synthesize(
        self,
        query: str,
        query_embedding: Optional[Sequence[float]] = None,
        top_k: int = 10,
        include_facts: bool = True,
        fact_categories: Optional[List[str]] = None,
        use_mmr: bool = True,
    ) -> SynthesizedContext:
        """Synthesize context from multiple memory sources.

        Args:
            query: Text query for retrieval.
            query_embedding: Optional pre-computed embedding.
            top_k: Number of memories to retrieve.
            include_facts: Whether to include fact store results.
            fact_categories: Specific fact categories to include.
            use_mmr: Use MMR for diverse results.

        Returns:
            SynthesizedContext with formatted context and metadata.
        """
        # Retrieve relevant memories via hybrid search
        results = self._db.search(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            min_score=0.1,
        )

        # Categorize by kind
        facts: List[MemoryRecord] = []
        episodes: List[MemoryRecord] = []
        procedures: List[MemoryRecord] = []
        others: List[MemoryRecord] = []

        for r in results:
            rec = r.record
            if rec.kind == MemoryKind.SEMANTIC:
                facts.append(rec)
            elif rec.kind == MemoryKind.EPISODIC:
                episodes.append(rec)
            elif rec.kind == MemoryKind.PROCEDURAL:
                procedures.append(rec)
            else:
                others.append(rec)

        # Include explicit facts if requested
        if include_facts and fact_categories:
            for cat in fact_categories:
                cat_facts = self._facts.get_facts(cat, min_confidence=0.5)
                for f in cat_facts:
                    if f.id not in {r.record.id for r in results}:
                        facts.append(f)

        # Build structured context
        sections = []

        if facts:
            section_lines = ["## Relevant Knowledge"]
            for f in facts[:5]:
                section_lines.append(f"- {f.content}")
            sections.append("\n".join(section_lines))

        if procedures:
            section_lines = ["## Known Procedures"]
            for p in procedures[:3]:
                section_lines.append(f"- {p.content}")
            sections.append("\n".join(section_lines))

        if episodes:
            section_lines = ["## Related Experiences"]
            for e in episodes[:5]:
                role = e.metadata.get("role", "")
                prefix = f"[{role}] " if role else ""
                section_lines.append(f"- {prefix}{e.content}")
            sections.append("\n".join(section_lines))

        if others:
            section_lines = ["## Additional Context"]
            for o in others[:3]:
                section_lines.append(f"- {o.content}")
            sections.append("\n".join(section_lines))

        context = "\n\n".join(sections) if sections else ""
        total_score = sum(r.score for r in results) / max(len(results), 1)
        all_sources = facts + episodes + procedures + others

        return SynthesizedContext(
            context=context,
            sources=all_sources,
            fact_count=len(facts),
            episodic_count=len(episodes),
            procedural_count=len(procedures),
            total_score=round(total_score, 4),
        )

    def build_prompt_context(
        self,
        query: str,
        max_chars: int = 4000,
        query_embedding: Optional[Sequence[float]] = None,
    ) -> str:
        """Build a compact context string for prompt injection.

        Truncates to max_chars to respect token budgets.

        Args:
            query: Text query.
            max_chars: Maximum character length.
            query_embedding: Optional pre-computed embedding.

        Returns:
            Formatted context string, truncated if necessary.
        """
        synth = self.synthesize(
            query=query,
            query_embedding=query_embedding,
            top_k=8,
            include_facts=False,
        )
        context = synth.context
        if len(context) > max_chars:
            context = context[: max_chars - 3] + "..."
        return context
