"""
Living Memory Core — Self-Organizing Knowledge Store

The heart of BIZRA's living memory system, implementing:
- Episodic memory (experiences)
- Semantic memory (facts)
- Procedural memory (skills)
- Working memory (active context)
- Prospective memory (goals/plans)

Memory lifecycle: ENCODE → CONSOLIDATE → RETRIEVE → FORGET
With self-healing at each stage.
"""

import asyncio
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
)
from core.proof_engine.canonical import hex_digest

# Windows-compatible default storage path
if sys.platform == "win32":
    _DEFAULT_STORAGE = Path.home() / ".bizra" / "memory"
else:
    _DEFAULT_STORAGE = Path("/var/lib/bizra/memory")

logger = logging.getLogger(__name__)


class MemoryType(str, Enum):
    """Types of memory in the living system."""

    EPISODIC = "episodic"  # Event memories (what happened)
    SEMANTIC = "semantic"  # Fact memories (what is true)
    PROCEDURAL = "procedural"  # Skill memories (how to do)
    WORKING = "working"  # Active context (what's now)
    PROSPECTIVE = "prospective"  # Future memories (what will be)


class MemoryState(str, Enum):
    """State of a memory entry."""

    ACTIVE = "active"  # Currently accessible
    CONSOLIDATING = "consolidating"  # Being processed
    ARCHIVED = "archived"  # Stored long-term
    DECAYING = "decaying"  # Fading out
    CORRUPTED = "corrupted"  # Needs repair
    DELETED = "deleted"  # Marked for removal


@dataclass
class MemoryEntry:
    """A single memory entry in the living system."""

    id: str
    content: str
    memory_type: MemoryType
    embedding: Optional[np.ndarray] = None

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0

    # Quality metrics
    ihsan_score: float = 1.0
    snr_score: float = 1.0
    confidence: float = 1.0

    # State
    state: MemoryState = MemoryState.ACTIVE
    source: str = "unknown"

    # Relationships
    related_ids: Set[str] = field(default_factory=set)
    parent_id: Optional[str] = None

    # Decay tracking
    importance: float = 1.0  # Base importance
    emotional_weight: float = 0.5  # Emotional significance

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "ihsan_score": self.ihsan_score,
            "snr_score": self.snr_score,
            "confidence": self.confidence,
            "state": self.state.value,
            "source": self.source,
            "related_ids": list(self.related_ids),
            "importance": self.importance,
            "emotional_weight": self.emotional_weight,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        entry = cls(
            id=data["id"],
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
        )
        entry.created_at = datetime.fromisoformat(
            data.get("created_at", datetime.now(timezone.utc).isoformat())
        )
        entry.last_accessed = datetime.fromisoformat(
            data.get("last_accessed", datetime.now(timezone.utc).isoformat())
        )
        entry.access_count = data.get("access_count", 0)
        entry.ihsan_score = data.get("ihsan_score", 1.0)
        entry.snr_score = data.get("snr_score", 1.0)
        entry.confidence = data.get("confidence", 1.0)
        entry.state = MemoryState(data.get("state", "active"))
        entry.source = data.get("source", "unknown")
        entry.related_ids = set(data.get("related_ids", []))
        entry.importance = data.get("importance", 1.0)
        entry.emotional_weight = data.get("emotional_weight", 0.5)
        return entry

    def compute_retrieval_score(
        self, query_embedding: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute retrieval priority score.

        Formula combines:
        - Recency (when was it last accessed)
        - Frequency (how often accessed)
        - Importance (base importance + emotional weight)
        - Relevance (embedding similarity if available)
        """
        now = datetime.now(timezone.utc)
        hours_since_access = (now - self.last_accessed).total_seconds() / 3600

        # Recency decay (exponential)
        recency = np.exp(-hours_since_access / 168)  # 1 week half-life

        # Frequency bonus (logarithmic)
        frequency = np.log1p(self.access_count) / 10

        # Importance (weighted)
        importance = 0.7 * self.importance + 0.3 * self.emotional_weight

        # Relevance (if query provided)
        relevance = 0.5  # Default
        if query_embedding is not None and self.embedding is not None:
            similarity = np.dot(query_embedding, self.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(self.embedding) + 1e-10
            )
            relevance = max(0, similarity)

        # Combined score
        score = 0.3 * recency + 0.1 * frequency + 0.2 * importance + 0.4 * relevance

        return min(1.0, score)


@dataclass
class MemoryStats:
    """Statistics about the living memory system."""

    total_entries: int = 0
    active_entries: int = 0
    archived_entries: int = 0
    corrupted_entries: int = 0

    by_type: Dict[str, int] = field(default_factory=dict)

    avg_ihsan: float = 1.0
    avg_snr: float = 1.0
    avg_access_count: float = 0.0

    last_consolidation: Optional[datetime] = None
    last_cleanup: Optional[datetime] = None

    memory_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_entries": self.total_entries,
            "active_entries": self.active_entries,
            "archived_entries": self.archived_entries,
            "corrupted_entries": self.corrupted_entries,
            "by_type": self.by_type,
            "avg_ihsan": self.avg_ihsan,
            "avg_snr": self.avg_snr,
            "avg_access_count": self.avg_access_count,
            "last_consolidation": (
                self.last_consolidation.isoformat() if self.last_consolidation else None
            ),
            "last_cleanup": (
                self.last_cleanup.isoformat() if self.last_cleanup else None
            ),
            "memory_bytes": self.memory_bytes,
        }


# Stopwords for keyword extraction (common English words that add noise)
_STOPWORDS = frozenset(
    "a an the is was were be been being have has had do does did will would "
    "shall should can could may might must need dare to of in for on with at "
    "by from up about into over after i me my we our you your he she it they "
    "them his her its their this that these those and but or nor not so yet "
    "if then else when while as than both each few more most other some such "
    "no nor only own same too very just don doesn didn wasn weren isn aren "
    "what which who whom how all any".split()
)


# Pre-compiled regex for keyword extraction (Arabic + Latin, 3+ chars)
_KEYWORD_RE = re.compile(r"[a-zA-Z\u0600-\u06FF]{3,}")


def _extract_keywords(text: str) -> Set[str]:
    """Extract meaningful keywords from text for retrieval matching."""
    words = _KEYWORD_RE.findall(text.lower())
    return {w for w in words if w not in _STOPWORDS}


def _keyword_relevance(query_keywords: Set[str], content: str) -> float:
    """Compute keyword overlap relevance between query and content (0.0-1.0)."""
    if not query_keywords:
        return 0.0
    content_words = set(_KEYWORD_RE.findall(content.lower()))
    overlap = query_keywords & content_words
    if not overlap:
        return 0.0
    # Jaccard-like but weighted toward query coverage
    coverage = len(overlap) / len(query_keywords)
    return min(1.0, coverage)


class LivingMemoryCore:
    """
    The central living memory system.

    Self-organizing, self-healing, self-optimizing memory that:
    - Encodes new information with quality validation
    - Consolidates memories during idle periods
    - Retrieves relevant knowledge proactively
    - Forgets low-value information gracefully
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        embedding_fn: Optional[Callable[[str], np.ndarray]] = None,
        llm_fn: Optional[Callable[[str], str]] = None,
        max_entries: int = 100_000,
        ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD,
    ):
        self.storage_path = storage_path or _DEFAULT_STORAGE
        self.embedding_fn = embedding_fn
        self.llm_fn = llm_fn
        self.max_entries = max_entries
        self.ihsan_threshold = ihsan_threshold

        # SQLite persistence backend (lazy init)
        self._store: Optional["SQLiteMemoryStore"] = None

        # Memory stores by type
        self._memories: Dict[str, MemoryEntry] = {}
        self._type_index: Dict[MemoryType, Set[str]] = {t: set() for t in MemoryType}
        self._embedding_index: Dict[str, np.ndarray] = {}

        # Working memory (limited size)
        self._working_memory: List[str] = []
        self._working_memory_limit = 20

        # State
        self._initialized = False
        self._consolidation_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the living memory system."""
        if self._initialized:
            return

        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize SQLite backend
        from .persistence import SQLiteMemoryStore

        db_path = self.storage_path / "memory.db"
        self._store = SQLiteMemoryStore(db_path)
        self._store.initialize()

        # Auto-migrate from JSONL if legacy file exists
        jsonl_file = self.storage_path / "memories.jsonl"
        if jsonl_file.exists() and not db_path.exists():
            logger.info("Migrating from JSONL to SQLite...")
            self._store.migrate_from_jsonl(jsonl_file)

        # Load persisted memories from SQLite
        await self._load_memories()

        self._initialized = True
        logger.info(f"Living Memory initialized with {len(self._memories)} entries")

    async def _load_memories(self) -> None:
        """Load memories from SQLite persistent storage."""
        if self._store is None:
            return

        try:
            loaded = self._store.load_all()  # Dict[str, MemoryEntry]
            for entry_id, entry in loaded.items():
                self._memories[entry_id] = entry
                self._type_index[entry.memory_type].add(entry_id)
                if entry.embedding is not None:
                    self._embedding_index[entry_id] = entry.embedding
        except Exception as e:
            logger.error(f"Failed to load memories from SQLite: {e}")

    async def _save_memories(self) -> None:
        """Persist memories to SQLite storage."""
        if self._store is None:
            return

        try:
            active = [
                e for e in self._memories.values() if e.state != MemoryState.DELETED
            ]
            self._store.save_batch(active)
        except Exception as e:
            logger.error(f"Failed to save memories to SQLite: {e}")

    async def _save_entry(self, entry: MemoryEntry) -> None:
        """Persist a single entry (incremental save)."""
        if self._store is not None:
            try:
                self._store.save_entry(entry)
            except Exception as e:
                logger.error(f"Failed to save entry {entry.id[:8]}: {e}")

    def _generate_id(self, content: str) -> str:
        """Generate unique ID for memory entry."""
        timestamp = datetime.now(timezone.utc).isoformat()
        hash_input = f"{content[:100]}_{timestamp}"
        return hex_digest(hash_input.encode())[:16]

    async def _compute_embedding(self, content: str) -> Optional[np.ndarray]:
        """Compute embedding for content."""
        if self.embedding_fn:
            try:
                return self.embedding_fn(content)
            except Exception as e:
                logger.warning(f"Embedding computation failed: {e}")
        return None

    async def _compute_quality(self, content: str) -> Tuple[float, float]:
        """Compute Ihsān and SNR scores for content."""
        # Heuristic quality estimation
        words = content.split()

        if not words:
            return 0.0, 0.0

        # Ihsān: Well-formedness
        has_structure = len(words) > 3
        unique_ratio = len(set(words)) / len(words)
        ihsan = 0.5 * float(has_structure) + 0.5 * unique_ratio

        # SNR: Information density
        avg_word_len = sum(len(w) for w in words) / len(words)
        snr = min(avg_word_len / 8, 1.0)

        return min(ihsan, 1.0), min(snr, 1.0)

    async def encode(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        source: str = "user",
        importance: float = 0.5,
        emotional_weight: float = 0.5,
        related_ids: Optional[Set[str]] = None,
    ) -> Optional[MemoryEntry]:
        """
        Encode new information into living memory.

        Validates quality before storing.
        """
        if not content.strip():
            return None

        # Quality check
        ihsan, snr = await self._compute_quality(content)

        if ihsan < self.ihsan_threshold * 0.8:  # Soft threshold for memory
            logger.warning(f"Content below quality threshold: ihsan={ihsan:.3f}")
            return None

        # Create entry
        entry_id = self._generate_id(content)

        entry = MemoryEntry(
            id=entry_id,
            content=content,
            memory_type=memory_type,
            ihsan_score=ihsan,
            snr_score=snr,
            source=source,
            importance=importance,
            emotional_weight=emotional_weight,
            related_ids=related_ids or set(),
        )

        # Compute embedding
        entry.embedding = await self._compute_embedding(content)
        if entry.embedding is not None:
            self._embedding_index[entry_id] = entry.embedding

        # Store
        self._memories[entry_id] = entry
        self._type_index[memory_type].add(entry_id)

        # Persist immediately
        await self._save_entry(entry)

        # Add to working memory if episodic or working type
        if memory_type in (MemoryType.EPISODIC, MemoryType.WORKING):
            self._working_memory.append(entry_id)
            if len(self._working_memory) > self._working_memory_limit:
                # Overflow to long-term
                overflow_id = self._working_memory.pop(0)
                if overflow_id in self._memories:
                    self._memories[overflow_id].memory_type = MemoryType.EPISODIC

        # Check memory limits
        if len(self._memories) > self.max_entries:
            await self._cleanup_memories()

        logger.debug(f"Encoded memory {entry_id[:8]}... type={memory_type.value}")
        return entry

    async def retrieve(
        self,
        query: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        top_k: int = 10,
        min_score: float = 0.1,
    ) -> List[MemoryEntry]:
        """
        Retrieve relevant memories.

        Uses embedding similarity when available, falls back to keyword matching.
        """
        candidates = []

        # Get query embedding
        query_embedding = None
        if query and self.embedding_fn:
            query_embedding = await self._compute_embedding(query)

        # Build keyword set for text-based retrieval fallback
        query_keywords = set()
        if query and not self.embedding_fn:
            query_keywords = _extract_keywords(query)

        # Filter by type if specified
        if memory_type:
            candidate_ids = self._type_index.get(memory_type, set())
        else:
            candidate_ids = set(self._memories.keys())

        # Score candidates
        for entry_id in candidate_ids:
            entry = self._memories.get(entry_id)
            if entry and entry.state == MemoryState.ACTIVE:
                score = entry.compute_retrieval_score(query_embedding)

                # Keyword boost when no embeddings available
                if query_keywords and not self.embedding_fn:
                    keyword_score = _keyword_relevance(query_keywords, entry.content)
                    # Replace the default 0.5 relevance with keyword score
                    # Retrieval formula: 0.3*recency + 0.1*freq + 0.2*imp + 0.4*rel
                    # Default rel=0.5 contributes 0.2 → replace with keyword score
                    score = score - 0.4 * 0.5 + 0.4 * keyword_score

                if score >= min_score:
                    candidates.append((entry, score))

        # Sort by score and return top-k
        candidates.sort(key=lambda x: x[1], reverse=True)

        results = []
        for entry, score in candidates[:top_k]:
            # Update access stats
            entry.last_accessed = datetime.now(timezone.utc)
            entry.access_count += 1
            results.append(entry)

        return results

    async def forget(
        self,
        entry_id: str,
        hard_delete: bool = False,
    ) -> bool:
        """
        Forget a memory entry.

        Soft delete marks as deleted, hard delete removes permanently.
        """
        if entry_id not in self._memories:
            return False

        entry = self._memories[entry_id]

        if hard_delete:
            # Remove from all indexes
            self._type_index[entry.memory_type].discard(entry_id)
            if entry_id in self._embedding_index:
                del self._embedding_index[entry_id]
            if entry_id in self._working_memory:
                self._working_memory.remove(entry_id)
            del self._memories[entry_id]
        else:
            entry.state = MemoryState.DELETED

        return True

    async def _cleanup_memories(self) -> int:
        """Clean up low-value memories to stay within limits."""
        if len(self._memories) <= self.max_entries * 0.9:
            return 0

        # Score all memories
        scored = []
        for entry_id, entry in self._memories.items():
            if entry.state == MemoryState.ACTIVE:
                score = entry.compute_retrieval_score()
                scored.append((entry_id, score, entry))

        # Sort by score (lowest first)
        scored.sort(key=lambda x: x[1])

        # Remove lowest-scoring entries until under limit
        target = int(self.max_entries * 0.8)
        to_remove = len(self._memories) - target
        removed = 0

        for entry_id, score, entry in scored[:to_remove]:
            await self.forget(entry_id, hard_delete=True)
            removed += 1

        logger.info(f"Cleaned up {removed} low-value memories")
        return removed

    async def consolidate(self) -> Dict[str, int]:
        """
        Consolidate memories (merge similar, archive old).

        This is the "sleep" function of living memory.
        """
        async with self._consolidation_lock:
            stats = {"merged": 0, "archived": 0, "repaired": 0}

            now = datetime.now(timezone.utc)
            archive_threshold = now - timedelta(days=7)

            for entry_id, entry in list(self._memories.items()):
                # Archive old episodic memories
                if (
                    entry.memory_type == MemoryType.EPISODIC
                    and entry.last_accessed < archive_threshold
                    and entry.state == MemoryState.ACTIVE
                ):
                    entry.state = MemoryState.ARCHIVED
                    stats["archived"] += 1

                # Repair corrupted entries
                if entry.state == MemoryState.CORRUPTED:
                    # Attempt repair by recomputing quality
                    ihsan, snr = await self._compute_quality(entry.content)
                    entry.ihsan_score = ihsan
                    entry.snr_score = snr
                    if ihsan >= self.ihsan_threshold * 0.8:
                        entry.state = MemoryState.ACTIVE
                        stats["repaired"] += 1

            # Save after consolidation
            await self._save_memories()

            return stats

    def get_working_context(self, max_entries: int = 10) -> str:
        """Get current working memory context as text."""
        context_parts = []

        for entry_id in self._working_memory[-max_entries:]:
            if entry_id in self._memories:
                entry = self._memories[entry_id]
                context_parts.append(f"[{entry.memory_type.value}] {entry.content}")

        return "\n".join(context_parts)

    def get_stats(self) -> MemoryStats:
        """Get statistics about the memory system."""
        stats = MemoryStats()

        stats.total_entries = len(self._memories)

        ihsan_sum = 0.0
        snr_sum = 0.0
        access_sum = 0

        for entry in self._memories.values():
            if entry.state == MemoryState.ACTIVE:
                stats.active_entries += 1
            elif entry.state == MemoryState.ARCHIVED:
                stats.archived_entries += 1
            elif entry.state == MemoryState.CORRUPTED:
                stats.corrupted_entries += 1

            # Count by type
            type_key = entry.memory_type.value
            stats.by_type[type_key] = stats.by_type.get(type_key, 0) + 1

            ihsan_sum += entry.ihsan_score
            snr_sum += entry.snr_score
            access_sum += entry.access_count

        if stats.total_entries > 0:
            stats.avg_ihsan = ihsan_sum / stats.total_entries
            stats.avg_snr = snr_sum / stats.total_entries
            stats.avg_access_count = access_sum / stats.total_entries

        return stats
