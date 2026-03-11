"""
Memory Types — Shared data structures for the unified memory system.

MemoryRecord is the canonical internal representation. All adapters
convert their native types to/from MemoryRecord.

Standing on Giants: Content-addressable storage (Merkle, 1979)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence


class MemoryKind(str, Enum):
    """Kind of memory stored."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"
    PROSPECTIVE = "prospective"


class RecordState(str, Enum):
    """Lifecycle state of a memory record."""

    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


@dataclass
class MemoryRecord:
    """Canonical memory record in the unified store.

    Every memory — regardless of origin (LivingMemory, SEL, PatternMemory)
    — is normalized to this shape before storage in AgentDB.
    """

    id: str
    content: str
    kind: MemoryKind = MemoryKind.SEMANTIC
    state: RecordState = RecordState.ACTIVE

    # Embedding (float32 vector, typically dim=768)
    embedding: Optional[List[float]] = None

    # Quality scores (from constants.py thresholds)
    ihsan_score: float = 1.0
    snr_score: float = 1.0
    importance: float = 0.5

    # Provenance
    source: str = "unknown"
    source_id: Optional[str] = None  # Original ID in source system

    # Relationships
    related_ids: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0

    # Extensible metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "kind": self.kind.value,
            "state": self.state.value,
            "ihsan_score": self.ihsan_score,
            "snr_score": self.snr_score,
            "importance": self.importance,
            "source": self.source,
            "source_id": self.source_id,
            "related_ids": self.related_ids,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "metadata": self.metadata,
        }


@dataclass
class SearchResult:
    """A single search result with fused score breakdown."""

    record: MemoryRecord
    score: float  # Final fused score (0.0 - 1.0)

    # Score components for transparency
    vector_score: float = 0.0
    keyword_score: float = 0.0
    recency_score: float = 0.0
    importance_score: float = 0.0
    graph_score: float = 0.0


@dataclass
class QueryOptions:
    """Options for memory search queries."""

    query_text: Optional[str] = None
    query_embedding: Optional[Sequence[float]] = None
    top_k: int = 10
    min_score: float = 0.1
    kinds: Optional[List[MemoryKind]] = None
    tags: Optional[List[str]] = None
    source: Optional[str] = None
    include_archived: bool = False

    # MMR (Maximal Marginal Relevance) — diversifies results
    use_mmr: bool = False
    mmr_lambda: float = 0.5  # 0.0=max diversity, 1.0=max relevance

    # Metadata filters (key → value or {$gte/$lte/$in/$contains} dict)
    metadata_filters: Optional[Dict[str, Any]] = None
