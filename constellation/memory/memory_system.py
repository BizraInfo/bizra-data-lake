# ═══════════════════════════════════════════════════════════════════════════════
# BIZRA Constellation - Autonomous Memory System v1.0
# ═══════════════════════════════════════════════════════════════════════════════
"""
Multi-tier memory system providing:
- Short-term: Session context and working memory
- Long-term: Persistent knowledge graph storage
- Episodic: Session-based memory with temporal context
- Semantic: Conceptual understanding and relationships
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum
from collections import deque


# ─────────────────────────────────────────────────────────────────────────────
# MEMORY TYPES
# ─────────────────────────────────────────────────────────────────────────────

class MemoryTier(str, Enum):
    """Tiers of memory storage."""
    WORKING = "working"      # Current task context
    SHORT_TERM = "short_term"  # Recent interactions
    LONG_TERM = "long_term"  # Persistent knowledge
    EPISODIC = "episodic"    # Session-based memories
    SEMANTIC = "semantic"    # Conceptual relationships


class MemoryPriority(str, Enum):
    """Priority levels for memory retention."""
    CRITICAL = "critical"    # Never forget
    HIGH = "high"           # Retain for weeks
    MEDIUM = "medium"       # Retain for days
    LOW = "low"            # Retain for hours
    EPHEMERAL = "ephemeral" # Discard after session


@dataclass
class MemoryEntry:
    """A single memory entry."""
    id: str
    tier: MemoryTier
    content: str
    agent_slug: str
    session_id: str
    priority: MemoryPriority = MemoryPriority.MEDIUM
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_accessed: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    access_count: int = 0
    metadata: dict = field(default_factory=dict)
    embedding: Optional[list[float]] = None
    expires_at: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "tier": self.tier.value,
            "content": self.content,
            "agent_slug": self.agent_slug,
            "session_id": self.session_id,
            "priority": self.priority.value,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "metadata": self.metadata,
            "embedding": self.embedding,
            "expires_at": self.expires_at,
        }
        
    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEntry":
        return cls(
            id=data["id"],
            tier=MemoryTier(data["tier"]),
            content=data["content"],
            agent_slug=data["agent_slug"],
            session_id=data["session_id"],
            priority=MemoryPriority(data.get("priority", "medium")),
            created_at=data.get("created_at", ""),
            last_accessed=data.get("last_accessed", ""),
            access_count=data.get("access_count", 0),
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
            expires_at=data.get("expires_at"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# WORKING MEMORY
# ─────────────────────────────────────────────────────────────────────────────

class WorkingMemory:
    """
    Fast-access working memory for current task context.
    Uses a bounded buffer with LRU eviction.
    """
    
    def __init__(self, capacity: int = 50):
        self.capacity = capacity
        self._buffer: deque[MemoryEntry] = deque(maxlen=capacity)
        self._index: dict[str, MemoryEntry] = {}
        
    def add(self, entry: MemoryEntry) -> None:
        """Add entry to working memory."""
        if entry.id in self._index:
            # Move to front (most recent)
            self._buffer = deque(
                [e for e in self._buffer if e.id != entry.id],
                maxlen=self.capacity
            )
        self._buffer.append(entry)
        self._index[entry.id] = entry
        
        # Evict if over capacity
        while len(self._buffer) > self.capacity:
            evicted = self._buffer.popleft()
            del self._index[evicted.id]
            
    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get entry by ID."""
        entry = self._index.get(entry_id)
        if entry:
            entry.access_count += 1
            entry.last_accessed = datetime.now(timezone.utc).isoformat()
        return entry
        
    def search(self, query: str, limit: int = 10) -> list[MemoryEntry]:
        """Simple text search in working memory."""
        results = []
        query_lower = query.lower()
        for entry in self._buffer:
            if query_lower in entry.content.lower():
                results.append(entry)
                if len(results) >= limit:
                    break
        return results
        
    def get_context(self, limit: int = 10) -> list[MemoryEntry]:
        """Get most recent entries as context."""
        return list(self._buffer)[-limit:]
        
    def clear(self) -> None:
        """Clear working memory."""
        self._buffer.clear()
        self._index.clear()


# ─────────────────────────────────────────────────────────────────────────────
# SHORT-TERM MEMORY
# ─────────────────────────────────────────────────────────────────────────────

class ShortTermMemory:
    """
    Short-term memory with time-based expiration.
    Typically retains entries for hours to days.
    """
    
    def __init__(self, default_ttl_hours: int = 24):
        self.default_ttl_hours = default_ttl_hours
        self._entries: dict[str, MemoryEntry] = {}
        
    def add(self, entry: MemoryEntry, ttl_hours: Optional[int] = None) -> None:
        """Add entry with expiration."""
        ttl = ttl_hours or self.default_ttl_hours
        expires = datetime.now(timezone.utc) + timedelta(hours=ttl)
        entry.expires_at = expires.isoformat()
        entry.tier = MemoryTier.SHORT_TERM
        self._entries[entry.id] = entry
        
    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get entry if not expired."""
        entry = self._entries.get(entry_id)
        if entry and not self._is_expired(entry):
            entry.access_count += 1
            entry.last_accessed = datetime.now(timezone.utc).isoformat()
            return entry
        elif entry:
            # Expired - remove it
            del self._entries[entry_id]
        return None
        
    def search(self, query: str, limit: int = 10) -> list[MemoryEntry]:
        """Search non-expired entries."""
        self._cleanup_expired()
        results = []
        query_lower = query.lower()
        for entry in self._entries.values():
            if query_lower in entry.content.lower():
                results.append(entry)
                if len(results) >= limit:
                    break
        return results
        
    def get_by_agent(self, agent_slug: str) -> list[MemoryEntry]:
        """Get all entries for an agent."""
        self._cleanup_expired()
        return [e for e in self._entries.values() if e.agent_slug == agent_slug]
        
    def get_by_session(self, session_id: str) -> list[MemoryEntry]:
        """Get all entries for a session."""
        self._cleanup_expired()
        return [e for e in self._entries.values() if e.session_id == session_id]
        
    def _is_expired(self, entry: MemoryEntry) -> bool:
        """Check if entry is expired."""
        if not entry.expires_at:
            return False
        expires = datetime.fromisoformat(entry.expires_at.replace('Z', '+00:00'))
        return datetime.now(timezone.utc) > expires
        
    def _cleanup_expired(self) -> None:
        """Remove all expired entries."""
        expired_ids = [
            eid for eid, entry in self._entries.items()
            if self._is_expired(entry)
        ]
        for eid in expired_ids:
            del self._entries[eid]


# ─────────────────────────────────────────────────────────────────────────────
# LONG-TERM MEMORY
# ─────────────────────────────────────────────────────────────────────────────

class LongTermMemory:
    """
    Persistent long-term memory backed by file storage.
    Connects to HyperGraphRAG for semantic retrieval.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("bizra_data_vault/memory/long_term")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, MemoryEntry] = {}
        self._load()
        
    def _load(self) -> None:
        """Load memories from storage."""
        memory_file = self.storage_path / "memories.jsonl"
        if memory_file.exists():
            with open(memory_file, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    entry = MemoryEntry.from_dict(data)
                    self._entries[entry.id] = entry
                    
    def add(self, entry: MemoryEntry) -> None:
        """Add entry to long-term memory."""
        entry.tier = MemoryTier.LONG_TERM
        self._entries[entry.id] = entry
        self._persist(entry)
        
    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get entry by ID."""
        entry = self._entries.get(entry_id)
        if entry:
            entry.access_count += 1
            entry.last_accessed = datetime.now(timezone.utc).isoformat()
        return entry
        
    def search(
        self,
        query: str,
        agent_filter: Optional[str] = None,
        limit: int = 50,
    ) -> list[MemoryEntry]:
        """Search long-term memories."""
        results = []
        query_lower = query.lower()
        
        for entry in self._entries.values():
            if agent_filter and entry.agent_slug != agent_filter:
                continue
            if query_lower in entry.content.lower():
                results.append(entry)
                if len(results) >= limit:
                    break
                    
        return results
        
    def consolidate(
        self,
        entries: list[MemoryEntry],
        summary: str,
    ) -> MemoryEntry:
        """Consolidate multiple memories into one."""
        import hashlib
        
        consolidated = MemoryEntry(
            id=hashlib.sha256(summary.encode()).hexdigest()[:16],
            tier=MemoryTier.LONG_TERM,
            content=summary,
            agent_slug=entries[0].agent_slug if entries else "system",
            session_id="consolidated",
            priority=MemoryPriority.HIGH,
            metadata={
                "consolidated_from": [e.id for e in entries],
                "consolidation_time": datetime.now(timezone.utc).isoformat(),
            },
        )
        
        self.add(consolidated)
        return consolidated
        
    def _persist(self, entry: MemoryEntry) -> None:
        """Persist entry to storage."""
        memory_file = self.storage_path / "memories.jsonl"
        with open(memory_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# EPISODIC MEMORY
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Episode:
    """A complete episode (session) of interactions."""
    id: str
    session_id: str
    agent_slug: str
    start_time: str
    end_time: Optional[str] = None
    memories: list[MemoryEntry] = field(default_factory=list)
    summary: Optional[str] = None
    outcome: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "agent_slug": self.agent_slug,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "memories": [m.to_dict() for m in self.memories],
            "summary": self.summary,
            "outcome": self.outcome,
        }


class EpisodicMemory:
    """
    Episodic memory organized by sessions.
    Provides temporal context and episode retrieval.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("bizra_data_vault/memory/episodic")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._episodes: dict[str, Episode] = {}
        self._active_episodes: dict[str, str] = {}  # session_id -> episode_id
        
    def start_episode(
        self,
        session_id: str,
        agent_slug: str,
    ) -> Episode:
        """Start a new episode for a session."""
        import hashlib
        
        episode = Episode(
            id=hashlib.sha256(
                f"{session_id}:{datetime.now(timezone.utc).isoformat()}".encode()
            ).hexdigest()[:16],
            session_id=session_id,
            agent_slug=agent_slug,
            start_time=datetime.now(timezone.utc).isoformat(),
        )
        
        self._episodes[episode.id] = episode
        self._active_episodes[session_id] = episode.id
        return episode
        
    def add_memory(
        self,
        session_id: str,
        entry: MemoryEntry,
    ) -> None:
        """Add memory to current episode."""
        episode_id = self._active_episodes.get(session_id)
        if episode_id and episode_id in self._episodes:
            entry.tier = MemoryTier.EPISODIC
            self._episodes[episode_id].memories.append(entry)
            
    def end_episode(
        self,
        session_id: str,
        summary: Optional[str] = None,
        outcome: Optional[str] = None,
    ) -> Optional[Episode]:
        """End an episode and persist it."""
        episode_id = self._active_episodes.get(session_id)
        if not episode_id:
            return None
            
        episode = self._episodes.get(episode_id)
        if episode:
            episode.end_time = datetime.now(timezone.utc).isoformat()
            episode.summary = summary
            episode.outcome = outcome
            self._persist_episode(episode)
            
        del self._active_episodes[session_id]
        return episode
        
    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Get episode by ID."""
        return self._episodes.get(episode_id)
        
    def get_recent_episodes(
        self,
        agent_slug: Optional[str] = None,
        limit: int = 10,
    ) -> list[Episode]:
        """Get recent episodes, optionally filtered by agent."""
        episodes = list(self._episodes.values())
        
        if agent_slug:
            episodes = [e for e in episodes if e.agent_slug == agent_slug]
            
        # Sort by start time descending
        episodes.sort(key=lambda e: e.start_time, reverse=True)
        
        return episodes[:limit]
        
    def _persist_episode(self, episode: Episode) -> None:
        """Persist episode to storage."""
        episode_file = self.storage_path / f"{episode.id}.json"
        with open(episode_file, "w", encoding="utf-8") as f:
            json.dump(episode.to_dict(), f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# MEMORY MANAGER (ORCHESTRATOR)
# ─────────────────────────────────────────────────────────────────────────────

class MemoryManager:
    """
    Central memory manager orchestrating all memory tiers.
    
    Provides:
    - Unified memory interface for agents
    - Automatic tier promotion/demotion
    - Memory consolidation and pruning
    - Cross-agent memory sharing
    """
    
    def __init__(
        self,
        storage_base: Optional[Path] = None,
        working_capacity: int = 50,
        short_term_ttl: int = 24,
    ):
        self.storage_base = storage_base or Path("bizra_data_vault/memory")
        
        self.working = WorkingMemory(capacity=working_capacity)
        self.short_term = ShortTermMemory(default_ttl_hours=short_term_ttl)
        self.long_term = LongTermMemory(self.storage_base / "long_term")
        self.episodic = EpisodicMemory(self.storage_base / "episodic")
        
        self._agent_interfaces: dict[str, "AgentMemoryInterface"] = {}
        
    def get_interface(self, agent_slug: str) -> "AgentMemoryInterface":
        """Get or create memory interface for an agent."""
        if agent_slug not in self._agent_interfaces:
            self._agent_interfaces[agent_slug] = AgentMemoryInterface(
                agent_slug=agent_slug,
                manager=self,
            )
        return self._agent_interfaces[agent_slug]
        
    def remember(
        self,
        content: str,
        agent_slug: str,
        session_id: str,
        priority: MemoryPriority = MemoryPriority.MEDIUM,
        metadata: Optional[dict] = None,
    ) -> MemoryEntry:
        """Create and store a memory across appropriate tiers."""
        import hashlib
        
        entry = MemoryEntry(
            id=hashlib.sha256(
                f"{content}:{datetime.now(timezone.utc).isoformat()}".encode()
            ).hexdigest()[:16],
            tier=MemoryTier.WORKING,
            content=content,
            agent_slug=agent_slug,
            session_id=session_id,
            priority=priority,
            metadata=metadata or {},
        )
        
        # Add to working memory
        self.working.add(entry)
        
        # Add to short-term if not ephemeral
        if priority != MemoryPriority.EPHEMERAL:
            self.short_term.add(entry)
            
        # Add to episodic
        self.episodic.add_memory(session_id, entry)
        
        # Add to long-term if high priority
        if priority in [MemoryPriority.CRITICAL, MemoryPriority.HIGH]:
            self.long_term.add(entry)
            
        return entry
        
    def recall(
        self,
        query: str,
        agent_slug: Optional[str] = None,
        session_id: Optional[str] = None,
        tiers: Optional[list[MemoryTier]] = None,
        limit: int = 20,
    ) -> list[MemoryEntry]:
        """Recall memories matching query across specified tiers."""
        tiers = tiers or list(MemoryTier)
        all_results: list[MemoryEntry] = []
        
        if MemoryTier.WORKING in tiers:
            all_results.extend(self.working.search(query, limit))
            
        if MemoryTier.SHORT_TERM in tiers:
            results = self.short_term.search(query, limit)
            if agent_slug:
                results = [r for r in results if r.agent_slug == agent_slug]
            all_results.extend(results)
            
        if MemoryTier.LONG_TERM in tiers:
            all_results.extend(
                self.long_term.search(query, agent_slug, limit)
            )
            
        # Deduplicate and limit
        seen = set()
        unique = []
        for entry in all_results:
            if entry.id not in seen:
                seen.add(entry.id)
                unique.append(entry)
                
        return unique[:limit]
        
    def promote_to_long_term(self, entry_id: str) -> bool:
        """Promote a memory to long-term storage."""
        # Check short-term first
        entry = self.short_term.get(entry_id)
        if entry:
            self.long_term.add(entry)
            return True
        return False
        
    def consolidate_session(
        self,
        session_id: str,
        summary: str,
    ) -> Optional[MemoryEntry]:
        """Consolidate all memories from a session."""
        # Get all session memories
        entries = self.short_term.get_by_session(session_id)
        
        if not entries:
            return None
            
        # Create consolidated memory
        consolidated = self.long_term.consolidate(entries, summary)
        
        return consolidated
        
    def start_session(self, session_id: str, agent_slug: str) -> Episode:
        """Start a new session episode."""
        return self.episodic.start_episode(session_id, agent_slug)
        
    def end_session(
        self,
        session_id: str,
        summary: Optional[str] = None,
        outcome: Optional[str] = None,
    ) -> Optional[Episode]:
        """End a session and consolidate memories."""
        episode = self.episodic.end_episode(session_id, summary, outcome)
        
        # Auto-consolidate if summary provided
        if summary:
            self.consolidate_session(session_id, summary)
            
        return episode


# ─────────────────────────────────────────────────────────────────────────────
# AGENT MEMORY INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

class AgentMemoryInterface:
    """
    Agent-specific interface to the memory system.
    
    Provides simplified methods for agents to:
    - Remember new information
    - Recall relevant memories
    - Manage session context
    """
    
    def __init__(
        self,
        agent_slug: str,
        manager: MemoryManager,
    ):
        self.agent_slug = agent_slug
        self.manager = manager
        self._current_session: Optional[str] = None
        
    def start_session(self, session_id: str) -> None:
        """Start a new session for this agent."""
        self._current_session = session_id
        self.manager.start_session(session_id, self.agent_slug)
        
    def end_session(
        self,
        summary: Optional[str] = None,
        outcome: Optional[str] = None,
    ) -> None:
        """End current session."""
        if self._current_session:
            self.manager.end_session(
                self._current_session,
                summary,
                outcome,
            )
            self._current_session = None
            
    def remember(
        self,
        content: str,
        priority: MemoryPriority = MemoryPriority.MEDIUM,
        metadata: Optional[dict] = None,
    ) -> MemoryEntry:
        """Remember something."""
        return self.manager.remember(
            content=content,
            agent_slug=self.agent_slug,
            session_id=self._current_session or "default",
            priority=priority,
            metadata=metadata,
        )
        
    def recall(
        self,
        query: str,
        include_other_agents: bool = False,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Recall relevant memories."""
        return self.manager.recall(
            query=query,
            agent_slug=None if include_other_agents else self.agent_slug,
            session_id=self._current_session,
            limit=limit,
        )
        
    def get_context(self, limit: int = 10) -> list[MemoryEntry]:
        """Get current working context."""
        return self.manager.working.get_context(limit)
