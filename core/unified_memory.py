"""
BIZRA Unified Memory System
Provides seamless access across all 6 memory tiers (M1-M6 + L1-L5 mapped).

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                    UNIFIED MEMORY SYSTEM                        │
├─────────────────────────────────────────────────────────────────┤
│  BIZRA-DATA-LAKE (M1-M6)    ←→    Dual-Agentic (L1-L5)         │
├─────────────────────────────────────────────────────────────────┤
│  M1 TaskMaster Session      ←→    L1 Immediate Memory          │
│  M2 Short-term Episodic     ←→    L2 Working Memory            │
│  M3 Medium-term Semantic    ←→    L3 Episodic Memory           │
│  M4 Long-term Procedural    ←→    L4 Semantic Memory           │
│  M5 Historical Archive      ←→    L5 Procedural Memory         │
│  M6 Sovereign (Cross-Domain) →    Global Context Layer         │
└─────────────────────────────────────────────────────────────────┘
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Optional import - graceful degradation if data_lake_bridge deps not installed
try:
    from core.data_lake_bridge import (
        DataLakeBridge,
        KnowledgeResult,
        MemoryTier,
        get_data_lake_bridge,
    )
    DATA_LAKE_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    # Create stub classes for when data lake bridge isn't available
    DataLakeBridge = None  # type: ignore
    KnowledgeResult = None  # type: ignore
    get_data_lake_bridge = None  # type: ignore
    DATA_LAKE_AVAILABLE = False

    # Define MemoryTier locally if import failed
    class MemoryTier(Enum):
        """Unified memory tiers (fallback definition)."""
        M1_TASKMASTER = "M1"
        M2_SHORT_TERM = "M2"
        M3_SEMANTIC = "M3"
        M4_PROCEDURAL = "M4"
        M5_HISTORICAL = "M5"
        M6_SOVEREIGN = "M6"
        L1_IMMEDIATE = "L1"
        L2_WORKING = "L2"
        L3_EPISODIC = "L3"
        L4_SEMANTIC = "L4"
        L5_PROCEDURAL = "L5"

logger = logging.getLogger(__name__)

if not DATA_LAKE_AVAILABLE:
    logger.warning("Data Lake bridge not available - unified memory will use local tiers only")


class MemoryPriority(Enum):
    """Priority levels for memory entries."""

    CRITICAL = auto()    # Never expire, always accessible
    HIGH = auto()        # Long-term retention
    MEDIUM = auto()      # Standard retention
    LOW = auto()         # Short-term retention
    EPHEMERAL = auto()   # Session-only


@dataclass
class MemoryEntry:
    """A single memory entry in the unified system."""

    content: Any
    tier: MemoryTier
    priority: MemoryPriority
    timestamp: str
    fingerprint: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "local"  # local, data_lake, synapse
    ttl_seconds: Optional[int] = None
    access_count: int = 0

    @classmethod
    def create(
        cls,
        content: Any,
        tier: MemoryTier,
        priority: MemoryPriority = MemoryPriority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None,
        source: str = "local",
        ttl_seconds: Optional[int] = None,
    ) -> "MemoryEntry":
        """Create a new memory entry with auto-generated fingerprint."""
        content_str = json.dumps(content, sort_keys=True, default=str)
        fingerprint = hashlib.sha256(content_str.encode()).hexdigest()[:16]

        return cls(
            content=content,
            tier=tier,
            priority=priority,
            timestamp=datetime.utcnow().isoformat(),
            fingerprint=fingerprint,
            metadata=metadata or {},
            source=source,
            ttl_seconds=ttl_seconds,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "tier": self.tier.value,
            "priority": self.priority.name,
            "timestamp": self.timestamp,
            "fingerprint": self.fingerprint,
            "metadata": self.metadata,
            "source": self.source,
            "ttl_seconds": self.ttl_seconds,
            "access_count": self.access_count,
        }


@dataclass
class MemoryQueryResult:
    """Result from a unified memory query."""

    query: str
    entries: List[MemoryEntry]
    tiers_searched: List[MemoryTier]
    total_count: int
    latency_ms: float
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "entries": [e.to_dict() for e in self.entries],
            "tiers_searched": [t.value for t in self.tiers_searched],
            "total_count": self.total_count,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
        }


class UnifiedMemory:
    """
    Unified Memory System for BIZRA.

    Provides seamless access to:
    - Local memory tiers (L1-L5)
    - Data Lake memory tiers (M1-M6)
    - Sovereign (M6) cross-domain queries

    Usage:
        memory = UnifiedMemory()
        await memory.initialize()

        # Store a memory
        await memory.store("Important fact", tier=MemoryTier.L4_SEMANTIC)

        # Query across all tiers
        result = await memory.query("SAPE architecture")

        # Query sovereign tier (cross-domain)
        result = await memory.query_sovereign("project history")
    """

    def __init__(self):
        self._local_tiers: Dict[MemoryTier, List[MemoryEntry]] = {
            MemoryTier.L1_IMMEDIATE: [],
            MemoryTier.L2_WORKING: [],
            MemoryTier.L3_EPISODIC: [],
            MemoryTier.L4_SEMANTIC: [],
            MemoryTier.L5_PROCEDURAL: [],
        }
        self._data_lake: Optional[DataLakeBridge] = None
        self._initialized = False

        # Tier limits (entries)
        self._tier_limits = {
            MemoryTier.L1_IMMEDIATE: 50,    # Current context
            MemoryTier.L2_WORKING: 100,      # Working memory
            MemoryTier.L3_EPISODIC: 500,     # Episodes
            MemoryTier.L4_SEMANTIC: 1000,    # Facts
            MemoryTier.L5_PROCEDURAL: 500,   # Skills
        }

    async def initialize(self) -> None:
        """Initialize the unified memory system."""
        if self._initialized:
            return

        # Initialize Data Lake bridge (if available)
        if DATA_LAKE_AVAILABLE and get_data_lake_bridge is not None:
            try:
                self._data_lake = get_data_lake_bridge()

                # Check connectivity
                status = await self._data_lake.health_check()
                if status.online:
                    logger.info("Data Lake connected: %s (%d nodes)", status.url, status.nodes)
                else:
                    logger.warning("Data Lake offline: %s", status.error)
            except Exception as e:
                logger.warning("Failed to initialize Data Lake bridge: %s", e)
                self._data_lake = None
        else:
            logger.info("Data Lake bridge not available - using local memory only")
            self._data_lake = None

        self._initialized = True
        logger.info("Unified Memory System initialized")

    async def close(self) -> None:
        """Close connections."""
        if self._data_lake:
            await self._data_lake.close()

    def _enforce_tier_limit(self, tier: MemoryTier) -> None:
        """Enforce tier entry limits (FIFO eviction)."""
        if tier not in self._local_tiers:
            return

        limit = self._tier_limits.get(tier, 100)
        entries = self._local_tiers[tier]

        if len(entries) > limit:
            # Evict oldest entries, preserving CRITICAL priority
            non_critical = [e for e in entries if e.priority != MemoryPriority.CRITICAL]
            critical = [e for e in entries if e.priority == MemoryPriority.CRITICAL]

            # Sort by timestamp, keep newest
            non_critical.sort(key=lambda e: e.timestamp, reverse=True)
            keep_count = limit - len(critical)
            self._local_tiers[tier] = critical + non_critical[:keep_count]

    async def store(
        self,
        content: Any,
        tier: MemoryTier = MemoryTier.L3_EPISODIC,
        priority: MemoryPriority = MemoryPriority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None,
        sync_to_lake: bool = True,
    ) -> MemoryEntry:
        """
        Store a memory entry.

        Args:
            content: The content to store
            tier: Which tier to store in (L1-L5)
            priority: Retention priority
            metadata: Additional metadata
            sync_to_lake: Whether to sync to Data Lake (bidirectional)

        Returns:
            The created MemoryEntry
        """
        entry = MemoryEntry.create(
            content=content,
            tier=tier,
            priority=priority,
            metadata=metadata,
            source="local",
        )

        # Store locally
        if tier in self._local_tiers:
            self._local_tiers[tier].append(entry)
            self._enforce_tier_limit(tier)
            logger.debug("Stored memory in %s: %s", tier.value, entry.fingerprint)

        # Sync to Data Lake (bidirectional)
        if sync_to_lake and self._data_lake:
            # TODO: Implement write-back to Data Lake
            # This would use a separate MCP method like "knowledge_store"
            pass

        return entry

    async def query(
        self,
        query: str,
        tiers: Optional[List[MemoryTier]] = None,
        limit: int = 10,
        include_data_lake: bool = True,
    ) -> MemoryQueryResult:
        """
        Query across memory tiers.

        Args:
            query: Search query
            tiers: Specific tiers to search (default: all)
            limit: Maximum results
            include_data_lake: Whether to query Data Lake

        Returns:
            MemoryQueryResult with matching entries
        """
        start_time = datetime.utcnow()
        entries: List[MemoryEntry] = []
        tiers_searched: List[MemoryTier] = []

        # Default to all local tiers
        if tiers is None:
            tiers = list(self._local_tiers.keys())

        # Search local tiers
        query_lower = query.lower()
        for tier in tiers:
            if tier in self._local_tiers:
                tiers_searched.append(tier)
                for entry in self._local_tiers[tier]:
                    content_str = json.dumps(entry.content, default=str).lower()
                    if query_lower in content_str:
                        entry.access_count += 1
                        entries.append(entry)

        # Query Data Lake
        if include_data_lake and self._data_lake:
            try:
                lake_result = await self._data_lake.query_sovereign(query, limit=limit)
                tiers_searched.append(MemoryTier.M6_SOVEREIGN)

                # Convert lake results to MemoryEntry
                for item in lake_result.results:
                    entry = MemoryEntry.create(
                        content=item,
                        tier=MemoryTier.M6_SOVEREIGN,
                        priority=MemoryPriority.MEDIUM,
                        source="data_lake",
                    )
                    entries.append(entry)
            except Exception as e:
                logger.warning("Data Lake query failed: %s", e)

        # Sort by priority and timestamp
        entries.sort(
            key=lambda e: (e.priority.value, e.timestamp),
            reverse=True,
        )

        # Apply limit
        entries = entries[:limit]

        latency = (datetime.utcnow() - start_time).total_seconds() * 1000

        return MemoryQueryResult(
            query=query,
            entries=entries,
            tiers_searched=tiers_searched,
            total_count=len(entries),
            latency_ms=latency,
            timestamp=datetime.utcnow().isoformat(),
        )

    async def query_sovereign(
        self,
        query: str,
        limit: int = 10,
    ) -> MemoryQueryResult:
        """
        Query the M6 Sovereign tier (cross-domain).

        This provides the "God view" across the entire 1.37TB Data Lake.
        """
        if not self._data_lake:
            return MemoryQueryResult(
                query=query,
                entries=[],
                tiers_searched=[],
                total_count=0,
                latency_ms=0,
                timestamp=datetime.utcnow().isoformat(),
            )

        start_time = datetime.utcnow()
        entries: List[MemoryEntry] = []

        lake_result = await self._data_lake.query_sovereign(query, limit=limit)

        for item in lake_result.results:
            entry = MemoryEntry.create(
                content=item,
                tier=MemoryTier.M6_SOVEREIGN,
                priority=MemoryPriority.MEDIUM,
                source="data_lake",
            )
            entries.append(entry)

        latency = (datetime.utcnow() - start_time).total_seconds() * 1000

        return MemoryQueryResult(
            query=query,
            entries=entries,
            tiers_searched=[MemoryTier.M6_SOVEREIGN],
            total_count=len(entries),
            latency_ms=latency,
            timestamp=datetime.utcnow().isoformat(),
        )

    async def query_local(
        self,
        query: str,
        tier: Optional[MemoryTier] = None,
        limit: int = 10,
    ) -> MemoryQueryResult:
        """
        Query only local memory tiers (L1-L5).

        Useful when Data Lake is offline or for fast local queries.
        """
        return await self.query(
            query=query,
            tiers=[tier] if tier else None,
            limit=limit,
            include_data_lake=False,
        )

    def get_tier_stats(self) -> Dict[str, Any]:
        """Get statistics for all memory tiers."""
        stats = {}
        for tier, entries in self._local_tiers.items():
            stats[tier.value] = {
                "count": len(entries),
                "limit": self._tier_limits.get(tier, 100),
                "priorities": {
                    p.name: len([e for e in entries if e.priority == p])
                    for p in MemoryPriority
                },
            }
        return stats

    def clear_tier(self, tier: MemoryTier) -> int:
        """Clear all entries from a tier. Returns count of cleared entries."""
        if tier not in self._local_tiers:
            return 0
        count = len(self._local_tiers[tier])
        self._local_tiers[tier] = []
        logger.info("Cleared %d entries from %s", count, tier.value)
        return count


# Singleton instance
_unified_memory: Optional[UnifiedMemory] = None


async def get_unified_memory() -> UnifiedMemory:
    """Get the singleton UnifiedMemory instance."""
    global _unified_memory
    if _unified_memory is None:
        _unified_memory = UnifiedMemory()
        await _unified_memory.initialize()
    return _unified_memory


# Convenience functions
async def remember(
    content: Any,
    tier: MemoryTier = MemoryTier.L3_EPISODIC,
    priority: MemoryPriority = MemoryPriority.MEDIUM,
) -> MemoryEntry:
    """Store a memory entry."""
    memory = await get_unified_memory()
    return await memory.store(content, tier=tier, priority=priority)


async def recall(query: str, limit: int = 10) -> MemoryQueryResult:
    """Query unified memory."""
    memory = await get_unified_memory()
    return await memory.query(query, limit=limit)


async def recall_sovereign(query: str, limit: int = 10) -> MemoryQueryResult:
    """Query M6 Sovereign tier (cross-domain)."""
    memory = await get_unified_memory()
    return await memory.query_sovereign(query, limit=limit)


if __name__ == "__main__":
    # Test the unified memory system
    async def test():
        memory = UnifiedMemory()
        await memory.initialize()

        # Store some test memories
        await memory.store("BIZRA uses PAT-SAT dual-agentic architecture", tier=MemoryTier.L4_SEMANTIC)
        await memory.store("SAPE has 9 probes for validation", tier=MemoryTier.L4_SEMANTIC)
        await memory.store("Ihsan threshold is 0.99", tier=MemoryTier.L4_SEMANTIC)

        # Query local
        local_result = await memory.query_local("BIZRA")
        print(f"Local query: {local_result.total_count} results")
        for entry in local_result.entries:
            print(f"  - {entry.content}")

        # Query sovereign (Data Lake)
        print("\nQuerying M6 Sovereign tier...")
        sovereign_result = await memory.query_sovereign("BIZRA architecture")
        print(f"Sovereign query: {sovereign_result.total_count} results")
        for entry in sovereign_result.entries[:3]:
            print(f"  - {entry.content}")

        # Stats
        print("\nTier Statistics:")
        stats = memory.get_tier_stats()
        for tier, info in stats.items():
            print(f"  {tier}: {info['count']}/{info['limit']} entries")

        await memory.close()

    asyncio.run(test())
