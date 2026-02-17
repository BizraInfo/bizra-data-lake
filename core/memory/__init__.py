"""
core.memory — Unified Memory System (V3 AgentDB)

Single entry point for all memory operations in BIZRA.

Usage:
    from core.memory import AgentDB, MemoryConfig
    db = AgentDB(MemoryConfig())
    db.initialize()
    db.store("knowledge", importance=0.9)
    results = db.search("query")

Architecture:
    AgentDB facade -> UnifiedStore (SQLite v2 + FTS5)
                   -> HNSWIndex (sub-linear vector search)
                   -> HybridQueryEngine (score fusion)
                   -> Adapters (LivingMemory, SEL, PatternMemory)
"""

from .agent_db import AgentDB
from .config import HNSWConfig, MemoryConfig
from .hnsw_index import HNSWIndex
from .hybrid_query import HybridQueryEngine
from .types import MemoryKind, MemoryRecord, QueryOptions, RecordState, SearchResult
from .unified_store import UnifiedStore

__all__ = [
    "AgentDB",
    "HNSWConfig",
    "HNSWIndex",
    "HybridQueryEngine",
    "MemoryConfig",
    "MemoryKind",
    "MemoryRecord",
    "QueryOptions",
    "RecordState",
    "SearchResult",
    "UnifiedStore",
]
