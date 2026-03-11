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

from .adapters.claude_flow import ClaudeFlowAdapter
from .agent_db import AgentDB
from .config import HNSWConfig, MemoryConfig
from .convergence import (
    ConvergencePolicy,
    format_convergence_summary,
    inspect_claude_flow_sources,
    run_convergence,
)
from .coordinator_bridge import AgentDBBridge
from .health import AgentDBHealthChecker, AgentDBMetrics, HealthReport, HealthStatus
from .hnsw_index import HNSWIndex
from .hybrid_query import HybridQueryEngine
from .memory_patterns import (
    ConsolidationResult,
    ContextSynthesizer,
    FactStore,
    HierarchicalMemory,
    MemoryConsolidator,
    MemoryTier,
    SessionMemory,
    SynthesizedContext,
)
from .orchestrator import MigrationOrchestrator, OrchestratorResult
from .types import MemoryKind, MemoryRecord, QueryOptions, RecordState, SearchResult
from .unified_store import UnifiedStore

__all__ = [
    "AgentDB",
    "AgentDBBridge",
    "AgentDBHealthChecker",
    "AgentDBMetrics",
    "ClaudeFlowAdapter",
    "ConsolidationResult",
    "ContextSynthesizer",
    "ConvergencePolicy",
    "format_convergence_summary",
    "FactStore",
    "HNSWConfig",
    "HNSWIndex",
    "HealthReport",
    "HealthStatus",
    "HierarchicalMemory",
    "HybridQueryEngine",
    "inspect_claude_flow_sources",
    "MemoryConfig",
    "MemoryConsolidator",
    "MemoryKind",
    "MemoryRecord",
    "MemoryTier",
    "MigrationOrchestrator",
    "OrchestratorResult",
    "QueryOptions",
    "RecordState",
    "run_convergence",
    "SearchResult",
    "SessionMemory",
    "SynthesizedContext",
    "UnifiedStore",
]
