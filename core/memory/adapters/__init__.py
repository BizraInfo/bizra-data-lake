"""Adapters for wrapping existing memory systems into AgentDB."""

from .claude_flow import ClaudeFlowAdapter
from .experience_ledger import ExperienceLedgerAdapter
from .living_memory import LivingMemoryAdapter

__all__ = [
    "ClaudeFlowAdapter",
    "ExperienceLedgerAdapter",
    "LivingMemoryAdapter",
    "PatternMemoryAdapter",
]

# PatternMemory adapter is optional (requires PyO3 build)
try:
    from .pattern_memory import PatternMemoryAdapter
except ImportError:
    pass
