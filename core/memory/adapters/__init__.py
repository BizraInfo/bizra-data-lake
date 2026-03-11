"""Adapters for wrapping existing memory systems into AgentDB."""

from .claude_flow import ClaudeFlowAdapter
from .experience_ledger import ExperienceLedgerAdapter
from .living_memory import LivingMemoryAdapter
from .reasoning_bank import ReasoningBankAdapter

__all__ = [
    "ClaudeFlowAdapter",
    "ExperienceLedgerAdapter",
    "LivingMemoryAdapter",
    "PatternMemoryAdapter",
    "ReasoningBankAdapter",
]

# PatternMemory adapter is optional (requires PyO3 build)
try:
    from .pattern_memory import PatternMemoryAdapter
except ImportError:
    pass
