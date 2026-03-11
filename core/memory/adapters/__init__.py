"""Adapters for wrapping existing memory systems into AgentDB."""

from .claude_flow import ClaudeFlowAdapter
from .evidence_chain import EvidenceAwareMemory
from .experience_ledger import ExperienceLedgerAdapter
from .living_memory import LivingMemoryAdapter, LivingMemoryBridge
from .reasoning_bank import ReasoningBankAdapter

__all__ = [
    "ClaudeFlowAdapter",
    "EvidenceAwareMemory",
    "ExperienceLedgerAdapter",
    "LivingMemoryAdapter",
    "LivingMemoryBridge",
    "PatternMemoryAdapter",
    "ReasoningBankAdapter",
]

# PatternMemory adapter is optional (requires PyO3 build)
try:
    from .pattern_memory import PatternMemoryAdapter
except ImportError:
    pass
