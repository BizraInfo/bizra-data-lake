"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BIZRA LIVING KNOWLEDGE MEMORY — Self-Evolving Intelligence Core           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   Standing on the Shoulders of Giants:                                       ║
║   • Maturana & Varela (Autopoiesis - Living Systems)                         ║
║   • Shannon (Information Theory)                                             ║
║   • Hopfield (Memory Networks)                                               ║
║   • Anthropic (Constitutional AI / Ihsān)                                    ║
║   • LangChain (Memory Patterns)                                              ║
║                                                                              ║
║   Living Memory Principles:                                                  ║
║   • Self-Organizing: Knowledge structures emerge from data                   ║
║   • Self-Healing: Corrupted memories are detected and repaired               ║
║   • Self-Optimizing: Frequently accessed knowledge is prioritized            ║
║   • Self-Sustainable: Energy-efficient memory management                     ║
║   • Proactive: Anticipates information needs                                 ║
║                                                                              ║
║   Created: 2026-02-02 | BIZRA Living Knowledge Integration                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# Constants
MEMORY_VERSION = "1.0.0"
DEFAULT_EMBEDDING_DIM = 384
MAX_MEMORY_ENTRIES = 100_000
CONSOLIDATION_INTERVAL_SECONDS = 300  # 5 minutes
DECAY_HALF_LIFE_HOURS = 24 * 7  # 1 week
IHSAN_MEMORY_THRESHOLD = 0.90  # Lower than operational threshold

# Memory types
MEMORY_TYPES = [
    "episodic",  # Event-based memories
    "semantic",  # Fact-based knowledge
    "procedural",  # How-to knowledge
    "working",  # Active context
    "prospective",  # Future-oriented (goals, plans)
]


# Lazy imports
def __getattr__(name: str):
    if name == "LivingMemoryCore":
        from .core import LivingMemoryCore

        return LivingMemoryCore
    elif name == "MemoryEntry":
        from .core import MemoryEntry

        return MemoryEntry
    elif name == "MemoryConsolidator":
        from .consolidation import MemoryConsolidator

        return MemoryConsolidator
    elif name == "ProactiveRetriever":
        from .proactive import ProactiveRetriever

        return ProactiveRetriever
    elif name == "KnowledgeGraph":
        from .graph import KnowledgeGraph

        return KnowledgeGraph
    elif name == "MemoryHealer":
        from .healing import MemoryHealer

        return MemoryHealer
    raise AttributeError(f"module 'core.living_memory' has no attribute '{name}'")


__all__ = [
    "LivingMemoryCore",
    "MemoryEntry",
    "MemoryConsolidator",
    "ProactiveRetriever",
    "KnowledgeGraph",
    "MemoryHealer",
    "MEMORY_VERSION",
    "DEFAULT_EMBEDDING_DIM",
    "MAX_MEMORY_ENTRIES",
    "MEMORY_TYPES",
]
