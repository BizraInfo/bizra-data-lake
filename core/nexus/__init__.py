"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                                  ║
║   ███████╗ ██████╗ ██╗   ██╗███████╗██████╗ ███████╗██╗ ██████╗ ███╗   ██╗                       ║
║   ██╔════╝██╔═══██╗██║   ██║██╔════╝██╔══██╗██╔════╝██║██╔════╝ ████╗  ██║                       ║
║   ███████╗██║   ██║██║   ██║█████╗  ██████╔╝█████╗  ██║██║  ███╗██╔██╗ ██║                       ║
║   ╚════██║██║   ██║╚██╗ ██╔╝██╔══╝  ██╔══██╗██╔══╝  ██║██║   ██║██║╚██╗██║                       ║
║   ███████║╚██████╔╝ ╚████╔╝ ███████╗██║  ██║███████╗██║╚██████╔╝██║ ╚████║                       ║
║   ╚══════╝ ╚═════╝   ╚═══╝  ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝                       ║
║                                                                                                  ║
║   ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗                                                    ║
║   ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝                                                    ║
║   ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗                                                    ║
║   ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║                                                    ║
║   ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║                                                    ║
║   ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝                                                    ║
║                                                                                                  ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════╣
║   The Ultimate Unified Orchestration Layer — Peak Masterpiece Implementation                    ║
║                                                                                                  ║
║   Integrates: Skills → A2A → Hooks → MCP → Inference → Graph-of-Thoughts → SNR                  ║
║                                                                                                  ║
║   إحسان — Excellence in all things                                                              ║
║   لا نفترض — We do not assume. We verify.                                                       ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════╝

Standing on the Shoulders of Giants:
─────────────────────────────────────
│ GIANT                  │ YEAR │ CONTRIBUTION                                    │
├────────────────────────┼──────┼─────────────────────────────────────────────────┤
│ Kurt Gödel             │ 1931 │ Incompleteness theorems (bounded rationality)   │
│ Alan Turing            │ 1936 │ Universal computation (Turing completeness)     │
│ Claude Shannon         │ 1948 │ Information theory (SNR foundation)             │
│ John Boyd              │ 1995 │ OODA Loop (observe-orient-decide-act)           │
│ Leslie Lamport         │ 1982 │ Byzantine consensus (distributed trust)         │
│ Edsger Dijkstra        │ 1959 │ Shortest path (optimal routing)                 │
│ Donald Knuth           │ 1968 │ Algorithm analysis (complexity bounds)          │
│ Abu Hamid Al-Ghazali   │ 1095 │ Ihsān excellence (ethical framework)            │
│ Ashish Vaswani         │ 2017 │ Transformer architecture (attention)            │
│ Maciej Besta           │ 2024 │ Graph-of-Thoughts (structured reasoning)        │
│ Anthropic              │ 2023 │ Constitutional AI (safety alignment)            │
│ Noam Shazeer           │ 2017 │ Mixture-of-Experts (adaptive routing)           │
│ W. Edwards Deming      │ 1950 │ PDCA Cycle (continuous improvement)             │
└────────────────────────┴──────┴─────────────────────────────────────────────────┘

Created: 2026-02-08 | BIZRA Sovereign Nexus v1.0.0
"""

from .sovereign_nexus import (
    # Core classes
    SovereignNexus,
    NexusConfig,
    NexusState,
    # Thought graph
    ThoughtNode,
    ThoughtEdge,
    ThoughtGraph,
    ThoughtType,
    # Execution
    NexusTask,
    NexusResult,
    NexusPhase,
    # Agents
    AgentRole,
    # SNR
    SNRGate,
    SNRScore,
    # Factory
    create_nexus,
)

__all__ = [
    # Core
    "SovereignNexus",
    "NexusConfig",
    "NexusState",
    # Thoughts
    "ThoughtNode",
    "ThoughtEdge",
    "ThoughtGraph",
    "ThoughtType",
    # Execution
    "NexusTask",
    "NexusResult",
    "NexusPhase",
    # Agents
    "AgentRole",
    # SNR
    "SNRGate",
    "SNRScore",
    # Factory
    "create_nexus",
]

__version__ = "1.0.0"

# Giants Protocol — Attribution is mandatory
__giants__ = [
    ("Kurt Gödel", 1931, "Incompleteness theorems"),
    ("Alan Turing", 1936, "Universal computation"),
    ("Claude Shannon", 1948, "Information theory"),
    ("John Boyd", 1995, "OODA Loop"),
    ("Leslie Lamport", 1982, "Byzantine consensus"),
    ("Edsger Dijkstra", 1959, "Shortest path algorithm"),
    ("Donald Knuth", 1968, "Algorithm analysis"),
    ("Abu Hamid Al-Ghazali", 1095, "Ihsān excellence framework"),
    ("Ashish Vaswani", 2017, "Transformer architecture"),
    ("Maciej Besta", 2024, "Graph-of-Thoughts"),
    ("Anthropic", 2023, "Constitutional AI"),
    ("Noam Shazeer", 2017, "Mixture-of-Experts"),
    ("W. Edwards Deming", 1950, "PDCA Cycle"),
]
