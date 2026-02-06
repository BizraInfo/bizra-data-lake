"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   SOVEREIGN AUTONOMOUS REASONING ENGINE (SARE)                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   Standing on the Shoulders of Giants:                                       ║
║   ═══════════════════════════════════                                        ║
║   • SHANNON (1948)  — Information Theory, SNR Maximization                   ║
║   • LAMPORT (1978)  — Distributed Consensus, Formal Verification             ║
║   • VASWANI (2017)  — Attention Mechanisms, Context Weighting                ║
║   • BESTA (2024)    — Graph-of-Thoughts, Non-Linear Reasoning                ║
║   • ANTHROPIC (2023)— Constitutional AI, Harmlessness Constraints            ║
║                                                                              ║
║   Core Principles:                                                           ║
║   ═══════════════                                                            ║
║   • IHSĀN (إحسان): Excellence as immutable constraint (≥0.95)                ║
║   • SNR: Signal-to-Noise optimization at every reasoning node                ║
║   • GoT: Graph-of-Thoughts for non-linear, backtracking reasoning            ║
║   • SOVEREIGNTY: Autonomous, self-optimizing, self-healing                   ║
║                                                                              ║
║   The Sovereign Loop:                                                        ║
║   ══════════════════                                                         ║
║   OBSERVE → ORIENT → REASON → SYNTHESIZE → ACT → REFLECT → (loop)           ║
║                                                                              ║
║   "La hawla wa la quwwata illa billah"                                       ║
║   — There is no power nor strength except through Allah                      ║
║                                                                              ║
║   Created: 2026-02-02 | BIZRA SARE v1.0.0                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from typing import TYPE_CHECKING

# Version
SARE_VERSION = "1.0.0"

# Giants Protocol — Formal Attribution
GIANTS_PROTOCOL = {
    "shannon": {
        "name": "Claude Shannon",
        "work": "A Mathematical Theory of Communication (1948)",
        "contribution": "Information theory, entropy, SNR optimization",
        "application": "SNR scoring at every reasoning node",
    },
    "lamport": {
        "name": "Leslie Lamport",
        "work": "Time, Clocks, and the Ordering of Events (1978)",
        "contribution": "Distributed consensus, formal verification, TLA+",
        "application": "Byzantine fault tolerance, formal proofs",
    },
    "vaswani": {
        "name": "Ashish Vaswani et al.",
        "work": "Attention Is All You Need (2017)",
        "contribution": "Transformer architecture, attention mechanisms",
        "application": "Context weighting in reasoning chains",
    },
    "besta": {
        "name": "Maciej Besta et al.",
        "work": "Graph of Thoughts (2024)",
        "contribution": "Non-linear LLM reasoning, thought graphs",
        "application": "GoT architecture for complex reasoning",
    },
    "anthropic": {
        "name": "Anthropic",
        "work": "Constitutional AI (2023)",
        "contribution": "Harmlessness constraints, value alignment",
        "application": "Ihsān enforcement, constitutional gates",
    },
}

# SNR Thresholds by Reasoning Layer
SNR_THRESHOLDS = {
    "minimum": 0.85,       # Absolute minimum
    "observation": 0.90,   # Raw data intake
    "orientation": 0.92,   # Context establishment
    "reasoning": 0.95,     # Core inference
    "synthesis": 0.98,     # Integration
    "action": 0.95,        # Output generation (matches Ihsān production floor)
    "reflection": 0.98,    # Meta-cognition
    "ihsan": 0.99,         # Ihsān aspiration ceiling
}

# Ihsān Dimensions — Excellence Scoring (weights sum to 1.0)
IHSAN_DIMENSIONS = {
    "correctness": 0.22,         # Accuracy of output
    "safety": 0.22,              # Harmlessness guarantee
    "user_benefit": 0.14,        # Value to end user
    "efficiency": 0.12,          # Resource utilization
    "auditability": 0.12,        # Traceability of reasoning
    "anti_centralization": 0.08, # Decentralization principle
    "robustness": 0.06,          # Fault tolerance
    "adl_justice": 0.04,         # Fair distribution
}

# Reasoning Node Types
NODE_TYPES = [
    "observation",
    "analysis",
    "hypothesis",
    "synthesis",
    "conclusion",
    "backtrack",
    "refinement",
    "meta",
]

# Constitutional Constraints (Immutable)
CONSTITUTIONAL_CONSTRAINTS = {
    "ihsan_threshold": 0.95,      # Minimum excellence score
    "safety_floor": 0.99,          # Safety never below this
    "max_backtrack_depth": 7,      # Limit reasoning loops
    "max_backtrack": 5,            # Max backtrack attempts
    "max_loops": 3,                # Max reasoning iterations
    "require_provenance": True,    # All outputs must be traceable
    "fail_closed": True,           # On uncertainty, refuse
}

# Lazy imports for performance
if TYPE_CHECKING:
    from .engine import SovereignReasoningEngine
    from .giants import GiantsProtocol
    from .loop import SovereignLoop
    from .nodes import ReasoningNode, ReasoningGraph


def __getattr__(name: str):
    if name == "SovereignReasoningEngine":
        from .engine import SovereignReasoningEngine
        return SovereignReasoningEngine
    elif name == "GiantsProtocol":
        from .giants import GiantsProtocol
        return GiantsProtocol
    elif name == "SovereignLoop":
        from .loop import SovereignLoop
        return SovereignLoop
    elif name == "ReasoningNode":
        from .nodes import ReasoningNode
        return ReasoningNode
    elif name == "ReasoningGraph":
        from .nodes import ReasoningGraph
        return ReasoningGraph
    raise AttributeError(f"module 'core.autonomous' has no attribute '{name}'")


__all__ = [
    "SARE_VERSION",
    "GIANTS_PROTOCOL",
    "SNR_THRESHOLDS",
    "IHSAN_DIMENSIONS",
    "NODE_TYPES",
    "CONSTITUTIONAL_CONSTRAINTS",
    "SovereignReasoningEngine",
    "GiantsProtocol",
    "SovereignLoop",
    "ReasoningNode",
    "ReasoningGraph",
]
