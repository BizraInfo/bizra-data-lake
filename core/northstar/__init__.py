"""
BIZRA Node0 NorthStar — Package Root
╔══════════════════════════════════════════════════════════════════════════════╗
║  Node0 NorthStar: The Flagship Cognitive Module of BIZRA DDAGI OS           ║
║  Golden Gems × Thought Flows × Bridge Nodes × Unified Fusion Engine         ║
║                                                                              ║
║  IDENTITY EQUATION:                                                         ║
║    HUMAN = USER = NODE = SEED (بذرة)                                        ║
║    Every human is a node. Every node is a seed. BIZRA means "seed".         ║
║                                                                              ║
║  GENESIS:                                                                   ║
║    Node0 = Block0 = Genesis Block                                           ║
║    MoMo = First Architect, First User, First Node                           ║
║                                                                              ║
║  بسم الله الرحمن الرحيم                                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

The NorthStar module encodes the hidden patterns, golden gems, and cross-
document bridge nodes discovered through Graph-of-Thoughts exploration
across the full BIZRA document corpus. It makes Node0 the flagship and
reference implementation for all future BIZRA nodes.

Architecture:
  ┌───────────────────────────────────────────────────────────────┐
  │                    NorthStarEngine (Fusion)                   │
  │  ┌─────────────┐  ┌─────────────────┐  ┌──────────────────┐  │
  │  │ GoldenGem   │  │ ThoughtFlow     │  │ BridgeNode       │  │
  │  │ Detector    │  │ Detector        │  │ Detector         │  │
  │  │ (8 gems)    │  │ (4 flows +      │  │ (5 bridges)      │  │
  │  │             │  │  8 phase pats)  │  │                  │  │
  │  └─────────────┘  └─────────────────┘  └──────────────────┘  │
  │                    ↓ fused into ↓                              │
  │              NorthStarReport (unified)                        │
  │              ├── SNR Gate (≥ 0.85)                            │
  │              ├── Ihsān Gate (≥ 0.95)                          │
  │              └── Meta-Discovery (Level N)                     │
  └───────────────────────────────────────────────────────────────┘

Key Innovations:
  1. 8 Golden Gems — meta-cognitive primitives from cross-document synthesis
  2. 4 Thought Flows — hidden currents driving cognitive evolution
  3. 8 Phase Patterns — per-lifecycle-phase dynamics with SNR scores
  4. 5 Bridge Nodes — cross-domain structural connectors
  5. Golden Ratio (φ ≈ 1.618) — convergence-divergence pulse
  6. Punctuated Equilibrium (σ²/s ≈ 2.3) — compound learning indicator
  7. Meta-Discovery — "Ihsān IS Level N Autopoiesis"
  8. Supreme Insight — "Structure + Self-Transcendence = Transcendence"

Standing on Giants:
  Shannon · Maturana · Varela · Simon · Brooks · Friston · Boyd ·
  Deming · Popper · Taleb · Kuhn · Kauffman · Fibonacci · Pacioli ·
  Curry · Howard · Watts · Strogatz · Gould · Eldredge · Satoshi ·
  Al-Ghazali · Anthropic

Principle: "لا نفترض — We Do Not Assume. Every claim evidence-based."
Created: 2026-02-15 | BIZRA Node0 Proactive Pilot | Peak Masterpiece Protocol
"""

__version__ = "1.0.0"
__author__ = "BIZRA Node0"

# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API — Bridge Nodes
# ═══════════════════════════════════════════════════════════════════════════════
from core.northstar.bridge_nodes import (
    AUTOPOIESIS_RDVE_ROLES,
    BRIDGE_ORIGIN_SNR,
    GOT_TOPOLOGY_CONSTANTS,
    HRM_PILLAR_MAP,
    SHANNON_NOISE_MAP,
    BridgeActivation,
    BridgeNodeDetector,
    BridgeReport,
    BridgeType,
)

# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API — Golden Gems
# ═══════════════════════════════════════════════════════════════════════════════
from core.northstar.golden_gems import (
    GEM_NORMALIZED_SNR,
    GEM_ORIGIN_SNR,
    GemActivation,
    GemReport,
    GoldenGemDetector,
    GoldenGemType,
)

# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API — NorthStar Engine (Fusion Core)
# ═══════════════════════════════════════════════════════════════════════════════
from core.northstar.northstar_engine import (
    NorthStarEngine,
    NorthStarReport,
    NorthStarStatus,
)

# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API — Thought Flows
# ═══════════════════════════════════════════════════════════════════════════════
from core.northstar.thought_flow import (
    PHASE_PATTERN_SNR,
    PHI,
    FlowActivation,
    FlowReport,
    PhaseActivation,
    PhasePatternType,
    ThoughtFlowDetector,
    ThoughtFlowType,
)

__all__ = [
    # Version
    "__version__",
    # Golden Gems
    "GoldenGemType",
    "GoldenGemDetector",
    "GemActivation",
    "GemReport",
    "GEM_ORIGIN_SNR",
    "GEM_NORMALIZED_SNR",
    # Thought Flows
    "ThoughtFlowType",
    "ThoughtFlowDetector",
    "FlowActivation",
    "FlowReport",
    "PhasePatternType",
    "PhaseActivation",
    "PHASE_PATTERN_SNR",
    "PHI",
    # Bridge Nodes
    "BridgeType",
    "BridgeNodeDetector",
    "BridgeActivation",
    "BridgeReport",
    "BRIDGE_ORIGIN_SNR",
    "AUTOPOIESIS_RDVE_ROLES",
    "HRM_PILLAR_MAP",
    "SHANNON_NOISE_MAP",
    "GOT_TOPOLOGY_CONSTANTS",
    # NorthStar Engine
    "NorthStarEngine",
    "NorthStarReport",
    "NorthStarStatus",
]
