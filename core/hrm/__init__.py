"""
BIZRA Hierarchical Reasoning Model (HRM) — Package Root

╔══════════════════════════════════════════════════════════════════════════════╗
║  Autopoietic Cognitive Architecture × Hierarchical Reasoning Model          ║
║  Graph-of-Thoughts Exploration · Multi-Level Learning · Meta-Autopoiesis    ║
║                                                                              ║
║  بسم الله الرحمن الرحيم                                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Implements the fusion of Hierarchical Reasoning Model (HRM) with Autopoietic
Cognitive Architecture, creating a cognitive system that learns at every level
of abstraction while maintaining coherent cross-level dynamics.

Architecture:
  Level N: Meta-Cognitive ← Autopoietic Loop N → (evolves the hierarchy)
    ↕ Top-Down / Bottom-Up Flow
  Level 3: Strategic     ← Autopoietic Loop 3 → (goals, planning)
    ↕ Cross-Level Integration
  Level 2: Tactical      ← Autopoietic Loop 2 → (actions, resources)
    ↕ Evidence Aggregation
  Level 1: Operational   ← Autopoietic Loop 1 → (patterns, features)
    ↕ Signal Detection
  Level 0: Perceptual    ← Autopoietic Loop 0 → (raw signals)

Key Innovations:
  - Nested autopoietic cycles at each abstraction level
  - 5 cross-level integration mechanisms (not just goals-down/evidence-up)
  - Meta-autopoietic Level N that evolves the hierarchy itself
  - Learning cascade: improvements compound across levels
  - Learning resonance: cross-level acceleration events
  - SNR gradient: level-specific quality thresholds

Standing on Giants:
  - Maturana & Varela (1980) — Autopoiesis (self-producing systems)
  - Simon (1962) — "The Architecture of Complexity" (near-decomposability)
  - Brooks (1986) — Subsumption Architecture (layered competence)
  - Friston (2010) — Free Energy Principle (hierarchical prediction)
  - Shannon (1948) — Information Theory (SNR optimization)
  - Boyd (1976) — OODA Loop (observe-orient-decide-act)
  - Al-Ghazali (1095) — Muraqabah/Ihsān (vigilance/excellence)

Principle: "لا نفترض — We Do Not Assume. Every claim evidence-based."
Created: 2026-02-15 | BIZRA Node0 Proactive Pilot | Peak Masterpiece Protocol
"""

__version__ = "1.0.0"
__author__ = "BIZRA Node0"

# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API — Abstraction Levels
# ═══════════════════════════════════════════════════════════════════════════════
from core.hrm.abstraction_levels import (
    AbstractionLevel,
    BridgeNodeType,
    LevelBoundary,
    LevelConfig,
    TemporalScale,
    HRM_SNR_GRADIENT,
    HRM_TEMPORAL_SCALE,
    default_level_configs,
    default_boundaries,
)

# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API — Cross-Level Bridge
# ═══════════════════════════════════════════════════════════════════════════════
from core.hrm.cross_level_bridge import (
    CascadeResult,
    CrossLevelBridge,
    CrossLevelMessage,
    MessageType,
    PropagationDirection,
    SyncResult,
)

# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API — Meta-Autopoietic Level
# ═══════════════════════════════════════════════════════════════════════════════
from core.hrm.meta_level import (
    MetaAutopoieticLevel,
    MetaObservation,
    MetaOperation,
    MetaProposal,
    TriggerCondition,
)

# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API — Hierarchical Engine (The Core)
# ═══════════════════════════════════════════════════════════════════════════════
from core.hrm.hierarchical_engine import (
    HierarchicalReasoningModel,
    HRMConfig,
    HRMCycleResult,
    HRMStatus,
    LevelCycleResult,
)

__all__ = [
    # Version
    "__version__",
    # Abstraction Levels
    "AbstractionLevel",
    "BridgeNodeType",
    "LevelBoundary",
    "LevelConfig",
    "TemporalScale",
    "HRM_SNR_GRADIENT",
    "HRM_TEMPORAL_SCALE",
    "default_level_configs",
    "default_boundaries",
    # Cross-Level Bridge
    "CascadeResult",
    "CrossLevelBridge",
    "CrossLevelMessage",
    "MessageType",
    "PropagationDirection",
    "SyncResult",
    # Meta-Autopoietic Level
    "MetaAutopoieticLevel",
    "MetaObservation",
    "MetaOperation",
    "MetaProposal",
    "TriggerCondition",
    # Hierarchical Engine
    "HierarchicalReasoningModel",
    "HRMConfig",
    "HRMCycleResult",
    "HRMStatus",
    "LevelCycleResult",
]
