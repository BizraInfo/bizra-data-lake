"""
+==============================================================================+
|   BIZRA REASONING -- Graph-of-Thoughts & Quality Validation                   |
+==============================================================================+
|   Advanced reasoning systems including graph-based thought exploration,      |
|   SNR maximization, and multi-guardian consensus validation.                 |
|                                                                              |
|   Components:                                                                |
|   - graph_reasoner: Graph-of-Thoughts exploration (Besta et al., 2024)       |
|   - guardian_council: Multi-perspective validation                           |
|   - snr_maximizer: Shannon-inspired signal quality optimization              |
|   - bicameral_engine: Dual-hemisphere reasoning (Jaynes, 1976)               |
|                                                                              |
|   Constitutional Constraint: All reasoning validated against Ihsan >= 0.95   |
|                                                                              |
|   Standing on Giants:                                                        |
|   - Besta et al. (2024): Graph of Thoughts                                   |
|   - Shannon (1948): Information Theory                                       |
|   - Jaynes (1976): Bicameral Mind                                            |
+==============================================================================+

Created: 2026-02-05 | SAPE Sovereign Module Decomposition
Migrated: 2026-02-05 | Files now in dedicated reasoning package
"""

# --------------------------------------------------------------------------
# PHASE 1: Safe imports (no cross-package dependencies)
# --------------------------------------------------------------------------

# graph_types: Enums and data classes
from .graph_types import (
    ThoughtType,
    EdgeType,
    ReasoningStrategy,
    ThoughtNode,
    ThoughtEdge,
    ReasoningPath,
    ReasoningResult,
)
# graph_operations / graph_search / graph_reasoning: Mixins (composed in GraphOfThoughts)
from .graph_operations import GraphOperationsMixin
from .graph_search import GraphSearchMixin
from .graph_reasoning import GraphReasoningMixin
# graph_core: Main composed class
from .graph_core import GraphOfThoughts
# Guardian council
from .guardian_council import (
    GuardianCouncil,
    Guardian,
    CouncilVerdict,
)
# SNR Maximizer
from .snr_maximizer import (
    SNRMaximizer,
)
# Bicameral Engine
from .bicameral_engine import BicameralReasoningEngine

# --------------------------------------------------------------------------
# PHASE 2: Lazy imports for modules with cross-package dependencies.
# collective_intelligence and collective_synthesizer import from
# core.orchestration.team_planner, which can cause circular imports.
# --------------------------------------------------------------------------
_LAZY_MODULES = {
    "CollectiveIntelligence": (".collective_intelligence", "CollectiveIntelligence"),
    "CollectiveSynthesizer": (".collective_synthesizer", "CollectiveSynthesizer"),
}


def __getattr__(name: str):
    if name in _LAZY_MODULES:
        module_path, attr_name = _LAZY_MODULES[name]
        import importlib
        mod = importlib.import_module(module_path, __name__)
        value = getattr(mod, attr_name)
        globals()[name] = value  # Cache for subsequent access
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Graph Types
    "ThoughtType",
    "EdgeType",
    "ReasoningStrategy",
    "ThoughtNode",
    "ThoughtEdge",
    "ReasoningPath",
    "ReasoningResult",
    # Graph Mixins
    "GraphOperationsMixin",
    "GraphSearchMixin",
    "GraphReasoningMixin",
    # Graph Reasoning (main class)
    "GraphOfThoughts",
    # Guardian Council
    "GuardianCouncil",
    "Guardian",
    "CouncilVerdict",
    # SNR Maximizer
    "SNRMaximizer",
    # Bicameral Engine
    "BicameralReasoningEngine",
    # Collective (lazy-loaded)
    "CollectiveIntelligence",
    "CollectiveSynthesizer",
]
