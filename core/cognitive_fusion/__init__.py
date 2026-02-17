"""
BIZRA Cognitive Fusion -- Package Root

Bridges MoE complexity routing, HRM hierarchical reasoning,
HyperGraph RAG retrieval, and NorthStar quality gates into a
single four-stage cognitive inference pipeline.

Standing on Giants: Vaswani + Simon + Shannon + Besta

Created: 2026-02-17 | BIZRA Node0 | Cognitive Fusion Phase
"""

__version__ = "1.0.0"
__author__ = "BIZRA Node0"

from core.cognitive_fusion.complexity_adapter import ComplexityAdapter
from core.cognitive_fusion.fusion_engine import (
    CognitiveFusionEngine,
    FusionResult,
    HRMResult,
    MoERouterProtocol,
    NorthStarResult,
    RoutingResult,
)

__all__ = [
    "CognitiveFusionEngine",
    "ComplexityAdapter",
    "FusionResult",
    "HRMResult",
    "MoERouterProtocol",
    "NorthStarResult",
    "RoutingResult",
]
