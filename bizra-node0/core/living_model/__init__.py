"""Living Model — Mixture-of-Experts routing engine.

Standing on: Shazeer (2017) sparsely-gated MOE, Kahneman (2011) System-2
multi-expert deliberation, Ibn Khaldun (1377) Asabiyyah specialization.
"""

from __future__ import annotations

from core.living_model.moe_engine import (
    Expert,
    ExpertAssignment,
    ExpertResult,
    MOEEngine,
    MOEEngineStats,
    SynthesisResult,
)

__all__ = [
    "Expert",
    "ExpertAssignment",
    "ExpertResult",
    "MOEEngine",
    "MOEEngineStats",
    "SynthesisResult",
]
