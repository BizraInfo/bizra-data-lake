"""
Embedding Service — Tiered embedding generation with local-first fallback.

Standing on Giants: Reimers & Gurevych (2019, sentence-BERT)
"""

from .quality_gate import EmbeddingQualityGate, GateResult
from .service import EmbeddingConfig, EmbeddingService, EmbeddingUnavailableError

__all__ = [
    "EmbeddingService",
    "EmbeddingConfig",
    "EmbeddingUnavailableError",
    "EmbeddingQualityGate",
    "GateResult",
]
