"""
SNR Protocol — Unified Interface for Signal-to-Noise Ratio Engines
====================================================================

Standing on Giants:
- Shannon (1948): Information Theory — foundational SNR definition
- PEP 544 (2017): Structural subtyping — Protocol for static conformance

BIZRA has TWO legitimate SNR engines with different measurement domains:

    ┌─────────────────────────────────────────────────────┐
    │                    SNRProtocol                       │
    │  calculate_snr() → (score, metrics, ihsan_achieved) │
    └───────────────┬─────────────────┬───────────────────┘
                    │                 │
    ┌───────────────┴──┐   ┌─────────┴──────────────┐
    │  arte_engine      │   │  snr_maximizer           │
    │  .SNREngine       │   │  .SNRMaximizer           │
    │                   │   │                          │
    │  Domain: Vectors  │   │  Domain: Text            │
    │  Method: Cosine   │   │  Method: Keyword + dB    │
    │  Scale: [0, 1]    │   │  Scale: linear & dB      │
    │  Input: np.ndarray│   │  Input: str              │
    │  Use: Retrieval   │   │  Use: Generation QA      │
    └───────────────────┘   └──────────────────────────┘

Consumers should depend on SNRProtocol, not concrete classes.
The SNRFacade provides a single entry point that dispatches to
the appropriate engine based on input type.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

logger = logging.getLogger(__name__)


# ── Unified Result ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class SNRResult:
    """
    Canonical SNR measurement result.

    All engines produce this, regardless of internal representation.
    Consumers only need to check `score`, `ihsan_achieved`, and optionally
    drill into `metrics` for engine-specific details.
    """
    score: float                         # Normalized to [0.0, 1.0]
    ihsan_achieved: bool                 # score >= ihsan_threshold
    engine: str                          # "embedding" | "text" | "ensemble"
    metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        status = "PASS" if self.ihsan_achieved else "FAIL"
        return f"SNRResult(score={self.score:.4f}, ihsan={status}, engine={self.engine})"


# ── Protocol ────────────────────────────────────────────────────────────

@runtime_checkable
class SNRProtocol(Protocol):
    """
    Structural typing contract for SNR engines.

    Any class implementing `calculate_snr_normalized()` conforming to this
    signature is a valid SNR engine — no inheritance required (PEP 544).
    """

    def calculate_snr_normalized(self, **kwargs: Any) -> SNRResult:
        """
        Compute SNR and return a normalized result.

        Keyword arguments vary by engine:
        - Embedding engine: query_embedding, context_embeddings, symbolic_facts, neural_results
        - Text engine: text, query, sources
        """
        ...


# ── Facade ──────────────────────────────────────────────────────────────

class SNRFacade:
    """
    Unified SNR entry point for the orchestrator.

    Routes to the appropriate engine based on available inputs:
    - If embeddings are provided → arte_engine.SNREngine (vector-space)
    - If text is provided → snr_maximizer.SNRMaximizer (text-space)
    - If both → ensemble (geometric mean of both scores)

    Usage:
        facade = SNRFacade(embedding_engine=snr_engine, text_engine=maximizer)
        result = facade.calculate(text=answer, query_embedding=qe, context_embeddings=ce, ...)
    """

    def __init__(
        self,
        embedding_engine: Optional[Any] = None,
        text_engine: Optional[Any] = None,
        ihsan_threshold: float = 0.95,
    ):
        self.embedding_engine = embedding_engine
        self.text_engine = text_engine
        self.ihsan_threshold = ihsan_threshold

    def calculate(
        self,
        *,
        # Text engine inputs
        text: Optional[str] = None,
        query: Optional[str] = None,
        sources: Optional[List[str]] = None,
        # Embedding engine inputs
        query_embedding: Optional[Any] = None,
        context_embeddings: Optional[Any] = None,
        symbolic_facts: Optional[List[Dict]] = None,
        neural_results: Optional[List[Dict]] = None,
    ) -> SNRResult:
        """
        Dispatch to the appropriate SNR engine(s) based on available inputs.

        Returns a single canonical SNRResult.
        """
        import numpy as np

        has_embeddings = (
            query_embedding is not None
            and context_embeddings is not None
            and self.embedding_engine is not None
        )
        has_text = text is not None and self.text_engine is not None

        if has_embeddings and has_text:
            return self._ensemble(
                text=text, query=query, sources=sources,
                query_embedding=query_embedding,
                context_embeddings=context_embeddings,
                symbolic_facts=symbolic_facts or [],
                neural_results=neural_results or [],
            )
        elif has_embeddings:
            return self._from_embedding_engine(
                query_embedding=query_embedding,
                context_embeddings=context_embeddings,
                symbolic_facts=symbolic_facts or [],
                neural_results=neural_results or [],
            )
        elif has_text:
            return self._from_text_engine(text=text, query=query, sources=sources)
        else:
            logger.warning("SNRFacade: No valid inputs — returning baseline")
            return SNRResult(
                score=0.0, ihsan_achieved=False, engine="none",
                recommendations=["Provide text or embeddings for SNR calculation"],
            )

    def _from_embedding_engine(self, **kwargs: Any) -> SNRResult:
        """Delegate to arte_engine.SNREngine and normalize."""
        score, metrics = self.embedding_engine.calculate_snr(**kwargs)
        return SNRResult(
            score=float(score),
            ihsan_achieved=score >= self.ihsan_threshold,
            engine="embedding",
            metrics=metrics,
        )

    def _from_text_engine(
        self, text: str, query: Optional[str] = None, sources: Optional[List[str]] = None,
    ) -> SNRResult:
        """Delegate to snr_maximizer.SNRMaximizer and normalize."""
        analysis = self.text_engine.analyze(text, query, sources)

        # Normalize dB-scale to [0, 1] via sigmoid-like mapping
        # snr_linear is already a ratio; clamp to [0, 1]
        normalized_score = min(max(analysis.snr_linear, 0.0), 1.0)

        return SNRResult(
            score=normalized_score,
            ihsan_achieved=analysis.ihsan_achieved,
            engine="text",
            metrics=analysis.to_dict(),
            recommendations=analysis.recommendations,
        )

    def _ensemble(self, *, text: str, query: Optional[str], sources: Optional[List[str]],
                  **embedding_kwargs: Any) -> SNRResult:
        """
        Ensemble: geometric mean of both engines.

        Geometric mean is appropriate because both scores are in [0,1]
        and we want a single zero to dominate (fail-closed behavior).
        """
        import math

        emb_result = self._from_embedding_engine(**embedding_kwargs)
        txt_result = self._from_text_engine(text=text, query=query, sources=sources)

        epsilon = 1e-10
        ensemble_score = math.exp(
            0.5 * math.log(emb_result.score + epsilon)
            + 0.5 * math.log(txt_result.score + epsilon)
        )
        ensemble_score = min(max(ensemble_score, 0.0), 1.0)

        # Merge recommendations
        all_recs = list(set(emb_result.recommendations + txt_result.recommendations))

        return SNRResult(
            score=round(ensemble_score, 4),
            ihsan_achieved=ensemble_score >= self.ihsan_threshold,
            engine="ensemble",
            metrics={
                "embedding_snr": emb_result.score,
                "text_snr": txt_result.score,
                "ensemble_method": "geometric_mean",
                "embedding_metrics": emb_result.metrics,
                "text_metrics": txt_result.metrics,
            },
            recommendations=all_recs,
        )
