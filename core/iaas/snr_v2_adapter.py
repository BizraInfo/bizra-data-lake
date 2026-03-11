"""
SNR v2 Protocol Adapter — Bridges SNRCalculatorV2 to SNRProtocol

Standing on Giants:
- Shannon (1948): Unified measurement framework
- PEP 544 (2017): Structural subtyping via Protocol

Phase 42 Spec 02: Makes SNRCalculatorV2 the primary embedding engine
in SNRFacade, conforming to the canonical SNRProtocol interface.
"""

from __future__ import annotations

import logging
from typing import Any

from core.integration.constants import UNIFIED_IHSAN_THRESHOLD
from core.snr_protocol import SNRResult

logger = logging.getLogger(__name__)


class SNRv2Adapter:
    """
    Adapts SNRCalculatorV2 to conform to SNRProtocol.

    Wraps compute_snr() / calculate_simple() and returns canonical SNRResult.
    """

    def __init__(
        self,
        calculator: Any,
        ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD,
    ):
        self._calculator = calculator
        self._ihsan_threshold = ihsan_threshold

    def calculate_snr_normalized(self, **kwargs: Any) -> SNRResult:
        """
        SNRProtocol-conforming entry point.

        Accepts flexible kwargs:
        - query (str): Query or task description
        - texts (list[str]): Texts to evaluate
        - text (str): Single text (wrapped to list)
        - query_embedding (ndarray): Optional query embedding
        - text_embeddings / context_embeddings (ndarray): Optional text embeddings
        """
        query = kwargs.get("query") or kwargs.get("text", "")
        texts = kwargs.get("texts")
        if texts is None:
            single = kwargs.get("text", "")
            texts = [single] if single else [""]

        query_embedding = kwargs.get("query_embedding")
        text_embeddings = kwargs.get("text_embeddings") or kwargs.get(
            "context_embeddings"
        )

        try:
            if query_embedding is not None and text_embeddings is not None:
                components = self._calculator.compute_snr(
                    query=query,
                    texts=texts,
                    query_embedding=query_embedding,
                    text_embeddings=text_embeddings,
                )
            else:
                components = self._calculator.calculate_simple(query=query, texts=texts)
        except Exception as e:  # noqa: BLE001 — boundary boundary
            logger.warning(f"SNRv2Adapter: compute failed: {e}")
            return SNRResult(
                score=0.0,
                ihsan_achieved=False,
                engine="snr_v2",
                recommendations=[f"SNR v2 computation failed: {e}"],
            )

        score = components.snr
        recs = _build_recommendations(components)

        return SNRResult(
            score=score,
            ihsan_achieved=score >= self._ihsan_threshold,
            engine="snr_v2",
            metrics={
                "signal_strength": components.signal_strength,
                "diversity": components.diversity,
                "grounding": components.grounding,
                "iaas_score": components.iaas_score,
                "semantic_relevance": components.semantic_relevance,
                "channel_efficiency": components.channel_efficiency,
                "quality_tier": components.quality_tier,
                "redundancy": components.redundancy,
                "entropy": components.entropy,
            },
            recommendations=recs,
        )


def _build_recommendations(components: Any) -> list[str]:
    """Generate actionable recommendations from v2 components."""
    recs: list[str] = []
    if components.signal_strength < 0.6:
        recs.append("Improve semantic alignment with query")
    if components.diversity < 0.5:
        recs.append("Increase source diversity (Renyi-2 detected concentration)")
    if components.redundancy > 0.4:
        recs.append("Reduce redundant content")
    if components.grounding < 0.5:
        recs.append("Add grounding evidence or citations")
    return recs


__all__ = ["SNRv2Adapter"]
