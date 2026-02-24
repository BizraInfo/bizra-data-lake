"""
SNR Rust Adapter — Bridges Rust SNREngine (PyO3) to SNRProtocol

Standing on Giants:
- Shannon (1948): Unified SNR measurement across Rust and Python
- PEP 544 (2017): Structural subtyping via Protocol

Gap G-2 Bridge: Exposes Rust's weighted geometric mean SNR engine
to the Python SNRFacade priority chain. Falls back gracefully when
the Rust binding (bizra) is not available.

Architecture:
    bizra-core/src/sovereign/snr_engine.rs (Rust)
        ↓ PyO3
    bizra-python/src/lib.rs (PySNREngine wrapper)
        ↓ import bizra
    core/iaas/snr_rust_adapter.py (this file — SNRProtocol conformance)
        ↓
    core/snr_protocol.py (SNRFacade.rust_engine parameter)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from core.integration.constants import UNIFIED_IHSAN_THRESHOLD, UNIFIED_SNR_THRESHOLD
from core.snr_protocol import SNRResult

logger = logging.getLogger(__name__)

# ── Lazy import of the Rust binding ─────────────────────────────────────

_RUST_SNR_AVAILABLE = False
_RustSNREngine: Optional[type] = None

try:
    from bizra import SNREngine as _RustSNREngine  # type: ignore[import-untyped]

    _RUST_SNR_AVAILABLE = True
except ImportError:
    pass


def is_rust_snr_available() -> bool:
    """Check if the Rust SNR engine binding is available."""
    return _RUST_SNR_AVAILABLE


# ── Adapter ────────────────────────────────────────────────────────────


class SNRRustAdapter:
    """
    Adapts the Rust PySNREngine to conform to SNRProtocol.

    Wraps analyze_text() and returns canonical SNRResult.
    When the Rust binding is unavailable, raises ImportError at construction.

    Usage:
        adapter = SNRRustAdapter()  # Uses default thresholds from constants.py
        result = adapter.calculate_snr_normalized(text="some content")
    """

    def __init__(
        self,
        snr_floor: float = UNIFIED_SNR_THRESHOLD,
        ihsan_target: float = UNIFIED_IHSAN_THRESHOLD,
    ):
        if not _RUST_SNR_AVAILABLE or _RustSNREngine is None:
            raise ImportError(
                "Rust SNR engine not available. "
                "Build with: cd bizra-omega/bizra-python && maturin develop --release"
            )
        self._engine = _RustSNREngine(
            snr_floor=snr_floor, ihsan_target=ihsan_target
        )
        self._ihsan_threshold = ihsan_target

    def calculate_snr_normalized(self, **kwargs: Any) -> SNRResult:
        """
        SNRProtocol-conforming entry point.

        Accepts:
        - text (str): Content to analyze (required)
        - query (str): Optional query (accepted for compatibility)
        """
        text = kwargs.get("text", "")
        if not text:
            return SNRResult(
                score=0.0,
                ihsan_achieved=False,
                engine="rust",
                recommendations=["No text provided for Rust SNR analysis"],
            )

        try:
            metrics = self._engine.analyze_text(text)
        except (ValueError, RuntimeError) as e:
            logger.warning("SNRRustAdapter: analyze_text failed: %s", e)
            return SNRResult(
                score=0.0,
                ihsan_achieved=False,
                engine="rust",
                recommendations=[f"Rust SNR analysis failed: {e}"],
            )

        score = float(metrics["snr"])

        return SNRResult(
            score=score,
            ihsan_achieved=score >= self._ihsan_threshold,
            engine="rust",
            metrics={
                "signal_strength": metrics["signal_strength"],
                "noise_level": metrics["noise_level"],
                "diversity": metrics["diversity"],
                "grounding": metrics["grounding"],
                "balance": metrics["balance"],
                "word_count": metrics["word_count"],
                "unique_words": metrics["unique_words"],
                "analysis_duration_us": metrics["analysis_duration_us"],
                "aggregation": "weighted_geometric_mean",
            },
            recommendations=_build_recommendations(metrics),
        )

    def stats(self) -> dict[str, Any]:
        """Return engine statistics."""
        return dict(self._engine.stats())

    def average_snr(self) -> float:
        """Return rolling average SNR."""
        return self._engine.average_snr()


def _build_recommendations(metrics: dict[str, Any]) -> list[str]:
    """Generate actionable recommendations from Rust metrics."""
    recs: list[str] = []
    if metrics.get("signal_strength", 1.0) < 0.6:
        recs.append("Improve content information density")
    if metrics.get("diversity", 1.0) < 0.5:
        recs.append("Increase vocabulary diversity")
    if metrics.get("noise_level", 0.0) > 0.3:
        recs.append("Reduce filler and redundant content")
    if metrics.get("grounding", 1.0) < 0.5:
        recs.append("Add factual grounding (evidence, citations)")
    return recs


def create_rust_snr_adapter(
    snr_floor: float = UNIFIED_SNR_THRESHOLD,
    ihsan_target: float = UNIFIED_IHSAN_THRESHOLD,
) -> Optional[SNRRustAdapter]:
    """
    Factory function: create adapter if Rust binding available, else None.

    Safe for use in initialization paths — never raises.
    """
    if not _RUST_SNR_AVAILABLE:
        return None
    try:
        return SNRRustAdapter(snr_floor=snr_floor, ihsan_target=ihsan_target)
    except Exception as e:
        logger.debug("Failed to create Rust SNR adapter: %s", e)
        return None


__all__ = [
    "SNRRustAdapter",
    "create_rust_snr_adapter",
    "is_rust_snr_available",
]
