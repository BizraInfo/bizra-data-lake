"""Cognitive Resonance Orchestrator — Phase 46.

Pipeline: search → reason-with-evidence → predict.
Gracefully degrades when any component is unavailable.

Standing on Giants: Shannon (1948) · Besta (GoT, 2024) · Rabiner (HMM, 1989)
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.memory.types import SearchResult

logger = logging.getLogger(__name__)

PHASE46_ENABLED: bool = os.getenv("BIZRA_PHASE46_SEARCH_ENABLED", "0").lower() in {
    "1",
    "true",
    "yes",
}


@dataclass(frozen=True)
class ResonanceResult:
    """Unified result from the cognitive resonance pipeline."""

    query: str
    search_results: List[SearchResult]
    reasoning: Any  # GoTBridgeResult | None (avoid import at module level)
    prediction: Any  # PredictionResult | None
    combined_snr: float
    processing_path: List[str]


class CognitiveResonance:
    """Orchestration facade: search → reason → predict.

    Each component is optional. Pipeline degrades gracefully
    when a component is unavailable or raises an exception.
    """

    def __init__(
        self,
        search: Optional[Any] = None,  # VectorSearchEngine
        reasoning: Optional[Any] = None,  # GoTBridge
        prediction: Optional[Any] = None,  # HMMEngine
    ) -> None:
        self._search = search
        self._reasoning = reasoning
        self._prediction = prediction

    # ------------------------------------------------------------------ #
    #  Main pipeline                                                      #
    # ------------------------------------------------------------------ #

    async def process(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ResonanceResult:
        """Run the full resonance pipeline: search → reason → predict."""
        ctx = context or {}
        path: List[str] = []
        search_results: List[SearchResult] = []
        reasoning_result = None
        prediction_result = None
        max_vector_score = 0.0

        # ---- Stage 1: Vector search ----
        if self._search is not None:
            try:
                search_results = self._search.search(query)
                if search_results:
                    max_vector_score = max(r.score for r in search_results)
                path.append(f"search:{len(search_results)}_hits")
                logger.debug(
                    "Resonance search returned %d results", len(search_results)
                )
            except Exception as exc:  # noqa: BLE001 — boundary boundary
                logger.warning("Resonance search failed: %s", exc)
                path.append("search:error")

        # ---- Stage 2: GoT reasoning with evidence ----
        if self._reasoning is not None:
            try:
                if search_results:
                    reasoning_result = await self._reasoning.reason_with_evidence(
                        query, search_results
                    )
                else:
                    reasoning_result = await self._reasoning.reason(query, ctx)
                path.append("reasoning:ok")
            except (
                asyncio.CancelledError,
                RuntimeError,
                OSError,
            ) as exc:  # SEC-003 — async boundary
                logger.warning("Resonance reasoning failed: %s", exc)
                path.append("reasoning:error")

        # ---- Stage 3: HMM observation ----
        if self._prediction is not None:
            try:
                # Map query keywords to closest observation symbol
                symbol = _query_to_symbol(query)
                prediction_result = self._prediction.observe(symbol)
                path.append(f"prediction:{prediction_result.most_likely_state.value}")
            except (
                asyncio.CancelledError,
                RuntimeError,
                OSError,
            ) as exc:  # SEC-003 — async boundary
                logger.warning("Resonance prediction failed: %s", exc)
                path.append("prediction:error")

        # ---- Compute combined SNR ----
        combined_snr = _compute_combined_snr(reasoning_result, max_vector_score)
        path.append(f"snr:{combined_snr:.3f}")

        return ResonanceResult(
            query=query,
            search_results=search_results,
            reasoning=reasoning_result,
            prediction=prediction_result,
            combined_snr=combined_snr,
            processing_path=path,
        )

    # ------------------------------------------------------------------ #
    #  Convenience: observe-only (no search/reason)                       #
    # ------------------------------------------------------------------ #

    def observe(self, action: str) -> Any:
        """Feed an action symbol to the HMM without full pipeline."""
        if self._prediction is None:
            return None
        try:
            return self._prediction.observe(action)
        except Exception as exc:  # noqa: BLE001 — boundary boundary
            logger.warning("Resonance observe failed: %s", exc)
            return None


# ====================================================================== #
#  Helpers                                                                #
# ====================================================================== #

_SYMBOL_KEYWORDS: Dict[str, List[str]] = {
    "search": ["search", "find", "query", "look", "where"],
    "edit": ["edit", "change", "modify", "update", "fix", "refactor"],
    "navigate": ["open", "go", "navigate", "show", "view"],
    "organize": ["organize", "sort", "move", "rename", "clean"],
    "review": ["review", "check", "verify", "audit", "inspect"],
    "compile": ["build", "compile", "run", "execute"],
    "test": ["test", "pytest", "assert", "validate"],
    "chat": ["chat", "ask", "tell", "explain", "summarize"],
    "deploy": ["deploy", "push", "release", "publish"],
    "file_open": ["read", "load", "import", "fetch"],
    "file_save": ["save", "write", "export", "commit"],
    "idle": ["idle", "wait", "pause"],
}


def _query_to_symbol(query: str) -> str:
    """Map a free-text query to the closest HMM observation symbol."""
    words = set(query.lower().split())
    best_symbol = "idle"
    best_overlap = 0
    for symbol, keywords in _SYMBOL_KEYWORDS.items():
        overlap = len(words & set(keywords))
        if overlap > best_overlap:
            best_overlap = overlap
            best_symbol = symbol
    return best_symbol


def _compute_combined_snr(
    reasoning_result: Any,  # GoTBridgeResult | None
    max_vector_score: float,
) -> float:
    """Compute combined SNR per policy.

    - Both search + reasoning: 0.7 * reasoning.snr + 0.3 * max_vector_score
    - Only reasoning:          reasoning.snr_score
    - Only search:             max_vector_score
    - Neither:                 0.0
    """
    has_reasoning = reasoning_result is not None and hasattr(
        reasoning_result, "snr_score"
    )
    has_search = max_vector_score > 0.0

    if has_reasoning and has_search:
        return 0.7 * reasoning_result.snr_score + 0.3 * max_vector_score
    if has_reasoning:
        return reasoning_result.snr_score
    if has_search:
        return max_vector_score
    return 0.0
