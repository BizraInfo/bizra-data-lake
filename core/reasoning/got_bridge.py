"""
+==============================================================================+
|   GoT Bridge -- Phase 46: Cognitive Resonance                                |
+==============================================================================+
|   Bridges FAISS vector search evidence into the Graph-of-Thoughts            |
|   reasoning pipeline.  Converts semantic search results into GoT             |
|   context facts, invokes the canonical GraphOfThoughts.reason() method,      |
|   and applies a convergence gate (SNR >= GOT_CONVERGENCE_SNR).               |
|                                                                              |
|   Graceful degradation:                                                      |
|   - No search engine  -> reason with user-supplied or empty facts            |
|   - No GoT engine     -> template-based fallback response                    |
|   - Neither available -> minimal result with converged=False                 |
|                                                                              |
|   Standing on Giants:                                                        |
|   Besta (GoT, 2024) . Shannon (1948) . Johnson (FAISS, 2021)                |
+==============================================================================+

Created: 2026-02-19 | Phase 46 Cognitive Resonance
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

from core.integration.constants import (
    GOT_CONVERGENCE_SNR,
    GOT_MAX_DEPTH,
    GOT_MAX_HYPOTHESES,
)
from core.memory.types import SearchResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature flag — mirrors pattern in core/search/vector_search.py
# ---------------------------------------------------------------------------
PHASE46_GOT_BRIDGE_ENABLED: bool = os.getenv(
    "BIZRA_PHASE46_GOT_BRIDGE_ENABLED", "0"
).lower() in {"1", "true", "yes"}


# ---------------------------------------------------------------------------
# Result dataclass (frozen -- immutable after construction)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GoTBridgeResult:
    """Immutable result from a GoT Bridge reasoning pass.

    Named GoTBridgeResult (not ReasoningResult) to avoid collision with
    ``core.reasoning.graph_types.ReasoningResult``.
    """

    answer: str
    hypotheses_explored: int
    hypotheses_surviving: int
    evidence: list[SearchResult]
    snr_score: float
    convergence_path: list[str]
    reasoning_depth: int
    converged: bool


# ---------------------------------------------------------------------------
# Bridge class
# ---------------------------------------------------------------------------
class GoTBridge:
    """Bridge that injects FAISS vector evidence into Graph-of-Thoughts reasoning.

    Parameters
    ----------
    search_engine:
        Optional ``VectorSearchEngine`` for evidence retrieval.
    got_engine:
        Optional ``GraphOfThoughts`` instance.  When *None*, the engine is
        lazily imported from ``core.reasoning.graph_core`` on first use.
    max_hypotheses:
        Cap on hypotheses explored (also used as ``top_k`` for search).
    convergence_snr:
        Minimum SNR for a result to be considered *converged*.
    max_depth:
        Maximum GoT reasoning depth passed to ``GraphOfThoughts.reason()``.
    """

    def __init__(
        self,
        search_engine: Optional[Any] = None,
        got_engine: Optional[Any] = None,
        max_hypotheses: int = GOT_MAX_HYPOTHESES,
        convergence_snr: float = GOT_CONVERGENCE_SNR,
        max_depth: int = GOT_MAX_DEPTH,
    ) -> None:
        self._search_engine = search_engine
        self._got_engine = got_engine
        self._max_hypotheses = max_hypotheses
        self._convergence_snr = convergence_snr
        self._max_depth = max_depth

        # P1: Degradation transparency
        from core.protocols.degradation import DegradationEmitter

        emitter = DegradationEmitter("GoTBridge")
        emitter.check("search_engine", search_engine)
        emitter.check("got_engine", got_engine)
        self._degradation_event = emitter.emit()
        self._degraded = self._degradation_event is not None

    # ------------------------------------------------------------------
    # Lazy GoT engine import (avoids circular imports at module load)
    # ------------------------------------------------------------------

    def _get_got_engine(self) -> Any:
        """Return the GoT engine, lazily importing if not yet initialised."""
        if self._got_engine is None:
            try:
                from core.reasoning.graph_core import GraphOfThoughts

                self._got_engine = GraphOfThoughts()
                logger.info("GoTBridge: lazily initialised GraphOfThoughts engine")
            except Exception:
                logger.warning(
                    "GoTBridge: GraphOfThoughts unavailable -- "
                    "falling back to template mode"
                )
        return self._got_engine

    # ------------------------------------------------------------------
    # Evidence helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _evidence_to_facts(results: list[SearchResult]) -> list[str]:
        """Convert search results into fact strings for GoT context.

        Each fact is formatted as ``[source] content_preview`` (max 200 chars
        of content) so the GoT engine can ground its hypotheses.
        """
        return [f"[{r.record.source}] {r.record.content[:200]}" for r in results]

    def _search_for_evidence(self, query: str) -> list[SearchResult]:
        """Run a FAISS search if a search engine is available."""
        if self._search_engine is None:
            return []
        try:
            results: list[SearchResult] = self._search_engine.search(
                query, top_k=self._max_hypotheses
            )
            logger.debug(
                "GoTBridge: retrieved %d evidence items for query",
                len(results),
            )
            return results
        except Exception as exc:
            logger.warning("GoTBridge: evidence search failed -- %s", exc)
            return []

    # ------------------------------------------------------------------
    # Template-based fallback (no GoT engine available)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_fallback_result(
        query: str,
        evidence: list[SearchResult],
        facts: list[str],
    ) -> GoTBridgeResult:
        """Build a minimal GoTBridgeResult when no GoT engine is available.

        The answer is a template concatenation of the query and any available
        evidence.  ``converged`` is always *False* since no real reasoning
        occurred.
        """
        if facts:
            answer = f"Based on {len(facts)} evidence item(s): " + "; ".join(
                f[:120] for f in facts[:3]
            )
        else:
            answer = query

        return GoTBridgeResult(
            answer=answer,
            hypotheses_explored=0,
            hypotheses_surviving=0,
            evidence=evidence,
            snr_score=0.0,
            convergence_path=["fallback_no_got_engine"],
            reasoning_depth=0,
            converged=False,
        )

    # ------------------------------------------------------------------
    # Core reasoning pipeline
    # ------------------------------------------------------------------

    async def _run_reasoning(
        self,
        query: str,
        context: dict[str, Any],
        evidence: list[SearchResult],
    ) -> GoTBridgeResult:
        """Execute the full bridge pipeline: evidence -> GoT -> convergence gate.

        This is the shared implementation behind both ``reason()`` and
        ``reason_with_evidence()``.
        """
        # -- Step 1: Convert evidence to facts and merge into context ------
        facts = self._evidence_to_facts(evidence)
        existing_facts: list[str] = context.get("facts", [])
        merged_facts = existing_facts + facts
        context_with_facts: dict[str, Any] = {**context, "facts": merged_facts}

        # -- Step 2: Try to invoke the GoT engine -------------------------
        got = self._get_got_engine()
        if got is None:
            return self._build_fallback_result(query, evidence, merged_facts)

        try:
            raw: dict[str, Any] = await got.reason(
                query, context_with_facts, self._max_depth
            )
        except Exception as exc:
            logger.error("GoTBridge: GoT engine raised -- %s", exc)
            return self._build_fallback_result(query, evidence, merged_facts)

        # -- Step 3: Extract fields from GoT result -----------------------
        conclusion: str = raw.get("conclusion", query)
        snr_score: float = float(raw.get("snr_score", 0.0))
        graph_stats: dict[str, Any] = raw.get("graph_stats", {})
        depth_reached: int = int(raw.get("depth_reached", 0))
        thoughts: list[str] = raw.get("thoughts", [])

        # Heuristic: hypotheses surviving = those that contributed to the
        # conclusion (proxy: thoughts containing "Hypothesis" that were not
        # pruned).
        hypotheses_explored = sum(
            1 for t in thoughts if "Hypothesis" in t or "hypothesis" in t
        )
        hypotheses_surviving = max(
            hypotheses_explored - graph_stats.get("nodes_pruned", 0), 0
        )

        # -- Step 4: Convergence gate -------------------------------------
        converged = snr_score >= self._convergence_snr

        # Build the convergence path from thought log
        convergence_path: list[str] = []
        for thought in thoughts:
            convergence_path.append(thought)
        if converged:
            convergence_path.append(
                f"CONVERGED: SNR {snr_score:.3f} >= {self._convergence_snr}"
            )
        else:
            convergence_path.append(
                f"NOT_CONVERGED: SNR {snr_score:.3f} < {self._convergence_snr}"
            )

        return GoTBridgeResult(
            answer=conclusion,
            hypotheses_explored=hypotheses_explored,
            hypotheses_surviving=hypotheses_surviving,
            evidence=evidence,
            snr_score=snr_score,
            convergence_path=convergence_path,
            reasoning_depth=depth_reached,
            converged=converged,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def reason(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> GoTBridgeResult:
        """Run GoT reasoning with optional automatic FAISS evidence retrieval.

        1. If a search engine is available, retrieve top-k evidence for *query*.
        2. Convert evidence to fact strings and merge into *context*.
        3. Invoke ``GraphOfThoughts.reason()`` (or fallback).
        4. Apply convergence gate (SNR >= ``convergence_snr``).

        Parameters
        ----------
        query:
            Natural language question or task.
        context:
            Optional dict with ``domain``, ``constraints``, ``facts`` keys.

        Returns
        -------
        GoTBridgeResult
            Frozen dataclass with answer, scores, evidence, and convergence
            status.
        """
        ctx = context if context is not None else {}
        evidence = self._search_for_evidence(query)
        return await self._run_reasoning(query, ctx, evidence)

    async def reason_with_evidence(
        self,
        query: str,
        evidence: list[SearchResult],
        context: dict[str, Any] | None = None,
    ) -> GoTBridgeResult:
        """Run GoT reasoning with pre-provided evidence (skips FAISS search).

        Identical to ``reason()`` but uses the caller-supplied *evidence*
        list instead of performing a search.

        Parameters
        ----------
        query:
            Natural language question or task.
        evidence:
            Pre-retrieved search results to inject as facts.
        context:
            Optional dict with ``domain``, ``constraints``, ``facts`` keys.

        Returns
        -------
        GoTBridgeResult
        """
        ctx = context if context is not None else {}
        return await self._run_reasoning(query, ctx, evidence)


__all__ = [
    "PHASE46_GOT_BRIDGE_ENABLED",
    "GoTBridgeResult",
    "GoTBridge",
]
