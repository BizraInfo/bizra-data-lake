"""Mixture-of-Experts routing engine — 5-expert top-K selection with synthesis.

Standing on Giants:
- Shazeer (2017): Sparsely-gated Mixture of Experts — top-K routing
- Kahneman (2011): System-2 deliberation uses multiple experts, not one
- Ibn Khaldun (1377): Asabiyyah — collective intelligence from specialized roles
- Boyd (1976): OODA loop — observe (score), orient (route), decide (synthesize), act (gate)

Architecture:
  Input → HHMM Router → Expert Scoring → Top-K Selection → Synthesis → Ihsan Gate

Each expert is a lightweight scoring function (keyword + HHMM state matching).
No LLM call in the routing layer — sub-millisecond overhead.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from core.integration.constants import (
    MOE_FALLBACK_EXPERT,
    MOE_MIN_CONFIDENCE,
    MOE_SYNTHESIS_STRATEGY,
    MOE_TOP_K,
    UNIFIED_IHSAN_THRESHOLD,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ExpertAssignment:
    """Routing decision: which expert handles the query, at what weight."""

    expert_id: str
    weight: float  # 0.0 - 1.0, normalized across selected experts


@dataclass(frozen=True)
class ExpertResult:
    """Output from a single expert's execution."""

    expert_id: str
    text: str
    ihsan: float
    confidence: float
    latency_ms: float = 0.0


@dataclass(frozen=True)
class SynthesisResult:
    """Combined output from all activated experts."""

    text: str
    ihsan: float
    passed_gate: bool
    reason: str = ""
    experts_used: tuple[str, ...] = ()
    total_latency_ms: float = 0.0


@dataclass
class MOEEngineStats:
    """Telemetry for MOE routing decisions."""

    total_routes: int = 0
    expert_activations: dict[str, int] = field(default_factory=dict)
    avg_experts_per_query: float = 0.0
    gate_rejections: int = 0
    _total_experts_activated: int = field(default=0, repr=False)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERT DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════

# Word boundary pattern cache — compiled once at module load
_WORD_RE_CACHE: dict[str, re.Pattern[str]] = {}


def _word_pattern(keyword: str) -> re.Pattern[str]:
    """Get or compile a word-boundary regex for a keyword."""
    if keyword not in _WORD_RE_CACHE:
        _WORD_RE_CACHE[keyword] = re.compile(
            r"\b" + re.escape(keyword) + r"\b", re.IGNORECASE
        )
    return _WORD_RE_CACHE[keyword]


@dataclass(frozen=True)
class Expert:
    """A lightweight expert that scores its relevance for a given input.

    No LLM call — keyword + HHMM state matching only.
    """

    id: str
    keywords: frozenset[str]
    hhmm_states: frozenset[str]
    base_weight: float = 1.0

    def score_relevance(
        self,
        input_text: str,
        macro_state: str = "general",
        context: dict[str, Any] | None = None,
    ) -> float:
        """Score this expert's relevance to the input. Returns [0.0, 1.0]."""
        if not input_text:
            return max(0.0, min(self.base_weight * 0.2, 1.0))

        # Keyword scoring: fraction of keywords found, with activation floor.
        # A single keyword match should meaningfully activate the expert.
        # Standing on: Shazeer (2017) — even weak gating signal beats no signal.
        hits = sum(1 for kw in self.keywords if _word_pattern(kw).search(input_text))
        keyword_score = hits / max(len(self.keywords), 1)
        if hits > 0:
            keyword_score = max(keyword_score, 0.3)  # Activation floor

        # HHMM state matching: binary boost
        state_score = 1.0 if macro_state in self.hhmm_states else 0.0

        # Context boost: domain-specific signals
        context_score = self._context_boost(context or {})

        raw = (keyword_score * 0.5) + (state_score * 0.3) + (context_score * 0.2)
        return max(0.0, min(raw * self.base_weight, 1.0))

    def _context_boost(self, context: dict[str, Any]) -> float:
        """Domain-specific context signal. Override in subclasses."""
        # Check if context explicitly requests this expert
        requested = context.get("expert_hint", "")
        if requested == self.id:
            return 1.0
        # Check if previous expert in conversation matches our domain
        prev_expert = context.get("previous_expert", "")
        if prev_expert == self.id:
            return 0.5  # Continuity bonus
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT EXPERT DEFINITIONS — The 5-Expert Panel
# ═══════════════════════════════════════════════════════════════════════════════

EXPERT_R = Expert(
    id="pat_r",
    keywords=frozenset(
        {
            "how",
            "why",
            "explain",
            "reason",
            "plan",
            "analyze",
            "think",
            "compare",
            "evaluate",
            "strategy",
            "decompose",
            "because",
        }
    ),
    hhmm_states=frozenset({"reasoning", "planning", "analysis"}),
    base_weight=1.0,
)

EXPERT_K = Expert(
    id="pat_k",
    keywords=frozenset(
        {
            "what",
            "who",
            "when",
            "where",
            "define",
            "list",
            "describe",
            "tell",
            "facts",
            "history",
            "knowledge",
            "information",
        }
    ),
    hhmm_states=frozenset({"retrieval", "knowledge", "memory"}),
    base_weight=1.0,
)

EXPERT_S = Expert(
    id="pat_s",
    keywords=frozenset(
        {
            "code",
            "write",
            "create",
            "build",
            "implement",
            "fix",
            "run",
            "execute",
            "file",
            "tool",
            "script",
            "function",
            "deploy",
        }
    ),
    hhmm_states=frozenset({"skills", "coding", "execution"}),
    base_weight=1.0,
)

EXPERT_G = Expert(
    id="sat_g",
    keywords=frozenset(
        {
            "governance",
            "policy",
            "constitutional",
            "threshold",
            "ihsan",
            "compliance",
            "rule",
            "regulation",
            "authority",
            "permission",
        }
    ),
    hhmm_states=frozenset({"governance", "constitutional", "policy"}),
    base_weight=0.9,
)

EXPERT_V = Expert(
    id="sat_v",
    keywords=frozenset(
        {
            "verify",
            "prove",
            "check",
            "validate",
            "evidence",
            "audit",
            "test",
            "confirm",
            "certify",
            "assure",
            "proof",
        }
    ),
    hhmm_states=frozenset({"verification", "validation", "audit"}),
    base_weight=0.9,
)

DEFAULT_EXPERTS: tuple[Expert, ...] = (EXPERT_R, EXPERT_K, EXPERT_S, EXPERT_G, EXPERT_V)


# ═══════════════════════════════════════════════════════════════════════════════
# MOE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════


class MOEEngine:
    """5-Expert Mixture-of-Experts routing engine.

    Thread-safe. Stateless routing with tracked statistics.

    Usage::

        engine = MOEEngine()
        assignments = engine.route("How do I optimize my pipeline?")
        # → [ExpertAssignment("pat_r", 0.65), ExpertAssignment("pat_s", 0.35)]

        # After expert execution:
        results = [ExpertResult("pat_r", "...", ihsan=0.96, confidence=0.8), ...]
        synthesis = engine.synthesize(results, assignments)
    """

    def __init__(
        self,
        experts: Sequence[Expert] | None = None,
        top_k: int = MOE_TOP_K,
        min_confidence: float = MOE_MIN_CONFIDENCE,
        ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD,
        fallback_expert: str = MOE_FALLBACK_EXPERT,
        synthesis_strategy: str = MOE_SYNTHESIS_STRATEGY,
        hhmm_predictor: Callable[[str], str] | None = None,
    ) -> None:
        self._experts = tuple(experts) if experts else DEFAULT_EXPERTS
        self._expert_map = {e.id: e for e in self._experts}
        self._top_k = max(1, min(top_k, len(self._experts)))
        self._min_confidence = min_confidence
        self._ihsan_threshold = ihsan_threshold
        self._fallback_expert = fallback_expert
        self._synthesis_strategy = synthesis_strategy
        self._hhmm_predictor = hhmm_predictor
        self._stats = MOEEngineStats()
        self._lock = threading.Lock()

    @property
    def stats(self) -> MOEEngineStats:
        return self._stats

    @property
    def experts(self) -> tuple[Expert, ...]:
        return self._experts

    # ───────────────────────────────────────────────────────────────────────
    # ROUTING — Observe + Orient (Boyd OODA)
    # ───────────────────────────────────────────────────────────────────────

    def route(
        self,
        input_text: str,
        context: dict[str, Any] | None = None,
        top_k: int | None = None,
        expert_override: str | Sequence[str] | None = None,
    ) -> list[ExpertAssignment]:
        """Route input to top-K experts by relevance scoring.

        Args:
            input_text: The query text to route.
            context: Optional context dict (expert_hint, previous_expert, etc.).
            top_k: Override the default top-K selection count.
            expert_override: Force specific expert(s), bypassing scoring.

        Returns:
            List of ExpertAssignment with normalized weights summing to 1.0.

        Invariants:
            - len(result) >= 1
            - sum(weights) ≈ 1.0
            - all(0.0 <= w <= 1.0 for w in weights)
        """
        ctx = context or {}
        k = top_k if top_k is not None else self._top_k

        # Expert override: bypass routing entirely
        if expert_override:
            return self._handle_override(expert_override)

        # HHMM macro-state prediction (graceful degradation if unavailable)
        macro_state = "general"
        if self._hhmm_predictor is not None:
            try:
                macro_state = self._hhmm_predictor(input_text)
            except (RuntimeError, ValueError, TypeError) as e:
                logger.debug("HHMM predictor failed, using 'general': %s", e)

        # Score all experts
        scores: dict[str, float] = {}
        for expert in self._experts:
            scores[expert.id] = expert.score_relevance(input_text, macro_state, ctx)

        # Filter by minimum confidence
        viable = {eid: s for eid, s in scores.items() if s >= self._min_confidence}

        # Fallback: if no expert meets minimum, use fallback
        if not viable:
            with self._lock:
                self._stats.total_routes += 1
                self._stats._total_experts_activated += 1
                self._stats.expert_activations[self._fallback_expert] = (
                    self._stats.expert_activations.get(self._fallback_expert, 0) + 1
                )
                if self._stats.total_routes > 0:
                    self._stats.avg_experts_per_query = (
                        self._stats._total_experts_activated / self._stats.total_routes
                    )
            return [ExpertAssignment(self._fallback_expert, weight=1.0)]

        # Top-K selection
        k = min(k, len(viable))
        top_ids = sorted(viable, key=viable.get, reverse=True)[:k]  # type: ignore[arg-type]

        # Normalize weights
        total = sum(viable[eid] for eid in top_ids)
        if total == 0:
            return [ExpertAssignment(self._fallback_expert, weight=1.0)]

        assignments = [
            ExpertAssignment(eid, weight=viable[eid] / total) for eid in top_ids
        ]

        # Track stats (thread-safe)
        with self._lock:
            self._stats.total_routes += 1
            self._stats._total_experts_activated += len(assignments)
            for a in assignments:
                self._stats.expert_activations[a.expert_id] = (
                    self._stats.expert_activations.get(a.expert_id, 0) + 1
                )
            if self._stats.total_routes > 0:
                self._stats.avg_experts_per_query = (
                    self._stats._total_experts_activated / self._stats.total_routes
                )

        return assignments

    def _handle_override(self, override: str | Sequence[str]) -> list[ExpertAssignment]:
        """Handle expert_override — force specific expert(s)."""
        if isinstance(override, str):
            override = [override]

        valid = [eid for eid in override if eid in self._expert_map]
        if not valid:
            return [ExpertAssignment(self._fallback_expert, weight=1.0)]

        weight = 1.0 / len(valid)
        return [ExpertAssignment(eid, weight=weight) for eid in valid]

    # ───────────────────────────────────────────────────────────────────────
    # SYNTHESIS — Decide + Act (Boyd OODA)
    # ───────────────────────────────────────────────────────────────────────

    def synthesize(
        self,
        results: Sequence[ExpertResult],
        assignments: Sequence[ExpertAssignment],
    ) -> SynthesisResult:
        """Combine expert outputs with weighted synthesis + Ihsan gate.

        Args:
            results: Expert execution results (one per activated expert).
            assignments: The routing assignments that produced these results.

        Returns:
            SynthesisResult with combined text, weighted ihsan, and gate decision.
        """
        if not results:
            with self._lock:
                self._stats.gate_rejections += 1
            return SynthesisResult(
                text="",
                ihsan=0.0,
                passed_gate=False,
                reason="No expert results to synthesize",
            )

        # Build assignment lookup
        weight_map = {a.expert_id: a.weight for a in assignments}

        if self._synthesis_strategy == "best_of":
            return self._synthesize_best_of(results, weight_map)
        # Default: weighted combination
        return self._synthesize_weighted(results, weight_map)

    def _synthesize_weighted(
        self,
        results: Sequence[ExpertResult],
        weight_map: dict[str, float],
    ) -> SynthesisResult:
        """Weighted combination of expert outputs."""
        parts: list[str] = []
        combined_ihsan = 0.0
        total_weight = 0.0
        total_latency = 0.0

        for result in results:
            w = weight_map.get(result.expert_id, 1.0 / len(results))
            parts.append(f"[{result.expert_id}] {result.text}")
            combined_ihsan += result.ihsan * w
            total_weight += w
            total_latency += result.latency_ms

        # Normalize if weights don't sum to 1.0 (e.g., expert failures)
        if total_weight > 0 and abs(total_weight - 1.0) > 1e-6:
            combined_ihsan /= total_weight

        combined_text = "\n".join(parts)
        experts_used = tuple(r.expert_id for r in results)

        # Constitutional gate — Ihsan threshold
        passed = combined_ihsan >= self._ihsan_threshold
        reason = ""
        if not passed:
            reason = (
                f"Combined Ihsan {combined_ihsan:.3f} below "
                f"threshold {self._ihsan_threshold}"
            )
            with self._lock:
                self._stats.gate_rejections += 1

        return SynthesisResult(
            text=combined_text,
            ihsan=combined_ihsan,
            passed_gate=passed,
            reason=reason,
            experts_used=experts_used,
            total_latency_ms=total_latency,
        )

    def _synthesize_best_of(
        self,
        results: Sequence[ExpertResult],
        weight_map: dict[str, float],
    ) -> SynthesisResult:
        """Select the single best expert result."""
        best = max(results, key=lambda r: r.ihsan * r.confidence)
        passed = best.ihsan >= self._ihsan_threshold
        reason = ""
        if not passed:
            reason = (
                f"Best expert {best.expert_id} Ihsan {best.ihsan:.3f} "
                f"below threshold {self._ihsan_threshold}"
            )
            with self._lock:
                self._stats.gate_rejections += 1

        return SynthesisResult(
            text=best.text,
            ihsan=best.ihsan,
            passed_gate=passed,
            reason=reason,
            experts_used=(best.expert_id,),
            total_latency_ms=best.latency_ms,
        )

    # ───────────────────────────────────────────────────────────────────────
    # FULL PIPELINE — route → execute → synthesize
    # ───────────────────────────────────────────────────────────────────────

    def run(
        self,
        input_text: str,
        executor: Callable[[ExpertAssignment, str, dict[str, Any]], ExpertResult],
        context: dict[str, Any] | None = None,
        top_k: int | None = None,
        expert_override: str | Sequence[str] | None = None,
    ) -> SynthesisResult:
        """Full MOE pipeline: route → execute each expert → synthesize.

        Args:
            input_text: The query to process.
            executor: Callable that takes (assignment, input_text, context)
                     and returns an ExpertResult.
            context: Optional context dict.
            top_k: Override default top-K.
            expert_override: Force specific expert(s).

        Returns:
            SynthesisResult from the combined expert outputs.
        """
        ctx = context or {}
        t0 = time.monotonic()

        assignments = self.route(input_text, ctx, top_k, expert_override)

        results: list[ExpertResult] = []
        for assignment in assignments:
            try:
                result = executor(assignment, input_text, ctx)
                results.append(result)
            except Exception as e:
                logger.warning(
                    "Expert %s execution failed: %s", assignment.expert_id, e
                )

        synthesis = self.synthesize(results, assignments)

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.debug(
            "MOE pipeline: %d experts, ihsan=%.3f, gate=%s, %.1fms",
            len(results),
            synthesis.ihsan,
            synthesis.passed_gate,
            elapsed_ms,
        )

        return synthesis
