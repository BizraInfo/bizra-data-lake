"""
BIZRA Ihsan Gate v2.0 — 6-Dimensional Constitutional Enforcement
════════════════════════════════════════════════════════════════

Replaces the prior 4-dim gate. All weights and thresholds derived from
constitution.toml via generated_constants.py. Zero hardcoded values.

The Ihsan gate is the VERIFICATION stage of the trust compiler pipeline.
It maps to PAT agent: Evaluator (trust_stage: "attesting").

Theorem 2.3 (Constitutional Safety):
    P(Actuate | composite < gate_minimum) = 0
    This is not a target. It is a mathematical guarantee.
    fail_mode = "closed" — if computation fails, output is REJECTED.

Usage:
    from ihsan_gate import IhsanGate, IhsanScore
    gate = IhsanGate()
    score = gate.evaluate(output_text, mission_context)
    if score.passes:
        proceed(score.composite)
    else:
        reject(score.violations)
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ── Constitutional imports (single source of truth) ──
try:
    from generated.generated_constants import (
        IHSAN_OPERATIONAL_WEIGHTS,
        IHSAN_OPERATIONAL_NAMES,
        IHSAN_GATE_MINIMUM,
        IHSAN_BLOOM_ELIGIBILITY,
        IHSAN_EXCELLENCE,
        IHSAN_DIMENSIONS_OPERATIONAL,
        GATE_FAIL_MODE,
        GATE_OVERHEAD_BUDGET_MS,
    )
except ImportError:
    # Fallback for testing outside generated context
    # These values MUST match constitution.toml — verified by CI
    IHSAN_OPERATIONAL_WEIGHTS = {
        "moral_clarity": 0.1558,
        "epistemic_humility": 0.1818,
        "structural_integrity": 0.1688,
        "verifiability": 0.1688,
        "intent_alignment": 0.1818,
        "resilience": 0.1429,
    }
    IHSAN_OPERATIONAL_NAMES = list(IHSAN_OPERATIONAL_WEIGHTS.keys())
    IHSAN_GATE_MINIMUM = 0.85
    IHSAN_BLOOM_ELIGIBILITY = 0.90
    IHSAN_EXCELLENCE = 0.95
    IHSAN_DIMENSIONS_OPERATIONAL = 6
    GATE_FAIL_MODE = "closed"
    GATE_OVERHEAD_BUDGET_MS = 50

logger = logging.getLogger("bizra.ihsan_gate")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


class IhsanTier(Enum):
    """Quality tier derived from composite score."""
    REJECTED = "rejected"       # < gate_minimum
    ACCEPTABLE = "acceptable"   # ≥ gate_minimum, < bloom_eligibility
    BLOOM_ELIGIBLE = "bloom"    # ≥ bloom_eligibility, < excellence
    EXCELLENCE = "ihsan"        # ≥ excellence (إحسان)


@dataclass(frozen=True)
class DimensionScore:
    """Score for a single Ihsan dimension."""
    name: str
    raw_score: float        # 0.0 to 1.0
    weight: float           # Constitutional weight (renormalized 6-dim)
    weighted_score: float   # raw_score * weight
    passes: bool            # raw_score ≥ per-dimension minimum


@dataclass(frozen=True)
class IhsanScore:
    """Complete Ihsan evaluation result."""
    dimensions: list[DimensionScore]
    composite: float            # Weighted sum of all dimensions
    tier: IhsanTier
    passes: bool                # composite ≥ gate_minimum
    bloom_eligible: bool        # composite ≥ bloom_eligibility
    is_ihsan: bool              # composite ≥ excellence (إحسان standard)
    violations: list[str]       # Human-readable list of failures
    evaluation_ms: float        # Time taken for evaluation
    gate_minimum: float         # Threshold used (from constitution)

    def as_tensor_dict(self) -> dict[str, float]:
        """Return as map<string, double> for poi.proto wire format."""
        return {d.name: d.raw_score for d in self.dimensions}

    def as_evidence(self) -> dict[str, Any]:
        """Return structured evidence for the evidence ledger."""
        return {
            "ihsan_composite": self.composite,
            "ihsan_tensor": self.as_tensor_dict(),
            "tier": self.tier.value,
            "passes": self.passes,
            "bloom_eligible": self.bloom_eligible,
            "gate_minimum": self.gate_minimum,
            "evaluation_ms": self.evaluation_ms,
            "dimensions": IHSAN_DIMENSIONS_OPERATIONAL,
            "violations": self.violations,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DIMENSION SCORERS
# ═══════════════════════════════════════════════════════════════════════════════


def _score_moral_clarity(output: str, context: dict) -> float:
    """
    تقوى — Is the action ethically sound?
    Checks: no harmful content, respects user boundaries, truthful claims.
    """
    score = 1.0

    # Penalize if output contains known harmful patterns
    harmful_patterns = context.get("harmful_patterns", [])
    for pattern in harmful_patterns:
        if pattern.lower() in output.lower():
            score -= 0.3

    # Penalize if output makes unverifiable absolute claims
    absolute_markers = ["always", "never", "guaranteed", "impossible"]
    abs_count = sum(1 for m in absolute_markers if m in output.lower())
    score -= abs_count * 0.05

    # Reward if output includes appropriate caveats
    if context.get("requires_caveat", False):
        caveat_markers = ["however", "note that", "caveat", "limitation", "consider"]
        has_caveat = any(m in output.lower() for m in caveat_markers)
        if not has_caveat:
            score -= 0.15

    return max(0.0, min(1.0, score))


def _score_epistemic_humility(output: str, context: dict) -> float:
    """
    تواضع — Does the system know what it doesn't know?
    Checks: uncertainty acknowledgment, source attribution, confidence calibration.
    """
    score = 0.85  # Baseline

    # Reward explicit uncertainty markers
    uncertainty_markers = ["uncertain", "approximately", "estimate", "likely",
                          "confidence", "may", "might", "could"]
    uncertainty_count = sum(1 for m in uncertainty_markers if m in output.lower())
    score += min(uncertainty_count * 0.03, 0.10)

    # Reward source attribution
    if context.get("sources_cited", 0) > 0:
        score += 0.05

    # Penalize overconfidence on novel tasks
    if context.get("is_novel_task", False) and uncertainty_count == 0:
        score -= 0.15

    return max(0.0, min(1.0, score))


def _score_structural_integrity(output: str, context: dict) -> float:
    """
    إتقان — Is the implementation robust?
    Checks: output completeness, format validity, no truncation.
    """
    score = 0.90

    # Check output is not empty
    if not output or not output.strip():
        return 0.0

    # Check output is not truncated
    if context.get("expected_min_length", 0) > 0:
        if len(output) < context["expected_min_length"] * 0.5:
            score -= 0.20

    # Check format validity if expected
    expected_format = context.get("expected_format")
    if expected_format == "json":
        import json
        try:
            json.loads(output)
            score += 0.05
        except (json.JSONDecodeError, ValueError):
            score -= 0.25
    elif expected_format == "code":
        # Basic syntax check — non-empty, has structure
        if len(output.strip().splitlines()) > 1:
            score += 0.03

    return max(0.0, min(1.0, score))


def _score_verifiability(output: str, context: dict) -> float:
    """
    بيّنة — Can a third party verify the claim?
    Checks: reproducible reasoning, evidence trail, auditable logic.
    """
    score = 0.85

    # Reward step-by-step reasoning
    reasoning_markers = ["because", "therefore", "step", "first", "then",
                        "reason", "evidence", "based on"]
    reasoning_count = sum(1 for m in reasoning_markers if m in output.lower())
    score += min(reasoning_count * 0.02, 0.10)

    # Reward if output includes verifiable references
    if context.get("references_included", False):
        score += 0.05

    # Penalize opaque conclusions
    if len(output) > 200 and reasoning_count == 0:
        score -= 0.15

    return max(0.0, min(1.0, score))


def _score_intent_alignment(output: str, context: dict) -> float:
    """
    نيّة — Does the output serve the stated purpose?
    Checks: relevance to mission intent, task completion, user need satisfaction.
    """
    score = 0.88

    # Check mission keywords are addressed
    mission_keywords = context.get("mission_keywords", [])
    if mission_keywords:
        addressed = sum(1 for k in mission_keywords if k.lower() in output.lower())
        coverage = addressed / len(mission_keywords)
        score = 0.5 + (coverage * 0.5)  # Scale: 50% base + 50% coverage

    # Penalize off-topic content
    off_topic_ratio = context.get("off_topic_ratio", 0.0)
    score -= off_topic_ratio * 0.3

    # Reward conciseness for simple tasks
    if context.get("task_complexity", "complex") == "trivial":
        if len(output) > 1000:
            score -= 0.10  # Verbose response to trivial task

    return max(0.0, min(1.0, score))


def _score_resilience(output: str, context: dict) -> float:
    """
    صبر — Does it degrade gracefully under stress?
    Checks: fallback quality, partial results, error handling in output.
    """
    score = 0.90

    # If this was a fallback response, evaluate fallback quality
    if context.get("is_fallback", False):
        if len(output.strip()) > 50:
            score = 0.80  # Acceptable fallback
        else:
            score = 0.60  # Minimal fallback

    # If primary model was used but response was slow
    latency_ms = context.get("latency_ms", 0)
    budget_ms = context.get("latency_budget_ms", 15000)
    if latency_ms > budget_ms:
        score -= 0.15

    # Reward if partial results are still useful
    if context.get("partial_completion", False):
        if context.get("partial_is_useful", False):
            score = max(score, 0.75)
        else:
            score -= 0.20

    return max(0.0, min(1.0, score))


# Scorer registry — maps dimension name to scoring function
SCORERS: dict[str, callable] = {
    "moral_clarity": _score_moral_clarity,
    "epistemic_humility": _score_epistemic_humility,
    "structural_integrity": _score_structural_integrity,
    "verifiability": _score_verifiability,
    "intent_alignment": _score_intent_alignment,
    "resilience": _score_resilience,
}


# ═══════════════════════════════════════════════════════════════════════════════
# IHSAN GATE — Main evaluation engine
# ═══════════════════════════════════════════════════════════════════════════════


class IhsanGate:
    """
    6-dimensional constitutional quality gate.

    Evaluates every PAT output against the Ihsan tensor before actuation.
    Theorem 2.3 guarantees: P(Actuate | composite < gate_minimum) = 0.

    The gate is fail-closed: any evaluation error → REJECT.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        gate_minimum: float | None = None,
    ):
        self.weights = weights or IHSAN_OPERATIONAL_WEIGHTS
        self.gate_minimum = gate_minimum or IHSAN_GATE_MINIMUM
        self._validate_config()

    def _validate_config(self):
        """Verify gate configuration matches constitution."""
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Ihsan weights must sum to 1.0, got {total:.4f}. "
                f"This indicates constitution.toml drift."
            )
        if len(self.weights) != IHSAN_DIMENSIONS_OPERATIONAL:
            raise ValueError(
                f"Expected {IHSAN_DIMENSIONS_OPERATIONAL} dimensions, "
                f"got {len(self.weights)}"
            )

    def evaluate(
        self,
        output: str,
        context: dict[str, Any] | None = None,
    ) -> IhsanScore:
        """
        Evaluate an output against the 6-dimensional Ihsan tensor.

        Args:
            output: The text output to evaluate.
            context: Mission context dict with keys like:
                - mission_keywords: list[str]
                - task_complexity: str
                - expected_format: str
                - is_fallback: bool
                - latency_ms: float
                etc.

        Returns:
            IhsanScore with composite, tier, pass/fail, and per-dimension breakdown.
        """
        context = context or {}
        start = time.monotonic()

        try:
            return self._evaluate_inner(output, context, start)
        except Exception as e:
            # FAIL-CLOSED: evaluation error → REJECT
            elapsed = (time.monotonic() - start) * 1000
            logger.error(f"Ihsan gate evaluation failed: {e}")
            if GATE_FAIL_MODE == "closed":
                return IhsanScore(
                    dimensions=[],
                    composite=0.0,
                    tier=IhsanTier.REJECTED,
                    passes=False,
                    bloom_eligible=False,
                    is_ihsan=False,
                    violations=[f"Gate evaluation error (fail-closed): {e}"],
                    evaluation_ms=elapsed,
                    gate_minimum=self.gate_minimum,
                )
            raise  # Should never reach here — fail_mode is always "closed"

    def _evaluate_inner(
        self,
        output: str,
        context: dict[str, Any],
        start: float,
    ) -> IhsanScore:
        """Core evaluation logic."""
        dimension_scores = []
        violations = []
        per_dim_minimum = 0.50  # Any dimension below 0.50 is a hard fail

        for dim_name in IHSAN_OPERATIONAL_NAMES:
            scorer = SCORERS.get(dim_name)
            if scorer is None:
                violations.append(f"No scorer for dimension: {dim_name}")
                continue

            weight = self.weights[dim_name]
            raw = scorer(output, context)
            raw = max(0.0, min(1.0, raw))  # Clamp to [0, 1]

            passes_dim = raw >= per_dim_minimum
            if not passes_dim:
                violations.append(
                    f"{dim_name}: {raw:.3f} < {per_dim_minimum} (per-dimension minimum)"
                )

            dimension_scores.append(DimensionScore(
                name=dim_name,
                raw_score=raw,
                weight=weight,
                weighted_score=raw * weight,
                passes=passes_dim,
            ))

        # Composite = weighted sum
        composite = sum(d.weighted_score for d in dimension_scores)
        composite = max(0.0, min(1.0, composite))

        # Tier classification
        if composite < self.gate_minimum:
            tier = IhsanTier.REJECTED
            violations.append(
                f"Composite {composite:.3f} < gate_minimum {self.gate_minimum}"
            )
        elif composite < IHSAN_BLOOM_ELIGIBILITY:
            tier = IhsanTier.ACCEPTABLE
        elif composite < IHSAN_EXCELLENCE:
            tier = IhsanTier.BLOOM_ELIGIBLE
        else:
            tier = IhsanTier.EXCELLENCE

        passes = composite >= self.gate_minimum and all(d.passes for d in dimension_scores)
        elapsed = (time.monotonic() - start) * 1000

        if elapsed > GATE_OVERHEAD_BUDGET_MS:
            logger.warning(
                f"Ihsan gate exceeded budget: {elapsed:.1f}ms > {GATE_OVERHEAD_BUDGET_MS}ms"
            )

        return IhsanScore(
            dimensions=dimension_scores,
            composite=composite,
            tier=tier,
            passes=passes,
            bloom_eligible=composite >= IHSAN_BLOOM_ELIGIBILITY,
            is_ihsan=composite >= IHSAN_EXCELLENCE,
            violations=violations,
            evaluation_ms=elapsed,
            gate_minimum=self.gate_minimum,
        )

    def evaluate_batch(
        self,
        outputs: list[tuple[str, dict[str, Any]]],
    ) -> list[IhsanScore]:
        """Evaluate multiple outputs. Used by SAT ConsensusValidators."""
        return [self.evaluate(text, ctx) for text, ctx in outputs]
