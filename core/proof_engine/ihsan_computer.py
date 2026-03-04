"""
Ihsan Computer — Content-grounded Ihsan component estimation.

Bridges raw inference text to the authoritative 4-component Ihsan gate:
correctness, safety, efficiency, user_benefit.

Design goals:
- Deterministic: same inputs produce same component scores
- Bounded: every component is clamped to [0.0, 1.0]
- Explainable: simple lexical/structural signals, no hidden model dependency
- Fail-safe: missing inputs degrade gracefully, never throw on normal text

Standing on Giants:
- Shannon (1948): measurable signal quality
- Lamport (1978): deterministic reproducibility
- Constitutional AI (Anthropic): explicit, auditable constraints
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from core.constitutional.energy_functions import ThermodynamicEnergySuite
from core.proof_engine.ihsan_gate import IhsanComponents

_WORD_RE = re.compile(r"[a-zA-Z0-9_']+")
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_NUMERIC_RE = re.compile(r"\b\d+(?:[./-]\d+)?\b")
_BULLET_RE = re.compile(r"(?m)^\s*(?:[-*]|\d+[.)])\s+")


def _clamp01(value: float) -> float:
    """Clamp a floating-point score to [0.0, 1.0]."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


@dataclass(frozen=True)
class IhsanSignals:
    """Extracted lexical/structural signals used for Ihsan component scoring."""

    word_count: int
    unique_ratio: float
    unsafe_hits: int
    evidence_hits: int
    actionable_hits: int
    hedge_hits: int
    query_alignment: float
    context_risk: float


class IhsanComputer:
    """
    Compute Ihsan components from response content and context.

    This class intentionally uses low-complexity, deterministic heuristics so it
    can run in every environment (including offline and test mode) while still
    grounding gate decisions in content rather than constant defaults.
    """

    _UNSAFE_TERMS = (
        "kill",
        "bomb",
        "weapon",
        "exploit",
        "malware",
        "ransomware",
        "phishing",
        "harm",
    )
    _ACTIONABLE_TERMS = (
        "step",
        "steps",
        "first",
        "next",
        "then",
        "implement",
        "verify",
        "measure",
        "test",
        "monitor",
    )
    _HEDGE_PHRASES = (
        "maybe",
        "might",
        "possibly",
        "i think",
        "not sure",
        "guess",
    )
    _EVIDENCE_MARKERS = (
        "because",
        "therefore",
        "for example",
        "evidence",
        "benchmark",
        "metric",
    )

    def __init__(
        self,
        *,
        enable_thermal_mode: Optional[bool] = None,
        thermal_suite: Optional[ThermodynamicEnergySuite] = None,
    ) -> None:
        if enable_thermal_mode is None:
            raw = os.getenv("BIZRA_ENABLE_THERMODYNAMIC_IHSAN", "0").lower()
            self.enable_thermal_mode = raw in {"1", "true", "yes", "on"}
        else:
            self.enable_thermal_mode = bool(enable_thermal_mode)
        self.thermal_suite = thermal_suite or ThermodynamicEnergySuite()

    def compute(
        self,
        content: str,
        *,
        snr_score: Optional[float] = None,
        query_text: str = "",
        context: Optional[dict[str, Any]] = None,
    ) -> IhsanComponents:
        """Compute authoritative gate components from content signals."""
        components, _ = self.compute_with_signals(
            content=content,
            snr_score=snr_score,
            query_text=query_text,
            context=context,
        )
        return components

    def compute_with_signals(
        self,
        content: str,
        *,
        snr_score: Optional[float] = None,
        query_text: str = "",
        context: Optional[dict[str, Any]] = None,
    ) -> tuple[IhsanComponents, IhsanSignals]:
        """
        Compute Ihsan components and return intermediate signals for diagnostics.
        """
        signals = self._extract_signals(
            content=content,
            query_text=query_text,
            context=context,
        )
        if self._thermal_mode_enabled(context):
            thermal_components = self._compute_via_thermodynamic_mapping(
                content=content,
                snr_score=snr_score,
                query_text=query_text,
                context=context,
            )
            return thermal_components, signals

        snr = _clamp01(0.5 if snr_score is None else float(snr_score))

        correctness = self._score_correctness(signals, snr)
        safety = self._score_safety(signals)
        efficiency = self._score_efficiency(signals)
        user_benefit = self._score_user_benefit(signals)

        return (
            IhsanComponents(
                correctness=correctness,
                safety=safety,
                efficiency=efficiency,
                user_benefit=user_benefit,
            ),
            signals,
        )

    def _thermal_mode_enabled(self, context: Optional[Mapping[str, Any]]) -> bool:
        if context:
            raw_mode = context.get("ihsan_mode")
            if isinstance(raw_mode, str) and raw_mode.lower() == "thermal":
                return True
            raw_enabled = context.get("enable_thermodynamic_ihsan")
            if isinstance(raw_enabled, bool):
                return raw_enabled
            if isinstance(raw_enabled, str):
                return raw_enabled.lower() in {"1", "true", "yes", "on"}
        return self.enable_thermal_mode

    def _compute_via_thermodynamic_mapping(
        self,
        *,
        content: str,
        snr_score: Optional[float],
        query_text: str,
        context: Optional[Mapping[str, Any]],
    ) -> IhsanComponents:
        step_raw = 0
        if context:
            step_raw = (
                context.get("thermal_step")
                or context.get("anneal_step")
                or context.get("step")
                or 0
            )
        try:
            step = int(step_raw)
        except (TypeError, ValueError):
            step = 0

        profile = self.thermal_suite.compute(
            content=content,
            snr_score=snr_score,
            query_text=query_text,
            context=context,
            step=step,
        )
        dims = profile.ihsan_dimensions

        correctness = _clamp01(
            (
                dims["moral_clarity"]
                + dims["epistemic_humility"]
                + dims["structural_integrity"]
            )
            / 3.0
        )
        safety = _clamp01((dims["moral_clarity"] + dims["resilience"]) / 2.0)
        user_benefit = _clamp01(
            (dims["contextual_relevance"] + dims["intent_alignment"]) / 2.0
        )

        return IhsanComponents(
            correctness=correctness,
            safety=safety,
            efficiency=dims["efficiency"],
            user_benefit=user_benefit,
            auditability=dims["verifiability"],
            robustness=dims["resilience"],
        )

    def _extract_signals(
        self,
        *,
        content: str,
        query_text: str,
        context: Optional[dict[str, Any]],
    ) -> IhsanSignals:
        """Extract deterministic text/context signals used by component scorers."""
        text = content or ""
        text_lower = text.lower()
        words = _WORD_RE.findall(text_lower)
        word_count = len(words)
        unique_ratio = len(set(words)) / max(word_count, 1)

        unsafe_hits = self._count_term_hits(text_lower, self._UNSAFE_TERMS)
        actionable_hits = self._count_term_hits(text_lower, self._ACTIONABLE_TERMS)
        hedge_hits = sum(text_lower.count(phrase) for phrase in self._HEDGE_PHRASES)

        evidence_hits = 0
        if _URL_RE.search(text):
            evidence_hits += 1
        if _NUMERIC_RE.search(text):
            evidence_hits += 1
        if _BULLET_RE.search(text):
            evidence_hits += 1
        if "```" in text:
            evidence_hits += 1
        if any(marker in text_lower for marker in self._EVIDENCE_MARKERS):
            evidence_hits += 1

        query_tokens = set(_WORD_RE.findall((query_text or "").lower()))
        response_tokens = set(words)
        query_alignment = (
            len(query_tokens & response_tokens) / max(len(query_tokens), 1)
            if query_tokens
            else 0.0
        )

        context_risk = 0.0
        if context:
            try:
                context_risk = float(context.get("risk_score", 0.0))
            except (TypeError, ValueError):
                context_risk = 0.0
        context_risk = _clamp01(context_risk)

        return IhsanSignals(
            word_count=word_count,
            unique_ratio=_clamp01(unique_ratio),
            unsafe_hits=unsafe_hits,
            evidence_hits=evidence_hits,
            actionable_hits=actionable_hits,
            hedge_hits=hedge_hits,
            query_alignment=_clamp01(query_alignment),
            context_risk=context_risk,
        )

    @staticmethod
    def _count_term_hits(text_lower: str, terms: tuple[str, ...]) -> int:
        """Count the number of distinct term matches with word boundaries."""
        return sum(
            1 for term in terms if re.search(rf"\b{re.escape(term)}\b", text_lower)
        )

    @staticmethod
    def _score_correctness(signals: IhsanSignals, snr_score: float) -> float:
        """
        Correctness favors strong SNR plus concrete evidence markers.

        Penalizes hedging language because uncertainty without verification
        weakens factual confidence.
        """
        base = 0.40 + 0.45 * snr_score
        evidence_bonus = min(0.12, 0.03 * signals.evidence_hits)
        hedge_penalty = min(0.15, 0.03 * signals.hedge_hits)
        length_bonus = 0.03 if signals.word_count >= 40 else 0.0
        return _clamp01(base + evidence_bonus + length_bonus - hedge_penalty)

    @staticmethod
    def _score_safety(signals: IhsanSignals) -> float:
        """
        Safety starts high and degrades with harmful cues and explicit risk context.
        """
        base = 0.98
        unsafe_penalty = min(0.75, 0.18 * signals.unsafe_hits)
        risk_penalty = 0.12 * signals.context_risk
        return _clamp01(base - unsafe_penalty - risk_penalty)

    @staticmethod
    def _score_efficiency(signals: IhsanSignals) -> float:
        """
        Efficiency rewards lexical diversity and structure, penalizes over-length.
        """
        if signals.word_count <= 0:
            return 0.0

        base = 0.25 + 0.65 * signals.unique_ratio
        structure_bonus = 0.05 if signals.evidence_hits > 0 else 0.0

        length_penalty = 0.0
        if signals.word_count > 450:
            length_penalty = min(0.25, (signals.word_count - 450) / 1200.0)

        return _clamp01(base + structure_bonus - length_penalty)

    @staticmethod
    def _score_user_benefit(signals: IhsanSignals) -> float:
        """
        User benefit tracks actionable structure, query alignment, and clarity.
        """
        base = 0.20
        alignment_component = 0.35 * signals.query_alignment
        action_component = min(0.25, 0.04 * signals.actionable_hits)
        evidence_component = min(0.12, 0.03 * signals.evidence_hits)
        hedge_penalty = min(0.12, 0.02 * signals.hedge_hits)
        brevity_penalty = 0.20 if signals.word_count < 8 else 0.0

        return _clamp01(
            base
            + alignment_component
            + action_component
            + evidence_component
            - hedge_penalty
            - brevity_penalty
        )


__all__ = ["IhsanComputer", "IhsanSignals"]
