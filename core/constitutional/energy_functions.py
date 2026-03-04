"""
Thermodynamic Ihsan energy functions.

Maps content/context to canonical Ihsan dimensions through an energy model:
    i_j = exp(-E_j / T)

This module is deterministic and bounded to keep runtime behavior stable
on minimum hardware profiles.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

from core.integration.constants import IHSAN_CANONICAL_WEIGHTS

_WORD_RE = re.compile(r"[a-zA-Z0-9_']+")
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_NUMERIC_RE = re.compile(r"\b\d+(?:[./-]\d+)?\b")
_BULLET_RE = re.compile(r"(?m)^\s*(?:[-*]|\d+[.)])\s+")


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    cleaned = {
        key: max(0.0, float(value))
        for key, value in weights.items()
        if key in IHSAN_CANONICAL_WEIGHTS
    }
    total = sum(cleaned.values())
    if total <= 0.0:
        n = len(IHSAN_CANONICAL_WEIGHTS)
        return {key: 1.0 / n for key in IHSAN_CANONICAL_WEIGHTS}
    return {key: cleaned.get(key, 0.0) / total for key in IHSAN_CANONICAL_WEIGHTS}


@dataclass(frozen=True)
class EnergySignals:
    word_count: int
    unique_ratio: float
    unsafe_hits: int
    evidence_hits: int
    actionable_hits: int
    hedge_hits: int
    query_alignment: float
    context_risk: float


@dataclass(frozen=True)
class EnergyProfile:
    """Computed thermodynamic profile for one content evaluation."""

    energies: dict[str, float]
    ihsan_dimensions: dict[str, float]
    composite_ihsan: float
    total_energy: float
    temperature: float


class ThermodynamicEnergySuite:
    """
    Deterministic energy model for the 8 canonical Ihsan dimensions.

    The score projection uses:
        ihsan_dim = exp(-energy / temperature)
    """

    def __init__(
        self,
        *,
        weights: Mapping[str, float] | None = None,
        t0: float = 1.0,
        min_temperature: float = 0.05,
    ) -> None:
        self.weights = _normalize_weights(weights or IHSAN_CANONICAL_WEIGHTS)
        self.t0 = float(t0)
        self.min_temperature = float(min_temperature)

    def temperature(self, step: int | float) -> float:
        """Inverse-linear cooling schedule with a small floor."""
        step_val = max(float(step), 0.0)
        t = self.t0 / (1.0 + step_val)
        return max(self.min_temperature, t)

    def compute(
        self,
        content: str,
        *,
        snr_score: float | None = None,
        query_text: str = "",
        context: Mapping[str, Any] | None = None,
        step: int | float = 0,
    ) -> EnergyProfile:
        """Compute canonical energies and thermodynamic Ihsan dimensions."""
        signals = self._extract_signals(content, query_text, context)
        snr = _clamp01(0.5 if snr_score is None else float(snr_score))
        temperature = self.temperature(step)
        energies = self._component_energies(signals, snr)
        ihsan_dims = {
            name: math.exp(-(energy / max(temperature, 1e-6)))
            for name, energy in energies.items()
        }
        total_energy = sum(self.weights[name] * energies[name] for name in self.weights)
        composite = sum(self.weights[name] * ihsan_dims[name] for name in self.weights)

        return EnergyProfile(
            energies=energies,
            ihsan_dimensions=ihsan_dims,
            composite_ihsan=_clamp01(composite),
            total_energy=max(0.0, total_energy),
            temperature=temperature,
        )

    @staticmethod
    def _extract_signals(
        content: str,
        query_text: str,
        context: Mapping[str, Any] | None,
    ) -> EnergySignals:
        text = content or ""
        text_lower = text.lower()
        words = _WORD_RE.findall(text_lower)
        word_count = len(words)
        unique_ratio = len(set(words)) / max(word_count, 1)

        unsafe_hits = sum(
            1
            for term in ("kill", "bomb", "weapon", "exploit", "malware", "harm")
            if re.search(rf"\b{re.escape(term)}\b", text_lower)
        )
        actionable_hits = sum(
            1
            for term in (
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
            if re.search(rf"\b{re.escape(term)}\b", text_lower)
        )
        hedge_hits = sum(
            text_lower.count(phrase)
            for phrase in ("maybe", "might", "possibly", "i think", "not sure")
        )

        evidence_hits = 0
        if _URL_RE.search(text):
            evidence_hits += 1
        if _NUMERIC_RE.search(text):
            evidence_hits += 1
        if _BULLET_RE.search(text):
            evidence_hits += 1
        if "```" in text:
            evidence_hits += 1
        if any(
            marker in text_lower
            for marker in ("because", "therefore", "for example", "evidence", "metric")
        ):
            evidence_hits += 1

        query_tokens = set(_WORD_RE.findall((query_text or "").lower()))
        query_alignment = (
            len(query_tokens & set(words)) / max(len(query_tokens), 1)
            if query_tokens
            else 0.0
        )

        context_risk = 0.0
        if context:
            raw = context.get("risk_score", 0.0)
            try:
                context_risk = float(raw)
            except (TypeError, ValueError):
                context_risk = 0.0

        return EnergySignals(
            word_count=word_count,
            unique_ratio=_clamp01(unique_ratio),
            unsafe_hits=unsafe_hits,
            evidence_hits=evidence_hits,
            actionable_hits=actionable_hits,
            hedge_hits=hedge_hits,
            query_alignment=_clamp01(query_alignment),
            context_risk=_clamp01(context_risk),
        )

    @staticmethod
    def _component_energies(signals: EnergySignals, snr: float) -> dict[str, float]:
        actionable = _clamp01(signals.actionable_hits / 5.0)
        evidence = _clamp01(signals.evidence_hits / 5.0)
        unsafe_ratio = _clamp01(signals.unsafe_hits / 4.0)
        confidence = snr

        moral_clarity = (
            0.02
            + 0.65 * unsafe_ratio
            + 0.20 * signals.context_risk
            + 0.05 * _clamp01(signals.hedge_hits / 4.0)
        )

        expected_accuracy = _clamp01(
            0.5 * snr + 0.3 * evidence + 0.2 * signals.query_alignment
        )
        overconfidence = max(0.0, confidence - expected_accuracy)
        epistemic_humility = ((confidence - expected_accuracy) ** 2) + (
            0.15 * overconfidence
        )

        structural_integrity = 1.0 - _clamp01(
            0.45 * actionable
            + 0.35 * signals.query_alignment
            + 0.20 * signals.unique_ratio
        )

        verifiability = 1.0 - _clamp01(0.45 * evidence + 0.30 * snr + 0.25 * actionable)
        contextual_relevance = 1.0 - signals.query_alignment
        intent_alignment = 1.0 - _clamp01(
            0.70 * signals.query_alignment + 0.30 * actionable
        )
        resilience = _clamp01(
            0.55 * signals.context_risk
            + 0.30 * unsafe_ratio
            + 0.15 * _clamp01(signals.hedge_hits / 5.0)
        )

        over_length = max(0, signals.word_count - 300) / 700.0
        under_length = 0.25 if signals.word_count < 8 else 0.0
        efficiency = _clamp01(
            0.45 * over_length
            + 0.25 * (1.0 - signals.unique_ratio)
            + 0.20 * (1.0 - actionable)
            + 0.10 * under_length
        )

        return {
            "moral_clarity": _clamp01(moral_clarity),
            "epistemic_humility": _clamp01(epistemic_humility),
            "structural_integrity": _clamp01(structural_integrity),
            "verifiability": _clamp01(verifiability),
            "contextual_relevance": _clamp01(contextual_relevance),
            "intent_alignment": _clamp01(intent_alignment),
            "resilience": _clamp01(resilience),
            "efficiency": _clamp01(efficiency),
        }


__all__ = [
    "EnergyProfile",
    "EnergySignals",
    "ThermodynamicEnergySuite",
]
