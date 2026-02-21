"""
Entropy Router -- System 1/2 Query Routing by Complexity
=========================================================

Routes queries to the appropriate processing tier based on Shannon
entropy, structural complexity signals, and domain heuristics.

System 1 (Reflexive): Low-entropy queries -> fast path, no consensus
System 2 (Deliberative): High-entropy queries -> GoT + SAT consensus

Standing on Giants:
- Kahneman (2011): System 1/2 dual-process theory
- Shannon (1948): Information entropy as complexity measure
- Boyd (1976): OODA loop -- observe complexity before committing resources
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional

from core.integration.constants import (
    SNR_THRESHOLD_T0_ELITE,
    SNR_THRESHOLD_T1_HIGH,
    UNIFIED_AGENT_TIMEOUT_MS,
    UNIFIED_SNR_THRESHOLD,
)


class QueryComplexity(Enum):
    """Five-tier complexity taxonomy aligned with MoE router."""

    TRIVIAL = auto()  # Simple factual, yes/no -- System 1
    SIMPLE = auto()  # Basic QA, formatting -- System 1
    MODERATE = auto()  # Multi-step reasoning -- System 1.5
    COMPLEX = auto()  # Deep reasoning, planning -- System 2
    FRONTIER = auto()  # Research-level, novel -- System 2+


@dataclass(frozen=True)
class RoutingDecision:
    """Output of the Entropy Router -- determines processing path."""

    query_complexity: QueryComplexity
    system: str  # "S1_REFLEXIVE" | "S1_5_MODERATE" | "S2_DELIBERATIVE"
    max_latency_ms: int  # Upper bound for this tier
    quorum_size: int  # SAT consensus quorum (0 = no consensus)
    snr_requirement: float  # Minimum SNR for this tier
    use_got: bool  # Whether to engage Graph-of-Thoughts
    use_orchestrator: bool  # Whether to use full orchestrator decomposition
    reasoning: str = ""  # Human-readable routing explanation


# Complexity thresholds (Shannon entropy of query text, normalized 0-1)
_TRIVIAL_CEILING = 0.30
_SIMPLE_CEILING = 0.50
_MODERATE_CEILING = 0.70
_COMPLEX_CEILING = 0.85
# Above 0.85 = FRONTIER

# Structural complexity signals (regex patterns)
_SUB_QUESTION_PATTERNS = [
    r"\b(compare|contrast|analyze|evaluate|synthesize)\b",
    r"\b(how does .+ relate to)\b",
    r"\b(what are the implications)\b",
    r"\b(step[- ]by[- ]step)\b",
    r"\b(pros? and cons?)\b",
    r"\b(trade[- ]?offs?)\b",
]

_MULTI_DOMAIN_PATTERNS = [
    r"\b(and also|additionally|furthermore|moreover)\b",
    r"\b(from .+ perspective)\b",
    r"\b(considering .+ and .+)\b",
]


class EntropyRouter:
    """Routes queries to System 1 or System 2 based on entropy analysis."""

    def __init__(self) -> None:
        self._sub_q_patterns = [
            re.compile(p, re.IGNORECASE) for p in _SUB_QUESTION_PATTERNS
        ]
        self._domain_patterns = [
            re.compile(p, re.IGNORECASE) for p in _MULTI_DOMAIN_PATTERNS
        ]

    def route(
        self, query_text: str, context: Optional[dict[str, Any]] = None
    ) -> RoutingDecision:
        """Classify query complexity and return routing decision."""
        context = context or {}
        complexity_score = self.estimate_complexity(query_text, context)
        complexity = self._score_to_tier(complexity_score)
        return self._build_decision(complexity, complexity_score, query_text)

    def estimate_complexity(
        self, query_text: str, context: Optional[dict[str, Any]] = None
    ) -> float:
        """Estimate query complexity on 0.0-1.0 scale using entropy + heuristics."""
        context = context or {}

        # 1. Shannon entropy of character distribution (normalized)
        entropy = self._text_entropy(query_text)

        # 2. Length signal (longer queries tend to be more complex)
        words = query_text.split()
        length_score = min(len(words) / 80.0, 1.0)

        # 3. Sub-question structural markers
        sub_q_count = sum(1 for p in self._sub_q_patterns if p.search(query_text))
        sub_q_score = min(sub_q_count / 3.0, 1.0)

        # 4. Multi-domain markers
        domain_count = sum(1 for p in self._domain_patterns if p.search(query_text))
        domain_score = min(domain_count / 2.0, 1.0)

        # 5. Question mark density (multiple questions = more complex)
        q_marks = query_text.count("?")
        q_score = min(q_marks / 3.0, 1.0)

        # 6. Explicit complexity hint from context
        hint = float(context.get("complexity_hint", 0.0))

        # Weighted combination
        score = (
            0.25 * entropy
            + 0.15 * length_score
            + 0.20 * sub_q_score
            + 0.15 * domain_score
            + 0.10 * q_score
            + 0.15 * hint
        )
        return min(max(score, 0.0), 1.0)

    @staticmethod
    def _text_entropy(text: str) -> float:
        """Shannon entropy of character distribution, normalized to 0-1."""
        if not text:
            return 0.0
        freq: dict[str, int] = {}
        for ch in text.lower():
            freq[ch] = freq.get(ch, 0) + 1
        n = len(text)
        entropy = -sum((c / n) * math.log2(c / n) for c in freq.values() if c > 0)
        # Normalize by log2(alphabet_size) to get 0-1 range
        max_entropy = math.log2(len(freq)) if len(freq) > 1 else 1.0
        return entropy / max_entropy if max_entropy > 0 else 0.0

    @staticmethod
    def _score_to_tier(score: float) -> QueryComplexity:
        """Map continuous score to discrete complexity tier."""
        if score < _TRIVIAL_CEILING:
            return QueryComplexity.TRIVIAL
        elif score < _SIMPLE_CEILING:
            return QueryComplexity.SIMPLE
        elif score < _MODERATE_CEILING:
            return QueryComplexity.MODERATE
        elif score < _COMPLEX_CEILING:
            return QueryComplexity.COMPLEX
        else:
            return QueryComplexity.FRONTIER

    @staticmethod
    def _build_decision(
        complexity: QueryComplexity, score: float, query: str
    ) -> RoutingDecision:
        """Build routing decision from classified complexity."""
        tier_config = {
            QueryComplexity.TRIVIAL: {
                "system": "S1_REFLEXIVE",
                "max_latency_ms": 200,
                "quorum_size": 0,
                "snr_requirement": UNIFIED_SNR_THRESHOLD,
                "use_got": False,
                "use_orchestrator": False,
            },
            QueryComplexity.SIMPLE: {
                "system": "S1_REFLEXIVE",
                "max_latency_ms": 500,
                "quorum_size": 0,
                "snr_requirement": UNIFIED_SNR_THRESHOLD,
                "use_got": False,
                "use_orchestrator": False,
            },
            QueryComplexity.MODERATE: {
                "system": "S1_5_MODERATE",
                "max_latency_ms": 5000,
                "quorum_size": 3,
                "snr_requirement": SNR_THRESHOLD_T1_HIGH,
                "use_got": True,
                "use_orchestrator": False,
            },
            QueryComplexity.COMPLEX: {
                "system": "S2_DELIBERATIVE",
                "max_latency_ms": UNIFIED_AGENT_TIMEOUT_MS,
                "quorum_size": 5,
                "snr_requirement": SNR_THRESHOLD_T1_HIGH,
                "use_got": True,
                "use_orchestrator": True,
            },
            QueryComplexity.FRONTIER: {
                "system": "S2_DELIBERATIVE",
                "max_latency_ms": UNIFIED_AGENT_TIMEOUT_MS,
                "quorum_size": 33,  # Full SAT-49 quorum (2f+1)
                "snr_requirement": SNR_THRESHOLD_T0_ELITE,
                "use_got": True,
                "use_orchestrator": True,
            },
        }
        config = tier_config[complexity]
        short_query = query[:60] + "..." if len(query) > 60 else query
        return RoutingDecision(
            query_complexity=complexity,
            reasoning=f"score={score:.3f} -> {complexity.name} for '{short_query}'",
            **config,
        )
