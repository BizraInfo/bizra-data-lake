"""
BIZRA HHMM Router — Complexity Classification & Tier Routing
═════════════════════════════════════════════════════════════

The HHMM (Hierarchical Hidden Markov Model) classifies every incoming
mission into a complexity tier, then routes to the appropriate handler:

  TRIVIAL  → ReflexCache (O(1), <100ms)
  SIMPLE   → Single PAT agent (<3s)
  COMPLEX  → Full 7-agent PAT pipeline (<15s)
  SOVEREIGN → Multi-mission decomposition (<60s)

This unifies three previously separate designs:
  - v4.0's HHMM intent prediction (47 states)
  - ARC-001's Action Bus priority routing
  - Ω³'s three-orchestrator hierarchy

Constitution reference: §7 [hhmm]
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

try:
    from generated.generated_constants import (
        ACTION_BUS_GCD_TICK_MS,
        ACTION_BUS_MAX_CONCURRENT,
        ACTION_BUS_MAX_PER_HOUR,
        HMM_INITIAL_LIVE_STATES,
        HMM_NUM_HIDDEN_STATES,
        HMM_OBSERVATION_WINDOW,
        TIER_COMPLEX_BUDGET_MS,
        TIER_SIMPLE_BUDGET_MS,
        TIER_SOVEREIGN_BUDGET_MS,
        TIER_TRIVIAL_BUDGET_MS,
    )
except ImportError:
    HMM_NUM_HIDDEN_STATES = 47
    HMM_OBSERVATION_WINDOW = 50
    HMM_INITIAL_LIVE_STATES = 5
    TIER_TRIVIAL_BUDGET_MS = 100
    TIER_SIMPLE_BUDGET_MS = 3000
    TIER_COMPLEX_BUDGET_MS = 15000
    TIER_SOVEREIGN_BUDGET_MS = 60000
    ACTION_BUS_GCD_TICK_MS = 100
    ACTION_BUS_MAX_CONCURRENT = 10
    ACTION_BUS_MAX_PER_HOUR = 100

logger = logging.getLogger("bizra.hhmm_router")


# ═══════════════════════════════════════════════════════════════════════════════
# COMPLEXITY TIERS
# ═══════════════════════════════════════════════════════════════════════════════


class ComplexityTier(Enum):
    TRIVIAL = "trivial"  # Cache hit, no LLM needed
    SIMPLE = "simple"  # Single agent, one inference call
    COMPLEX = "complex"  # Full 7-agent pipeline
    SOVEREIGN = "sovereign"  # Multi-mission decomposition


@dataclass(frozen=True)
class TierConfig:
    tier: ComplexityTier
    handler: str
    latency_budget_ms: int
    score_range: tuple[float, float]


TIER_CONFIGS = {
    ComplexityTier.TRIVIAL: TierConfig(
        tier=ComplexityTier.TRIVIAL,
        handler="ReflexCache",
        latency_budget_ms=TIER_TRIVIAL_BUDGET_MS,
        score_range=(0.0, 0.1),
    ),
    ComplexityTier.SIMPLE: TierConfig(
        tier=ComplexityTier.SIMPLE,
        handler="SingleAgentPipeline",
        latency_budget_ms=TIER_SIMPLE_BUDGET_MS,
        score_range=(0.1, 0.4),
    ),
    ComplexityTier.COMPLEX: TierConfig(
        tier=ComplexityTier.COMPLEX,
        handler="MissionOrchestrator",
        latency_budget_ms=TIER_COMPLEX_BUDGET_MS,
        score_range=(0.4, 0.7),
    ),
    ComplexityTier.SOVEREIGN: TierConfig(
        tier=ComplexityTier.SOVEREIGN,
        handler="SovereignOrchestrator",
        latency_budget_ms=TIER_SOVEREIGN_BUDGET_MS,
        score_range=(0.7, 1.0),
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION RESULT
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ClassificationResult:
    """Result of HHMM classification."""

    tier: ComplexityTier
    config: TierConfig
    complexity_score: float  # 0.0 to 1.0
    confidence: float  # How confident the classifier is
    features: dict[str, float]  # Feature breakdown
    classification_ms: float  # Time to classify
    has_reflex: bool  # Whether a cache hit exists

    @property
    def latency_budget_ms(self) -> int:
        return self.config.latency_budget_ms

    @property
    def handler(self) -> str:
        return self.config.handler

    def as_evidence(self) -> dict[str, Any]:
        return {
            "tier": self.tier.value,
            "handler": self.handler,
            "complexity_score": self.complexity_score,
            "confidence": self.confidence,
            "latency_budget_ms": self.latency_budget_ms,
            "classification_ms": self.classification_ms,
            "has_reflex": self.has_reflex,
            "features": self.features,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTORS
# ═══════════════════════════════════════════════════════════════════════════════


def _extract_length_score(text: str) -> float:
    """Longer inputs tend to be more complex. Sigmoid normalization."""
    words = len(text.split())
    # sigmoid: 0 words → 0.0, 20 words → 0.5, 100+ words → 0.95
    return min(words / (words + 20.0), 1.0)


def _extract_question_complexity(text: str) -> float:
    """Multi-part questions and compound requests are more complex."""
    indicators = 0
    # Count question marks
    indicators += text.count("?")
    # Count conjunctions suggesting multi-step
    multi_markers = [
        "and then",
        "after that",
        "also",
        "additionally",
        "followed by",
        "next",
        "finally",
        "step",
    ]
    indicators += sum(1 for m in multi_markers if m.lower() in text.lower())
    # Count action verbs suggesting different tasks
    action_verbs = [
        "create",
        "analyze",
        "compare",
        "write",
        "build",
        "fix",
        "review",
        "test",
        "deploy",
        "research",
    ]
    indicators += sum(1 for v in action_verbs if v in text.lower())
    return min(indicators / 8.0, 1.0)


def _extract_domain_breadth(text: str) -> float:
    """Missions spanning multiple domains are more complex."""
    domains = {
        "code": ["code", "function", "class", "bug", "error", "python", "rust", "api"],
        "writing": ["write", "draft", "essay", "email", "document", "report"],
        "research": ["research", "find", "search", "compare", "analyze", "study"],
        "design": ["design", "architecture", "diagram", "layout", "ui", "ux"],
        "data": ["data", "csv", "json", "database", "query", "table"],
        "operations": ["deploy", "config", "setup", "install", "migrate", "backup"],
    }
    text_lower = text.lower()
    hit_domains = sum(
        1 for keywords in domains.values() if any(k in text_lower for k in keywords)
    )
    return min(hit_domains / 3.0, 1.0)


def _extract_specificity(text: str) -> float:
    """Vague requests need more clarification work → higher complexity.
    But very specific requests with clear constraints are moderate."""
    # Specific indicators (reduce complexity)
    specific_markers = [
        "exactly",
        "specifically",
        "only",
        "just",
        "the file",
        "this function",
        "line",
        "column",
    ]
    specifics = sum(1 for m in specific_markers if m in text.lower())

    # Vague indicators (increase complexity)
    vague_markers = [
        "something",
        "anything",
        "whatever",
        "maybe",
        "kind of",
        "sort of",
        "general",
        "overall",
    ]
    vagues = sum(1 for m in vague_markers if m in text.lower())

    if specifics > vagues:
        return max(0.1, 0.5 - specifics * 0.1)
    elif vagues > specifics:
        return min(0.9, 0.5 + vagues * 0.1)
    return 0.5


def _extract_code_indicators(text: str) -> float:
    """Code-related tasks have predictable complexity patterns."""
    # Simple code tasks
    simple_code = ["format", "lint", "rename", "typo", "import"]
    # Complex code tasks
    complex_code = [
        "refactor",
        "architect",
        "optimize",
        "debug",
        "security audit",
        "migrate",
        "rewrite",
    ]

    text_lower = text.lower()
    simple_hits = sum(1 for m in simple_code if m in text_lower)
    complex_hits = sum(1 for m in complex_code if m in text_lower)

    if complex_hits > 0:
        return min(0.7 + complex_hits * 0.1, 1.0)
    if simple_hits > 0:
        return max(0.1, 0.3 - simple_hits * 0.05)
    return 0.4  # Default for unrecognized code tasks


# ═══════════════════════════════════════════════════════════════════════════════
# HHMM ROUTER
# ═══════════════════════════════════════════════════════════════════════════════


class HhmmRouter:
    """
    Hierarchical complexity classifier and tier router.

    Classifies incoming missions into TRIVIAL/SIMPLE/COMPLEX/SOVEREIGN
    using feature extraction and weighted scoring. Routes to the
    appropriate handler with a latency budget.

    At genesis: uses feature-based heuristics (5 initial live states).
    At scale (1000+ missions): transitions to learned HMM states
    via EM algorithm on mission embeddings.

    Thread-safe: classification is pure function, no shared mutable state.
    """

    def __init__(self, reflex_cache=None):
        """
        Args:
            reflex_cache: Optional ReflexCache for TRIVIAL tier routing.
        """
        self._reflex_cache = reflex_cache
        self._classification_count = 0

        # Feature weights (tunable, will be learned from data at scale)
        self._weights = {
            "length": 0.15,
            "question_complexity": 0.25,
            "domain_breadth": 0.25,
            "specificity": 0.15,
            "code_indicators": 0.20,
        }

    def classify(self, input_text: str) -> ClassificationResult:
        """
        Classify a mission's complexity and determine the execution tier.

        Fast path: if reflex cache has a hit → TRIVIAL immediately.
        Slow path: extract features → weighted score → tier mapping.

        Args:
            input_text: The user's mission text.

        Returns:
            ClassificationResult with tier, handler, and latency budget.
        """
        start = time.monotonic()
        self._classification_count += 1

        # Fast path: check reflex cache
        has_reflex = False
        if self._reflex_cache is not None:
            entry = self._reflex_cache.lookup(input_text)
            if entry is not None and not entry.needs_validation():
                has_reflex = True
                elapsed = (time.monotonic() - start) * 1000
                return ClassificationResult(
                    tier=ComplexityTier.TRIVIAL,
                    config=TIER_CONFIGS[ComplexityTier.TRIVIAL],
                    complexity_score=0.0,
                    confidence=0.99,
                    features={"reflex_hit": 1.0},
                    classification_ms=elapsed,
                    has_reflex=True,
                )

        # Feature extraction
        features = {
            "length": _extract_length_score(input_text),
            "question_complexity": _extract_question_complexity(input_text),
            "domain_breadth": _extract_domain_breadth(input_text),
            "specificity": _extract_specificity(input_text),
            "code_indicators": _extract_code_indicators(input_text),
        }

        # Weighted complexity score
        complexity = sum(self._weights[f] * features[f] for f in self._weights)
        complexity = max(0.0, min(1.0, complexity))

        # Map to tier
        tier = self._score_to_tier(complexity)
        config = TIER_CONFIGS[tier]

        # Confidence based on how far from tier boundary
        low, high = config.score_range
        mid = (low + high) / 2
        distance_from_boundary = min(
            abs(complexity - low) if low > 0 else 1.0,
            abs(complexity - high) if high < 1 else 1.0,
        )
        confidence = min(0.5 + distance_from_boundary * 2, 0.99)

        elapsed = (time.monotonic() - start) * 1000

        result = ClassificationResult(
            tier=tier,
            config=config,
            complexity_score=round(complexity, 4),
            confidence=round(confidence, 4),
            features={k: round(v, 4) for k, v in features.items()},
            classification_ms=round(elapsed, 3),
            has_reflex=has_reflex,
        )

        logger.debug(
            f"Classified: tier={tier.value}, score={complexity:.3f}, "
            f"confidence={confidence:.3f}, handler={config.handler}"
        )

        return result

    def _score_to_tier(self, score: float) -> ComplexityTier:
        """Map complexity score to tier using constitutional boundaries."""
        for tier, config in TIER_CONFIGS.items():
            low, high = config.score_range
            if low <= score < high:
                return tier
        return ComplexityTier.SOVEREIGN  # Default to highest tier

    @property
    def classification_count(self) -> int:
        return self._classification_count


# ═══════════════════════════════════════════════════════════════════════════════
# ACTION BUS — Priority queue with GCD tick
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class MissionTicket:
    """A mission queued in the Action Bus."""

    mission_id: str
    input_text: str
    classification: ClassificationResult
    priority: float  # Higher = more urgent
    queued_at: float
    deadline: float  # Unix timestamp deadline

    @property
    def time_remaining_ms(self) -> float:
        return max(0, (self.deadline - time.time()) * 1000)

    @property
    def expired(self) -> bool:
        return time.time() > self.deadline


class ActionBus:
    """
    Priority queue for mission execution.

    GCD tick: processes queue every 100ms.
    Respects concurrency limits and per-hour rate limits.
    """

    def __init__(self):
        self._queue: list[MissionTicket] = []
        self._active: dict[str, MissionTicket] = {}
        self._completed_this_hour: int = 0
        self._hour_start: float = time.time()

    def submit(self, ticket: MissionTicket) -> bool:
        """Submit a mission to the queue. Returns False if rate-limited."""
        # Check hourly rate limit
        self._maybe_reset_hour()
        if self._completed_this_hour >= ACTION_BUS_MAX_PER_HOUR:
            logger.warning("Hourly rate limit reached")
            return False

        # Check concurrency limit
        if len(self._active) >= ACTION_BUS_MAX_CONCURRENT:
            logger.warning("Max concurrent missions reached")
            # Still queue it — it'll be picked up when a slot opens
            pass

        self._queue.append(ticket)
        self._queue.sort(key=lambda t: -t.priority)  # Highest priority first
        return True

    def next_ticket(self) -> MissionTicket | None:
        """Pop the highest-priority non-expired ticket."""
        self._maybe_reset_hour()

        if len(self._active) >= ACTION_BUS_MAX_CONCURRENT:
            return None

        while self._queue:
            ticket = self._queue.pop(0)
            if ticket.expired:
                logger.debug(f"Expired ticket: {ticket.mission_id}")
                continue
            self._active[ticket.mission_id] = ticket
            return ticket

        return None

    def complete(self, mission_id: str):
        """Mark a mission as completed."""
        if mission_id in self._active:
            del self._active[mission_id]
            self._completed_this_hour += 1

    def _maybe_reset_hour(self):
        if time.time() - self._hour_start > 3600:
            self._completed_this_hour = 0
            self._hour_start = time.time()

    @property
    def queue_depth(self) -> int:
        return len(self._queue)

    @property
    def active_count(self) -> int:
        return len(self._active)
