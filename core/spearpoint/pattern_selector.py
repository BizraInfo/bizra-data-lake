"""
PatternStrategySelector — Intelligent Thinking Pattern Routing
==============================================================

Selects the optimal Sci-Reasoning thinking pattern for each
heartbeat cycle based on:
1. Past outcome history (don't repeat recent failures)
2. Pattern co-occurrence (complementary patterns after success)
3. Rotation (explore all 15 patterns over time)

The selector balances exploitation (use what works) with exploration
(try new cognitive strategies) — a multi-armed bandit problem solved
with Thompson Sampling intuition.

Standing on Giants:
- Li et al. (2025): Sci-Reasoning 15 thinking patterns
- Thompson (1933): Probability matching for exploration
- Boyd (1995): OODA Orient — pattern selection IS orientation
- Shannon (1948): Information gain from diverse exploration
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class PatternOutcome:
    """Record of a pattern's performance in a heartbeat cycle."""

    pattern_id: str
    approved: int = 0
    rejected: int = 0
    inconclusive: int = 0
    total: int = 0
    last_used_cycle: int = -1


@dataclass
class PatternSelectionResult:
    """Result of pattern selection."""

    pattern_id: str
    strategy: str  # "explore", "exploit", "complement", "rotate"
    reason: str
    confidence: float  # 0-1, how confident we are in this choice


class PatternStrategySelector:
    """
    Selects the best thinking pattern for each heartbeat cycle.

    Strategy priority:
    1. If a pattern was recently approved → try its complement (exploit)
    2. If all patterns tried recently → pick least-recently-used (rotate)
    3. If a pattern keeps failing → deprioritize it (learn)
    4. Default → weighted random by success rate (explore)

    Usage:
        selector = PatternStrategySelector()
        result = selector.select(cycle=0)
        # result.pattern_id = "P01"
        selector.record_outcome("P01", approved=1, rejected=0, cycle=0)
    """

    ALL_PATTERNS = [f"P{i:02d}" for i in range(1, 16)]

    def __init__(
        self,
        seed: Optional[int] = None,
        explore_probability: float = 0.3,
        cooldown_cycles: int = 2,
    ):
        self._rng = random.Random(seed)
        self._explore_prob = explore_probability
        self._cooldown = cooldown_cycles

        # Outcome history per pattern
        self._outcomes: dict[str, PatternOutcome] = {
            pid: PatternOutcome(pattern_id=pid) for pid in self.ALL_PATTERNS
        }

        # Co-occurrence data (loaded from taxonomy)
        self._cooccurrence: dict[str, dict[str, int]] = {}

        # Last approved pattern (for complement strategy)
        self._last_approved_pattern: Optional[str] = None

    def load_cooccurrence(self, cooccurrence: dict[str, dict[str, int]]) -> None:
        """Load pattern co-occurrence data from Sci-Reasoning taxonomy."""
        self._cooccurrence = cooccurrence
        logger.info(f"Loaded co-occurrence data for {len(cooccurrence)} patterns")

    def select(self, cycle: int = 0) -> PatternSelectionResult:
        """
        Select the optimal thinking pattern for this cycle.

        Args:
            cycle: Current heartbeat cycle number

        Returns:
            PatternSelectionResult with pattern_id and strategy rationale
        """
        # Strategy 1: Complement — if last cycle approved, try a complementary pattern
        if self._last_approved_pattern is not None:
            complement = self._pick_complement(self._last_approved_pattern, cycle)
            if complement is not None:
                return complement

        # Strategy 2: Explore — random exploration with probability
        if self._rng.random() < self._explore_prob:
            return self._explore(cycle)

        # Strategy 3: Exploit — pick the pattern with best success rate
        exploit = self._exploit(cycle)
        if exploit is not None:
            return exploit

        # Strategy 4: Rotate — least recently used
        return self._rotate(cycle)

    def record_outcome(
        self,
        pattern_id: str,
        approved: int = 0,
        rejected: int = 0,
        inconclusive: int = 0,
        cycle: int = 0,
    ) -> None:
        """Record the outcome of using a pattern in a cycle."""
        if pattern_id not in self._outcomes:
            self._outcomes[pattern_id] = PatternOutcome(pattern_id=pattern_id)

        outcome = self._outcomes[pattern_id]
        outcome.approved += approved
        outcome.rejected += rejected
        outcome.inconclusive += inconclusive
        outcome.total += approved + rejected + inconclusive
        outcome.last_used_cycle = cycle

        # Track last approved for complement strategy
        if approved > 0:
            self._last_approved_pattern = pattern_id
        elif rejected > 0 and self._last_approved_pattern == pattern_id:
            self._last_approved_pattern = None

    def _pick_complement(
        self, base_pattern: str, cycle: int
    ) -> Optional[PatternSelectionResult]:
        """Pick a complementary pattern based on co-occurrence."""
        cooc = self._cooccurrence.get(base_pattern, {})
        if not cooc:
            return None

        # Sort by co-occurrence count, filter out recently used and failed patterns
        candidates = []
        for pid, count in sorted(cooc.items(), key=lambda x: x[1], reverse=True):
            if pid not in self._outcomes:
                continue
            outcome = self._outcomes[pid]
            # Skip if used too recently
            if cycle - outcome.last_used_cycle < self._cooldown:
                continue
            # Skip if consistently failing
            if outcome.total >= 3 and outcome.approved == 0:
                continue
            candidates.append((pid, count))

        if not candidates:
            return None

        pid = candidates[0][0]
        return PatternSelectionResult(
            pattern_id=pid,
            strategy="complement",
            reason=f"Complementary to {base_pattern} (co-occurrence: {candidates[0][1]})",
            confidence=0.7,
        )

    def _exploit(self, cycle: int) -> Optional[PatternSelectionResult]:
        """Pick the pattern with best historical success rate."""
        scored: list[tuple[str, float]] = []

        for pid, outcome in self._outcomes.items():
            if outcome.total == 0:
                continue
            # Skip if used too recently
            if cycle - outcome.last_used_cycle < self._cooldown:
                continue
            success_rate = outcome.approved / outcome.total
            scored.append((pid, success_rate))

        if not scored:
            return None

        scored.sort(key=lambda x: x[1], reverse=True)
        best_pid, best_rate = scored[0]

        if best_rate == 0:
            return None  # No pattern has ever succeeded, fall to rotate

        return PatternSelectionResult(
            pattern_id=best_pid,
            strategy="exploit",
            reason=f"Best success rate: {best_rate:.1%} ({self._outcomes[best_pid].approved}/{self._outcomes[best_pid].total})",
            confidence=min(0.9, best_rate + 0.1),
        )

    def _explore(self, cycle: int) -> PatternSelectionResult:
        """Random exploration weighted by inverse failure rate."""
        weights: list[float] = []
        available: list[str] = []

        for pid in self.ALL_PATTERNS:
            outcome = self._outcomes[pid]
            # Higher weight for unexplored or successful patterns
            if outcome.total == 0:
                w = 2.0  # Unexplored → high curiosity
            else:
                success_rate = outcome.approved / outcome.total
                w = 0.5 + success_rate  # Base 0.5 + success bonus
            weights.append(w)
            available.append(pid)

        chosen = self._rng.choices(available, weights=weights, k=1)[0]
        return PatternSelectionResult(
            pattern_id=chosen,
            strategy="explore",
            reason="Exploration: weighted random selection",
            confidence=0.4,
        )

    def _rotate(self, cycle: int) -> PatternSelectionResult:
        """Pick the least recently used pattern."""
        sorted_by_recency = sorted(
            self._outcomes.values(),
            key=lambda o: o.last_used_cycle,
        )

        # Filter out consistently failing patterns (if they have enough trials)
        for outcome in sorted_by_recency:
            if outcome.total >= 5 and outcome.approved == 0:
                continue  # Skip chronic failures
            return PatternSelectionResult(
                pattern_id=outcome.pattern_id,
                strategy="rotate",
                reason=f"Least recently used (last cycle: {outcome.last_used_cycle})",
                confidence=0.5,
            )

        # Fallback: just pick the oldest
        return PatternSelectionResult(
            pattern_id=sorted_by_recency[0].pattern_id,
            strategy="rotate",
            reason="Rotation fallback (all patterns have chronic failures)",
            confidence=0.3,
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get selector statistics."""
        active = sum(1 for o in self._outcomes.values() if o.total > 0)
        total_approved = sum(o.approved for o in self._outcomes.values())
        total_rejected = sum(o.rejected for o in self._outcomes.values())
        return {
            "patterns_tried": active,
            "patterns_available": len(self.ALL_PATTERNS),
            "total_approved": total_approved,
            "total_rejected": total_rejected,
            "last_approved_pattern": self._last_approved_pattern,
            "has_cooccurrence": bool(self._cooccurrence),
            "per_pattern": {
                pid: {
                    "approved": o.approved,
                    "rejected": o.rejected,
                    "total": o.total,
                }
                for pid, o in self._outcomes.items()
                if o.total > 0
            },
        }


__all__ = [
    "PatternStrategySelector",
    "PatternSelectionResult",
    "PatternOutcome",
]
