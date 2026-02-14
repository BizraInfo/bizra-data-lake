"""
Self-Evolving Judgment Engine (SJE) — Observation Telemetry
============================================================
Phase A: Observation Mode Only.

Records verdict distributions and computes entropy to measure
judgment stability. NO policy mutation. NO threshold changes.

Standing on Giants:
  - Shannon (1948): Entropy as uncertainty measure
  - Aristotle (Nicomachean Ethics): Practical wisdom via observation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class JudgmentVerdict(Enum):
    """Possible judgment outcomes for an episode."""

    PROMOTE = "promote"
    NEUTRAL = "neutral"
    DEMOTE = "demote"
    FORBID = "forbid"


# Fixed-point precision for entropy (P = 10^6)
ENTROPY_PRECISION = 1_000_000


def _fixed_point_log2(n: int) -> int:
    """Integer-approximated log2 scaled by ENTROPY_PRECISION.

    Returns floor(log2(n)) * ENTROPY_PRECISION for n >= 2.
    Returns 0 for n <= 1.
    """
    if n <= 1:
        return 0
    bits = n.bit_length() - 1
    return bits * ENTROPY_PRECISION


@dataclass
class JudgmentTelemetry:
    """Observation-mode telemetry for the SJE.

    Tracks verdict distribution and computes Shannon entropy
    to measure judgment stability over time.
    """

    verdict_counts: dict[JudgmentVerdict, int] = field(
        default_factory=lambda: {
            JudgmentVerdict.PROMOTE: 0,
            JudgmentVerdict.NEUTRAL: 0,
            JudgmentVerdict.DEMOTE: 0,
            JudgmentVerdict.FORBID: 0,
        }
    )
    total_observations: int = 0

    def observe(self, verdict: JudgmentVerdict) -> None:
        """Record a verdict observation."""
        self.verdict_counts[verdict] = self.verdict_counts.get(verdict, 0) + 1
        self.total_observations += 1

    def entropy(self) -> float:
        """Compute Shannon entropy of the verdict distribution.

        H = -sum(p_i * log2(p_i)) for all verdicts with p_i > 0.
        Returns 0.0 for zero or one observations.
        Max entropy = log2(4) = 2.0 (uniform over 4 verdicts).
        """
        if self.total_observations <= 1:
            return 0.0

        h = 0.0
        for count in self.verdict_counts.values():
            if count > 0:
                p = count / self.total_observations
                h -= p * math.log2(p)
        return h

    def dominant_verdict(self) -> Optional[JudgmentVerdict]:
        """Return the most frequent verdict, or None if empty."""
        if self.total_observations == 0:
            return None
        return max(self.verdict_counts, key=lambda v: self.verdict_counts[v])

    def distribution(self) -> dict[str, float]:
        """Return normalized distribution as percentages."""
        if self.total_observations == 0:
            return {v.value: 0.0 for v in JudgmentVerdict}
        return {
            v.value: (c / self.total_observations) * 100.0
            for v, c in self.verdict_counts.items()
        }

    def is_stable(self, entropy_threshold: float = 0.5) -> bool:
        """Check if judgment is stable (low entropy = high consensus).

        A threshold of 0.5 means the distribution is strongly skewed
        toward one verdict.
        """
        return self.entropy() < entropy_threshold

    def to_dict(self) -> dict:
        """Serialize telemetry for logging/export."""
        return {
            "total_observations": self.total_observations,
            "verdict_counts": {v.value: c for v, c in self.verdict_counts.items()},
            "entropy": round(self.entropy(), 6),
            "dominant_verdict": (
                self.dominant_verdict().value if self.dominant_verdict() else None
            ),
            "is_stable": self.is_stable(),
        }


def simulate_epoch_distribution(
    impacts: list[int],
    epoch_cap: int,
) -> list[int]:
    """Simulate proportional epoch allocation. No tokens emitted.

    Pure mathematical rehearsal for genesis economy modeling.
    Each node receives: floor(impact_i * epoch_cap / total_impact).

    Edge cases:
    - Empty impacts: returns empty list
    - Zero total: returns list of zeros
    - Single node: receives full epoch_cap
    """
    if not impacts:
        return []

    total = sum(impacts)
    if total == 0:
        return [0] * len(impacts)

    return [(i * epoch_cap) // total for i in impacts]
