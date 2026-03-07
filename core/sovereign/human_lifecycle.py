"""
Human Lifecycle — Seed to Catalyst
===================================

Maps sovereignty scores to the 7-stage human growth progression.
This is the human-readable mirror of the agent skill tree
(Novice -> Grandmaster). Both progressions are earned through
verified work, gated by quality, and compound over time.

Pure functions, no I/O. All thresholds from constants.py.

Standing on Giants:
- Maslow (1943): Hierarchy of needs — growth is staged
- Kohlberg (1958): Moral development — earned through practice
- Al-Ghazali (1095): Ihsan — excellence as the floor

Phase 72 — Constitutional Kernel
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.integration.constants import (
    HUMAN_STAGE_ORDER,
    HUMAN_STAGE_THRESHOLDS,
)


# ---------------------------------------------------------------------------
# Human Growth Stage dataclass
# ---------------------------------------------------------------------------

_STAGE_DESCRIPTIONS = {
    "Seed": (
        "First install. Identity created. Potential is infinite.",
        "Install Node0, generate Ed25519 keypair",
    ),
    "Node": (
        "First mission completed. The seed has sprouted.",
        "Complete first mission with Ihsan >= 0.85",
    ),
    "Apprentice": (
        "Consistent work. Building habits. Learning the system.",
        "10+ qualified episodes, qualification rate >= 50%",
    ),
    "Builder": (
        "Compiled first reflex. Work is becoming automatic.",
        "First reflex compiled (3+ consecutive qualified)",
    ),
    "Verifier": (
        "Trusted to attest others' work. Quality is habitual.",
        "Sovereignty >= 0.55, qualification rate >= 75%",
    ),
    "Mentor": (
        "Skills published to marketplace. Helping others grow.",
        "Published 3+ compiled reflexes as tradeable skills",
    ),
    "Catalyst": (
        "Network effect multiplier. The seed has become a forest.",
        "Sovereignty >= 0.85, 5+ mentored nodes, FOREST tier",
    ),
}


@dataclass(frozen=True)
class HumanStage:
    """A single stage in the human growth lifecycle."""

    name: str
    rank: int
    score_low: float
    score_high: float
    description: str
    unlock_condition: str


# ---------------------------------------------------------------------------
# Build STAGES list from constants.py thresholds
# ---------------------------------------------------------------------------

def _build_stages() -> list[HumanStage]:
    """Construct stage list from centralized constants."""
    stages: list[HumanStage] = []
    order = HUMAN_STAGE_ORDER
    for i, name in enumerate(order):
        score_low = HUMAN_STAGE_THRESHOLDS[name]
        score_high = (
            HUMAN_STAGE_THRESHOLDS[order[i + 1]] if i + 1 < len(order) else 1.0
        )
        desc, unlock = _STAGE_DESCRIPTIONS.get(
            name, ("Unknown stage.", "Unknown condition")
        )
        stages.append(
            HumanStage(
                name=name,
                rank=i,
                score_low=score_low,
                score_high=score_high,
                description=desc,
                unlock_condition=unlock,
            )
        )
    return stages


STAGES: list[HumanStage] = _build_stages()


# ---------------------------------------------------------------------------
# Agent <-> Human tier alignment
# ---------------------------------------------------------------------------

AGENT_TIER_MAP = {
    "Seed": "Novice",
    "Node": "Apprentice",
    "Apprentice": "Journeyman",
    "Builder": "Craftsman",
    "Verifier": "Expert",
    "Mentor": "Master",
    "Catalyst": "Grandmaster",
}


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def human_stage(sovereignty_score: float) -> str:
    """Map sovereignty score to human lifecycle stage name."""
    clamped = _clamp(sovereignty_score)
    for stage in reversed(STAGES):
        if clamped >= stage.score_low:
            return stage.name
    return "Seed"


def human_stage_detail(sovereignty_score: float) -> HumanStage:
    """Full stage metadata for UI/API."""
    clamped = _clamp(sovereignty_score)
    for stage in reversed(STAGES):
        if clamped >= stage.score_low:
            return stage
    return STAGES[0]


def stage_progress(sovereignty_score: float) -> dict:
    """Progress within current stage + next stage info.

    Returns a dict suitable for direct JSON serialization
    (the /v1/node/lifecycle API response).
    """
    score = _clamp(sovereignty_score)
    stage = human_stage_detail(score)
    range_size = stage.score_high - stage.score_low
    progress = (score - stage.score_low) / range_size if range_size > 0 else 1.0

    next_stage: Optional[HumanStage] = None
    if stage.rank < len(STAGES) - 1:
        next_stage = STAGES[stage.rank + 1]

    points_to_next = 0.0
    if next_stage is not None and score < next_stage.score_low:
        points_to_next = round(next_stage.score_low - score, 4)

    return {
        "current_stage": stage.name,
        "rank": stage.rank,
        "progress": round(_clamp(progress), 4),
        "sovereignty_score": round(score, 4),
        "next_stage": next_stage.name if next_stage else None,
        "next_threshold": next_stage.score_low if next_stage else None,
        "points_to_next": points_to_next,
        "description": stage.description,
        "unlock_condition": stage.unlock_condition,
    }


def agent_tier_equivalent(human_stage_name: str) -> str:
    """What agent skill tier matches this human stage?"""
    return AGENT_TIER_MAP.get(human_stage_name, "Novice")
