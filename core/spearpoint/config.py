"""
Spearpoint Configuration — Types, Thresholds, and Tier Behavior Policy
=======================================================================

All thresholds imported from core/integration/constants.py (single source of truth).
Tier behavior policy maps credibility scores to permitted actions.

Standing on Giants: Shannon (SNR) + Deming (continuous improvement)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from core.integration.constants import (
    SNR_THRESHOLD_T0_ELITE,
    SNR_THRESHOLD_T1_HIGH,
    STRICT_IHSAN_THRESHOLD,
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)


class TierLevel(str, Enum):
    """Credibility tier levels mapped from evaluation scores."""

    REJECT = "reject"  # < 0.85: output rejected, demand more evidence
    DIAGNOSTICS = "diagnostics"  # 0.85 - 0.949: diagnostics-only, no recommendations
    OPERATIONAL = "operational"  # 0.95 - 0.979: recommend + confirmation gate
    ELITE = "elite"  # 0.98 - 0.989: stronger provenance requirement
    PROPOSAL = "proposal"  # >= 0.99: proposal-ready, can generate patch plan


class MissionType(str, Enum):
    """Types of spearpoint missions."""

    REPRODUCE = "reproduce"  # Evaluate/verify a claim (AutoEvaluator)
    IMPROVE = "improve"  # Research/propose improvements (AutoResearcher)


@dataclass(frozen=True)
class TierPolicy:
    """Behavior permissions for a credibility tier."""

    level: TierLevel
    min_score: float
    max_score: float
    may_recommend: bool
    may_propose_patch: bool
    requires_confirmation: bool
    requires_provenance: bool
    diagnostics_only: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "may_recommend": self.may_recommend,
            "may_propose_patch": self.may_propose_patch,
            "requires_confirmation": self.requires_confirmation,
            "requires_provenance": self.requires_provenance,
            "diagnostics_only": self.diagnostics_only,
        }


# Canonical tier policies — ordered from lowest to highest
TIER_POLICIES: list[TierPolicy] = [
    TierPolicy(
        level=TierLevel.REJECT,
        min_score=0.0,
        max_score=UNIFIED_SNR_THRESHOLD,
        may_recommend=False,
        may_propose_patch=False,
        requires_confirmation=False,
        requires_provenance=False,
        diagnostics_only=False,  # Output is rejected entirely
    ),
    TierPolicy(
        level=TierLevel.DIAGNOSTICS,
        min_score=UNIFIED_SNR_THRESHOLD,
        max_score=UNIFIED_IHSAN_THRESHOLD,
        may_recommend=False,
        may_propose_patch=False,
        requires_confirmation=False,
        requires_provenance=False,
        diagnostics_only=True,
    ),
    TierPolicy(
        level=TierLevel.OPERATIONAL,
        min_score=UNIFIED_IHSAN_THRESHOLD,
        max_score=SNR_THRESHOLD_T0_ELITE,
        may_recommend=True,
        may_propose_patch=False,
        requires_confirmation=True,
        requires_provenance=False,
        diagnostics_only=False,
    ),
    TierPolicy(
        level=TierLevel.ELITE,
        min_score=SNR_THRESHOLD_T0_ELITE,
        max_score=STRICT_IHSAN_THRESHOLD,
        may_recommend=True,
        may_propose_patch=False,
        requires_confirmation=True,
        requires_provenance=True,
        diagnostics_only=False,
    ),
    TierPolicy(
        level=TierLevel.PROPOSAL,
        min_score=STRICT_IHSAN_THRESHOLD,
        max_score=1.01,  # Above 1.0 to capture exactly 1.0
        may_recommend=True,
        may_propose_patch=True,
        requires_confirmation=True,
        requires_provenance=True,
        diagnostics_only=False,
    ),
]


def resolve_tier(score: float) -> TierPolicy:
    """Resolve credibility score to tier policy. Fail-closed: unknown -> REJECT."""
    for policy in reversed(TIER_POLICIES):
        if score >= policy.min_score:
            return policy
    return TIER_POLICIES[0]  # REJECT


@dataclass
class SpearpointConfig:
    """Configuration for the Spearpoint orchestration layer."""

    # Thresholds (from constants.py)
    ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD
    snr_threshold: float = UNIFIED_SNR_THRESHOLD
    elite_threshold: float = SNR_THRESHOLD_T0_ELITE
    t1_threshold: float = SNR_THRESHOLD_T1_HIGH

    # Loop constraints
    max_iterations_per_cycle: int = 10
    loop_interval_seconds: float = 300.0
    circuit_breaker_consecutive_rejections: int = 3
    circuit_breaker_backoff_seconds: float = 60.0

    # Paths — defaults align with bizra_config.SPEARPOINT_STATE_DIR
    state_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "SPEARPOINT_STATE_DIR",
                str(Path(__file__).resolve().parents[2] / ".spearpoint"),
            )
        )
    )
    evidence_ledger_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "SPEARPOINT_EVIDENCE_LEDGER",
                str(Path(__file__).resolve().parents[2] / ".spearpoint" / "evidence.jsonl"),
            )
        )
    )
    hypothesis_memory_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "SPEARPOINT_HYPOTHESIS_MEMORY",
                str(Path(__file__).resolve().parents[2] / ".spearpoint" / "hypothesis_memory"),
            )
        )
    )

    # Budget
    max_cost_usd: float = 10.0
    max_tokens: int = 1_000_000

    def ensure_dirs(self) -> None:
        """Create required directories."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.hypothesis_memory_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "SpearpointConfig":
        """Load config from environment variables with defaults."""
        config = cls()
        if interval := os.getenv("SPEARPOINT_LOOP_INTERVAL"):
            config.loop_interval_seconds = float(interval)
        if max_iter := os.getenv("SPEARPOINT_MAX_ITERATIONS"):
            config.max_iterations_per_cycle = int(max_iter)
        if budget := os.getenv("SPEARPOINT_BUDGET_USD"):
            config.max_cost_usd = float(budget)
        return config

    def to_dict(self) -> dict[str, Any]:
        return {
            "ihsan_threshold": self.ihsan_threshold,
            "snr_threshold": self.snr_threshold,
            "max_iterations_per_cycle": self.max_iterations_per_cycle,
            "loop_interval_seconds": self.loop_interval_seconds,
            "state_dir": str(self.state_dir),
            "evidence_ledger_path": str(self.evidence_ledger_path),
        }


__all__ = [
    "TierLevel",
    "TierPolicy",
    "MissionType",
    "SpearpointConfig",
    "TIER_POLICIES",
    "resolve_tier",
]
