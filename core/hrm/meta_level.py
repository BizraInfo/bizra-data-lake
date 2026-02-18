"""
Hierarchical Reasoning Model — Meta-Autopoietic Level N

Implements the highest abstraction level where the hierarchy reasons
about ITSELF. Level N observes performance of lower levels, generates
hypotheses about architectural improvements, explores alternatives
through GoT simulation, and integrates successful changes.

The hierarchy is not fixed — it EVOLVES.

HRM PDF Table 7: Meta-Autopoietic Operations at Level N
  - Level Addition: Create new abstraction layer
  - Level Merger: Combine redundant levels
  - Boundary Tuning: Adjust permeability
  - SNR Rebalancing: Shift quality thresholds
  - Protocol Evolution: Modify cross-level mechanisms

Golden Gem: The Architectural Evolution Principle
  The hierarchy the system starts with is not the hierarchy it ends with.
  The final architecture reflects not the designer's assumptions but the
  discovered structure of the problem space.

Standing on Giants:
  - Maturana & Varela (1980) — Autopoiesis as self-creation
  - Hofstadter (1979) — Strange loops and self-reference
  - Beer (1972) — Viable System Model (recursive self-organization)
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from core.hrm.abstraction_levels import (
    HRM_SNR_GRADIENT,
    AbstractionLevel,
    LevelBoundary,
    LevelConfig,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# META-OPERATIONS — How Level N Modifies the Hierarchy
# ═══════════════════════════════════════════════════════════════════════════════


class MetaOperation(str, Enum):
    """Operations Level N can perform on the hierarchy itself."""

    LEVEL_ADDITION = "level_addition"  # Create new abstraction layer
    LEVEL_MERGER = "level_merger"  # Combine redundant levels
    BOUNDARY_TUNING = "boundary_tuning"  # Adjust permeability
    SNR_REBALANCING = "snr_rebalancing"  # Shift quality thresholds
    PROTOCOL_EVOLUTION = "protocol_evolution"  # Modify cross-level mechanisms


class TriggerCondition(str, Enum):
    """Conditions that trigger meta-operations."""

    PERSISTENT_INTERMEDIATE_NEEDS = "persistent_intermediate_needs"
    HIGH_REDUNDANCY = "high_redundancy"
    INFORMATION_BOTTLENECK = "information_bottleneck"
    LEVEL_PERFORMANCE_ISSUE = "level_performance_issue"
    COORDINATION_FAILURE = "coordination_failure"


@dataclass
class MetaObservation:
    """
    Level N's observation of the hierarchy's performance.

    Captures level-specific metrics, cross-level dynamics,
    and architectural fitness indicators.
    """

    observation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: float = field(default_factory=time.time)

    # Per-level performance
    level_snr_scores: Dict[AbstractionLevel, float] = field(default_factory=dict)
    level_cycle_counts: Dict[AbstractionLevel, int] = field(default_factory=dict)
    level_learning_velocities: Dict[AbstractionLevel, float] = field(
        default_factory=dict
    )

    # Cross-level dynamics
    bottlenecks: List[Tuple[AbstractionLevel, AbstractionLevel]] = field(
        default_factory=list
    )
    redundancies: List[Tuple[AbstractionLevel, AbstractionLevel]] = field(
        default_factory=list
    )
    resonance_count: int = 0

    # Bridge metrics
    message_pass_rate: float = 0.0
    contradiction_count: int = 0
    sync_quality: float = 0.0

    @property
    def architectural_fitness(self) -> float:
        """
        Composite fitness score for the current architecture.

        Components:
          - Mean level SNR (weight: 0.3)
          - Sync quality (weight: 0.25)
          - Message pass rate (weight: 0.2)
          - Absence of bottlenecks (weight: 0.15)
          - Low redundancy (weight: 0.1)
        """
        if not self.level_snr_scores:
            return 0.0

        snr_mean = sum(self.level_snr_scores.values()) / len(self.level_snr_scores)
        bottleneck_penalty = min(len(self.bottlenecks) * 0.1, 0.5)
        redundancy_penalty = min(len(self.redundancies) * 0.05, 0.3)

        fitness = (
            snr_mean * 0.30
            + self.sync_quality * 0.25
            + self.message_pass_rate * 0.20
            + (1.0 - bottleneck_penalty) * 0.15
            + (1.0 - redundancy_penalty) * 0.10
        )
        return round(min(1.0, max(0.0, fitness)), 4)


@dataclass
class MetaProposal:
    """A proposed modification to the hierarchy."""

    proposal_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    operation: MetaOperation = MetaOperation.BOUNDARY_TUNING
    trigger: TriggerCondition = TriggerCondition.LEVEL_PERFORMANCE_ISSUE
    target: Dict[str, Any] = field(default_factory=dict)
    expected_improvement: float = 0.0
    risk_level: float = 0.0
    rationale: str = ""
    applied: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# META-AUTOPOIETIC LEVEL — The Hierarchy Reasoning About Itself
# ═══════════════════════════════════════════════════════════════════════════════


class MetaAutopoieticLevel:
    """
    Level N: The hierarchy's self-reflective layer.

    Observes all lower levels, detects architectural issues, proposes
    modifications, and applies changes. Each modification is treated
    as a hypothesis to be validated — the system restructures itself
    without external intervention.

    The system grows its cognitive organs in response to what it learns.
    """

    def __init__(
        self,
        min_improvement_threshold: float = 0.02,
        max_risk_tolerance: float = 0.3,
        observation_window: int = 10,
    ):
        self._min_improvement = min_improvement_threshold
        self._max_risk = max_risk_tolerance
        self._window = observation_window

        # History
        self._observations: List[MetaObservation] = []
        self._proposals: List[MetaProposal] = []
        self._applied_operations: List[MetaProposal] = []

    def observe_hierarchy(
        self,
        level_states: Dict[AbstractionLevel, Dict[str, Any]],
        bridge_metrics: Dict[str, Any],
    ) -> MetaObservation:
        """
        Observe the hierarchy's current state and performance.

        Collects level-specific metrics, identifies bottlenecks and
        redundancies, and computes architectural fitness.
        """
        obs = MetaObservation()

        # Extract per-level metrics
        for level, state in level_states.items():
            obs.level_snr_scores[level] = state.get("snr_score", 0.0)
            obs.level_cycle_counts[level] = state.get("cycle_count", 0)
            obs.level_learning_velocities[level] = state.get("learning_velocity", 0.0)

        # Extract bridge metrics
        obs.message_pass_rate = bridge_metrics.get("pass_rate", 0.0)
        obs.sync_quality = bridge_metrics.get("sync_quality", 0.0)
        obs.resonance_count = bridge_metrics.get("resonance_events", 0)

        # Detect bottlenecks (low pass rate between specific levels)
        boundary_health = bridge_metrics.get("boundary_health", [])
        for bh in boundary_health:
            if bh.get("utilization", 0) > 10 and bh.get("messages_blocked", 0) > bh.get(
                "messages_passed", 0
            ):
                # Parse boundary name
                parts = bh.get("boundary", "").split("→")
                if len(parts) == 2:
                    try:
                        src = AbstractionLevel[parts[0]]
                        tgt = AbstractionLevel[parts[1]]
                        obs.bottlenecks.append((src, tgt))
                    except (KeyError, ValueError):
                        pass

        # Detect redundancies (adjacent levels with very similar SNR)
        levels = sorted(obs.level_snr_scores.keys())
        for i in range(len(levels) - 1):
            snr_a = obs.level_snr_scores.get(levels[i], 0)
            snr_b = obs.level_snr_scores.get(levels[i + 1], 0)
            if abs(snr_a - snr_b) < 0.02 and snr_a > 0:
                obs.redundancies.append((levels[i], levels[i + 1]))

        self._observations.append(obs)
        return obs

    def propose_modification(
        self,
        observation: MetaObservation,
    ) -> Optional[MetaProposal]:
        """
        Based on observation, propose an architectural modification.

        Uses trigger conditions from HRM PDF Table 7 to determine
        what operation would most improve the hierarchy.
        """
        # Priority 1: Fix bottlenecks (highest impact)
        if observation.bottlenecks:
            src, tgt = observation.bottlenecks[0]
            proposal = MetaProposal(
                operation=MetaOperation.BOUNDARY_TUNING,
                trigger=TriggerCondition.INFORMATION_BOTTLENECK,
                target={
                    "source_level": src.name,
                    "target_level": tgt.name,
                    "action": "increase_permeability",
                    "delta": 0.1,
                },
                expected_improvement=0.08,
                risk_level=0.1,
                rationale=(
                    f"Bottleneck detected between {src.name} and {tgt.name}. "
                    f"Increasing boundary permeability to improve flow."
                ),
            )
            self._proposals.append(proposal)
            return proposal

        # Priority 2: Fix level performance issues (SNR below threshold)
        for level, snr in observation.level_snr_scores.items():
            expected = HRM_SNR_GRADIENT.get(level, 0.85)
            if snr < expected * 0.9:  # 10% below target
                proposal = MetaProposal(
                    operation=MetaOperation.SNR_REBALANCING,
                    trigger=TriggerCondition.LEVEL_PERFORMANCE_ISSUE,
                    target={
                        "level": level.name,
                        "current_snr": snr,
                        "target_snr": expected,
                        "action": "relax_threshold",
                        "new_threshold": expected * 0.95,
                    },
                    expected_improvement=0.05,
                    risk_level=0.15,
                    rationale=(
                        f"Level {level.name} SNR ({snr:.3f}) is below "
                        f"target ({expected:.3f}). Temporarily relaxing "
                        f"threshold to prevent stagnation."
                    ),
                )
                self._proposals.append(proposal)
                return proposal

        # Priority 3: Merge redundant levels
        if observation.redundancies:
            lvl_a, lvl_b = observation.redundancies[0]
            proposal = MetaProposal(
                operation=MetaOperation.LEVEL_MERGER,
                trigger=TriggerCondition.HIGH_REDUNDANCY,
                target={
                    "level_a": lvl_a.name,
                    "level_b": lvl_b.name,
                    "action": "flag_for_review",
                },
                expected_improvement=0.03,
                risk_level=0.25,
                rationale=(
                    f"Levels {lvl_a.name} and {lvl_b.name} show high "
                    f"redundancy (SNR delta < 0.02). Consider merging."
                ),
            )
            self._proposals.append(proposal)
            return proposal

        # No modification needed
        return None

    def evaluate_modification(
        self,
        proposal: MetaProposal,
    ) -> float:
        """
        Evaluate expected benefit vs risk of a proposed modification.

        Returns score from -1.0 (reject) to 1.0 (strongly approve).
        """
        if proposal.risk_level > self._max_risk:
            return -0.5  # Too risky

        if proposal.expected_improvement < self._min_improvement:
            return -0.2  # Too small to bother

        # Net benefit = improvement - risk
        net = proposal.expected_improvement - (proposal.risk_level * 0.5)
        return round(min(1.0, max(-1.0, net * 5.0)), 3)

    def apply_modification(
        self,
        proposal: MetaProposal,
        boundaries: Dict[tuple, LevelBoundary],
        level_configs: List[LevelConfig],
    ) -> bool:
        """
        Apply a meta-operation to the hierarchy.

        Returns True if applied, False if rejected.
        """
        score = self.evaluate_modification(proposal)
        if score <= 0:
            logger.info(
                "Meta-operation %s rejected (score=%.3f)",
                proposal.operation.value,
                score,
            )
            return False

        if proposal.operation == MetaOperation.BOUNDARY_TUNING:
            target = proposal.target
            src_name = target.get("source_level", "")
            tgt_name = target.get("target_level", "")
            delta = target.get("delta", 0.1)

            try:
                src = AbstractionLevel[src_name]
                tgt = AbstractionLevel[tgt_name]
                key = (src, tgt)
                if key in boundaries:
                    old_perm = boundaries[key].permeability
                    boundaries[key].permeability = min(1.0, old_perm + delta)
                    logger.info(
                        "Boundary %s→%s permeability: %.2f → %.2f",
                        src_name,
                        tgt_name,
                        old_perm,
                        boundaries[key].permeability,
                    )
            except (KeyError, ValueError):
                return False

        elif proposal.operation == MetaOperation.SNR_REBALANCING:
            # SNR rebalancing is advisory — actual thresholds are
            # constitutional and require formal amendment
            logger.info(
                "SNR rebalancing proposed for %s: %s → %s (advisory)",
                proposal.target.get("level", "unknown"),
                proposal.target.get("current_snr", "?"),
                proposal.target.get("new_threshold", "?"),
            )

        proposal.applied = True
        self._applied_operations.append(proposal)
        return True

    def get_architectural_fitness(self) -> float:
        """Get the most recent architectural fitness score."""
        if not self._observations:
            return 0.0
        return self._observations[-1].architectural_fitness

    def get_status(self) -> Dict[str, Any]:
        """Return meta-level status for telemetry."""
        return {
            "observations_count": len(self._observations),
            "proposals_count": len(self._proposals),
            "applied_count": len(self._applied_operations),
            "current_fitness": self.get_architectural_fitness(),
            "recent_proposals": [
                {
                    "id": p.proposal_id,
                    "operation": p.operation.value,
                    "applied": p.applied,
                    "improvement": p.expected_improvement,
                }
                for p in self._proposals[-5:]
            ],
        }
