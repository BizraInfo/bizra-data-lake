"""
Hierarchical Reasoning Model — The Core Engine

The unified orchestrator that creates a multi-dimensional cognitive space
where hierarchical depth and autopoietic depth intersect.

Each level runs its own complete autopoietic cycle (observe → generate →
explore → filter → validate → implement → integrate → learn) while
coordinating with adjacent levels through the CrossLevelBridge.

Key Dynamics:
  - Learning Cascade: Improvement at one level cascades to all others
  - Learning Resonance: Cross-level events that accelerate learning
  - Compound Learning Rate: Small improvements at many levels compound
  - SNR Gradient: Level-specific quality thresholds
  - Meta-Autopoiesis: Level N optimizes the hierarchy itself

HRM PDF: "The fusion creates a system simultaneously more capable
and more adaptive."

GoT PDF: "The autopoietic cognitive architecture does not process
information — it BECOMES information."

Standing on Giants:
  - Maturana & Varela (1980) — Autopoiesis
  - Simon (1962) — Hierarchical decomposition
  - Friston (2010) — Free Energy Principle
  - Brooks (1986) — Subsumption Architecture
  - Shannon (1948) — Information Theory
  - Boyd (1976) — OODA Loop
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)
from core.hrm.abstraction_levels import (
    AbstractionLevel,
    BridgeNodeType,
    LevelConfig,
    default_level_configs,
    HRM_SNR_GRADIENT,
)
from core.hrm.cross_level_bridge import (
    CrossLevelBridge,
    PropagationDirection,
)
from core.hrm.meta_level import (
    MetaAutopoieticLevel,
    MetaObservation,
    MetaProposal,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & RESULT TYPES
# ═══════════════════════════════════════════════════════════════════════════════


class HRMStatus(str, Enum):
    """Status of the Hierarchical Reasoning Model."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CONVERGED = "converged"


@dataclass
class HRMConfig:
    """
    Configuration for the Hierarchical Reasoning Model.

    Controls which levels are active, meta-learning, cascade dynamics,
    and synchronization frequency.
    """

    # Level configuration
    level_configs: List[LevelConfig] = field(default_factory=default_level_configs)

    # Meta-autopoietic level
    enable_meta_level: bool = True
    meta_observation_interval: int = 3  # Every N cycles

    # Learning cascade
    cascade_factor: float = 0.8  # How much learning transfers up/down
    cascade_decay: float = 0.9  # Decay per level distance

    # Synchronization
    sync_interval_cycles: int = 5  # How often to synchronize all levels

    # Convergence
    max_cycles: int = 50
    convergence_threshold: float = 0.01  # Min improvement to continue

    # Constitutional alignment
    ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD  # 0.95
    snr_floor: float = UNIFIED_SNR_THRESHOLD  # 0.85

    @property
    def active_levels(self) -> List[AbstractionLevel]:
        """Which levels are configured."""
        return [lc.level for lc in self.level_configs]


@dataclass
class LevelCycleResult:
    """Result of running one autopoietic cycle at a specific level."""

    level: AbstractionLevel
    cycle_number: int = 0
    snr_score: float = 0.0
    hypotheses_generated: int = 0
    hypotheses_validated: int = 0
    insights_discovered: int = 0
    learning_delta: float = 0.0
    duration_ms: float = 0.0
    bridge_node_type: BridgeNodeType = BridgeNodeType.INTRA_LEVEL

    @property
    def success(self) -> bool:
        """Did this level produce meaningful results?"""
        return self.snr_score >= HRM_SNR_GRADIENT.get(self.level, UNIFIED_SNR_THRESHOLD)


@dataclass
class HRMCycleResult:
    """Result of a complete hierarchical reasoning cycle."""

    cycle_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    cycle_number: int = 0
    status: HRMStatus = HRMStatus.COMPLETED

    # Per-level results
    level_results: Dict[AbstractionLevel, LevelCycleResult] = field(
        default_factory=dict
    )

    # Cross-level dynamics
    bridge_messages_sent: int = 0
    resonance_detected: bool = False
    cascade_events: int = 0

    # Compound metrics
    compound_snr: float = 0.0
    compound_learning_delta: float = 0.0

    # Meta-operations
    meta_observation: Optional[MetaObservation] = None
    meta_proposal: Optional[MetaProposal] = None

    # Timing
    total_duration_ms: float = 0.0

    @property
    def levels_passed(self) -> int:
        """How many levels achieved their SNR target."""
        return sum(1 for lr in self.level_results.values() if lr.success)

    @property
    def all_levels_passed(self) -> bool:
        """Did all levels meet their SNR threshold?"""
        return self.levels_passed == len(self.level_results)


# ═══════════════════════════════════════════════════════════════════════════════
# HIERARCHICAL REASONING MODEL — The Unified Engine
# ═══════════════════════════════════════════════════════════════════════════════


class HierarchicalReasoningModel:
    """
    Multi-level cognitive architecture with nested autopoietic loops.

    Creates a hierarchy of autopoietic learning systems, each operating
    at a specific abstraction level, coordinated through cross-level
    integration mechanisms, and self-optimized by meta-autopoiesis.

    Usage:
        from core.hrm import HierarchicalReasoningModel, HRMConfig

        hrm = HierarchicalReasoningModel()
        result = hrm.run_cycle({"context": "anomaly detected"})
        print(f"Compound SNR: {result.compound_snr:.4f}")
        print(f"Resonance: {result.resonance_detected}")

    The system does not process information — it BECOMES information.
    Each cycle transforms the system into a more faithful representation
    of the reality it seeks to understand.
    """

    VERSION = "1.0.0"
    CODENAME = "Ascending Spiral"

    def __init__(
        self,
        config: Optional[HRMConfig] = None,
        bridge: Optional[CrossLevelBridge] = None,
        meta_level: Optional[MetaAutopoieticLevel] = None,
    ):
        self._config = config or HRMConfig()
        self._bridge = bridge or CrossLevelBridge()
        self._meta_level = (
            meta_level
            if meta_level is not None
            else (MetaAutopoieticLevel() if self._config.enable_meta_level else None)
        )

        # Level state tracking
        self._level_states: Dict[AbstractionLevel, Dict[str, Any]] = {}
        for lc in self._config.level_configs:
            self._level_states[lc.level] = {
                "snr_score": 0.0,
                "cycle_count": 0,
                "learning_velocity": 0.0,
                "cumulative_learning": 0.0,
                "active_hypotheses": [],
                "insights": [],
                "snr_scores": [],
            }

        # Global state
        self._cycle_count = 0
        self._status = HRMStatus.IDLE
        self._history: List[HRMCycleResult] = []
        self._total_resonance_events = 0

    # ─── Primary API ───────────────────────────────────────────────────

    def run_cycle(
        self,
        observation: Optional[Dict[str, Any]] = None,
    ) -> HRMCycleResult:
        """
        Execute one complete hierarchical reasoning cycle.

        Runs autopoietic cycle at each level (bottom-up), propagates
        insights through the bridge, detects resonance, and optionally
        triggers meta-level observation.

        Returns a comprehensive HRMCycleResult.
        """
        start_time = time.time()
        self._status = HRMStatus.RUNNING
        self._cycle_count += 1

        result = HRMCycleResult(
            cycle_number=self._cycle_count,
            status=HRMStatus.RUNNING,
        )

        # Phase 1: Run each level's autopoietic cycle (bottom-up)
        for level_config in sorted(
            self._config.level_configs, key=lambda lc: int(lc.level)
        ):
            level = level_config.level
            level_result = self._run_level_cycle(level_config, observation or {})
            result.level_results[level] = level_result

            # Propagate insights upward via bridge
            if level_result.insights_discovered > 0:
                messages = self._bridge.propagate_hypothesis(
                    hypothesis={
                        "source": level.name,
                        "insights": level_result.insights_discovered,
                        "snr": level_result.snr_score,
                        "confidence": level_result.snr_score,
                    },
                    source_level=level,
                    direction=PropagationDirection.UPWARD,
                    confidence=level_result.snr_score,
                )
                result.bridge_messages_sent += len(messages)

        # Phase 2: Learning cascade
        cascade_count = self._cascade_learning(result.level_results)
        result.cascade_events = cascade_count

        # Phase 3: Detect resonance
        result.resonance_detected = self._detect_resonance(result.level_results)
        if result.resonance_detected:
            self._total_resonance_events += 1

        # Phase 4: Synchronize if interval reached
        if self._cycle_count % self._config.sync_interval_cycles == 0:
            sync_result = self._bridge.synchronize_integration(self._level_states)
            # Feed sync results to learning
            for level in self._level_states:
                self._level_states[level]["sync_quality"] = sync_result.sync_quality

        # Phase 5: Meta-autopoietic observation
        if (
            self._meta_level
            and self._cycle_count % self._config.meta_observation_interval == 0
        ):
            bridge_metrics = self._bridge.get_bridge_metrics()
            meta_obs = self._meta_level.observe_hierarchy(
                self._level_states, bridge_metrics
            )
            result.meta_observation = meta_obs

            # Propose and potentially apply modification
            proposal = self._meta_level.propose_modification(meta_obs)
            if proposal:
                result.meta_proposal = proposal
                # Apply if evaluated positively
                self._meta_level.apply_modification(
                    proposal,
                    self._bridge._boundaries,
                    self._config.level_configs,
                )

        # Compute compound metrics
        result.compound_snr = self._compute_compound_snr(result.level_results)
        result.compound_learning_delta = self._compute_compound_learning(
            result.level_results
        )

        # Finalize
        result.total_duration_ms = (time.time() - start_time) * 1000
        result.status = HRMStatus.COMPLETED
        self._status = HRMStatus.COMPLETED
        self._history.append(result)

        logger.info(
            "HRM Cycle %d complete: compound_snr=%.4f, resonance=%s, "
            "levels_passed=%d/%d, duration=%.1fms",
            self._cycle_count,
            result.compound_snr,
            result.resonance_detected,
            result.levels_passed,
            len(result.level_results),
            result.total_duration_ms,
        )

        return result

    def run_campaign(
        self,
        observation: Optional[Dict[str, Any]] = None,
        max_cycles: Optional[int] = None,
    ) -> List[HRMCycleResult]:
        """
        Run multiple cycles until convergence or max_cycles reached.

        Returns list of all cycle results.
        """
        max_c = max_cycles or self._config.max_cycles
        results = []
        prev_snr = 0.0

        for _ in range(max_c):
            result = self.run_cycle(observation)
            results.append(result)

            # Check convergence
            improvement = abs(result.compound_snr - prev_snr)
            if (
                improvement < self._config.convergence_threshold
                and self._cycle_count > 3
            ):
                result.status = HRMStatus.CONVERGED
                self._status = HRMStatus.CONVERGED
                logger.info(
                    "HRM converged at cycle %d (improvement=%.4f < %.4f)",
                    self._cycle_count,
                    improvement,
                    self._config.convergence_threshold,
                )
                break

            prev_snr = result.compound_snr

        return results

    # ─── Internal: Level Cycle ─────────────────────────────────────────

    def _run_level_cycle(
        self,
        level_config: LevelConfig,
        observation: Dict[str, Any],
    ) -> LevelCycleResult:
        """
        Run one autopoietic cycle at a specific level.

        This is a simulation of the full 8-stage RDVE pipeline
        operating at the level's abstraction granularity. In production,
        this would delegate to an actual AutopoieticLoop instance.
        """
        level = level_config.level
        start = time.time()
        state = self._level_states.get(level, {})

        # Simulate the 8-stage autopoietic cycle
        # Stage 1-2: Observe & Generate
        base_quality = 0.7 + (level_config.learning_rate_factor * 0.15)

        # Factor in cumulative learning
        cumulative = state.get("cumulative_learning", 0.0)
        quality_boost = min(cumulative * 0.05, 0.15)

        # Factor in level-specific noise tolerance
        noise_factor = 1.0 - (level_config.noise_tolerance * 0.2)

        # Compute cycle SNR
        snr = min(
            1.0,
            base_quality + quality_boost + noise_factor * 0.05,
        )

        # Stage 3-4: Explore & Filter
        hypotheses_gen = max(1, int(level_config.max_hypotheses * 0.6))
        hypotheses_valid = max(1, int(hypotheses_gen * snr))

        # Stage 5-7: Validate, Implement, Integrate
        insights = max(0, hypotheses_valid - int(hypotheses_gen * 0.4))

        # Stage 8: Learn
        prev_scores = state.get("snr_scores", [])
        if prev_scores:
            prev_avg = sum(prev_scores[-5:]) / len(prev_scores[-5:])
            learning_delta = snr - prev_avg
        else:
            learning_delta = 0.0

        # Update state
        state["snr_score"] = snr
        state["cycle_count"] = state.get("cycle_count", 0) + 1
        state["learning_velocity"] = learning_delta
        state["cumulative_learning"] = cumulative + max(0, learning_delta)
        state["snr_scores"] = (prev_scores + [snr])[-20:]  # Keep last 20
        state["active_hypotheses"] = [
            f"hyp_{level.name}_{i}" for i in range(hypotheses_valid)
        ]
        state["insights"] = [f"insight_{level.name}_{i}" for i in range(insights)]
        self._level_states[level] = state

        # Determine bridge node type
        bridge_type = BridgeNodeType.INTRA_LEVEL
        if insights > 3:
            bridge_type = BridgeNodeType.HUB
        elif learning_delta > 0.05:
            bridge_type = BridgeNodeType.BRIDGE

        duration = (time.time() - start) * 1000

        return LevelCycleResult(
            level=level,
            cycle_number=state["cycle_count"],
            snr_score=round(snr, 4),
            hypotheses_generated=hypotheses_gen,
            hypotheses_validated=hypotheses_valid,
            insights_discovered=insights,
            learning_delta=round(learning_delta, 4),
            duration_ms=round(duration, 2),
            bridge_node_type=bridge_type,
        )

    # ─── Internal: Learning Cascade ────────────────────────────────────

    def _cascade_learning(
        self,
        level_results: Dict[AbstractionLevel, LevelCycleResult],
    ) -> int:
        """
        Cascade learning across levels.

        Golden Gem: The Compound Learning Rate
          When Level 0 improves by 10%, Level 1 receives higher-quality
          patterns (improving by 8%), which improves Level 2 by 6%, and
          Level 3 by 4%. Small improvements at many levels compound into
          large system-level improvements.
        """
        cascade_count = 0
        levels = sorted(level_results.keys())

        for i, level in enumerate(levels):
            delta = level_results[level].learning_delta
            if delta <= 0:
                continue

            # Cascade upward
            for j in range(i + 1, len(levels)):
                target = levels[j]
                distance = j - i
                cascaded = delta * (self._config.cascade_decay**distance)

                if cascaded > 0.001:
                    state = self._level_states.get(target, {})
                    state["cumulative_learning"] = (
                        state.get("cumulative_learning", 0.0)
                        + cascaded * self._config.cascade_factor
                    )
                    self._level_states[target] = state
                    cascade_count += 1

            # Cascade downward (with less force)
            for j in range(i - 1, -1, -1):
                target = levels[j]
                distance = i - j
                cascaded = (
                    delta
                    * (self._config.cascade_decay**distance)
                    * 0.5  # Downward cascade is weaker
                )

                if cascaded > 0.001:
                    state = self._level_states.get(target, {})
                    state["cumulative_learning"] = (
                        state.get("cumulative_learning", 0.0)
                        + cascaded * self._config.cascade_factor
                    )
                    self._level_states[target] = state
                    cascade_count += 1

        return cascade_count

    # ─── Internal: Resonance Detection ─────────────────────────────────

    def _detect_resonance(
        self,
        level_results: Dict[AbstractionLevel, LevelCycleResult],
    ) -> bool:
        """
        Detect learning resonance across levels.

        Hidden Pattern: The Learning Resonance
          Occasionally, learning at multiple levels resonates — reinforcing
          each other in ways that accelerate learning across the entire
          hierarchy. Resonance is detected when multiple levels show
          simultaneous positive learning deltas.

        Resonance is the learning accelerator the system actively cultivates.
        """
        positive_deltas = [
            lr.learning_delta
            for lr in level_results.values()
            if lr.learning_delta > 0.01
        ]

        # Resonance requires at least 3 levels improving simultaneously
        if len(positive_deltas) >= 3:
            # Additional check: deltas should be correlated (similar magnitude)
            mean_delta = sum(positive_deltas) / len(positive_deltas)
            variance = sum((d - mean_delta) ** 2 for d in positive_deltas) / len(
                positive_deltas
            )
            coefficient_of_variation = (variance**0.5) / max(mean_delta, 0.001)

            # Low CV = high correlation = resonance
            if coefficient_of_variation < 1.0:
                logger.info(
                    "Learning RESONANCE detected: %d levels improving "
                    "simultaneously (CV=%.3f)",
                    len(positive_deltas),
                    coefficient_of_variation,
                )
                return True

        return False

    # ─── Internal: Compound Metrics ────────────────────────────────────

    def _compute_compound_snr(
        self,
        level_results: Dict[AbstractionLevel, LevelCycleResult],
    ) -> float:
        """
        Compute compound SNR across all levels.

        Weighted by level importance (higher levels weighted more
        because they make higher-stakes decisions).
        """
        if not level_results:
            return 0.0

        # Level weights: L0=0.10, L1=0.15, L2=0.20, L3=0.25, LN=0.30
        weights = {
            AbstractionLevel.PERCEPTUAL: 0.10,
            AbstractionLevel.OPERATIONAL: 0.15,
            AbstractionLevel.TACTICAL: 0.20,
            AbstractionLevel.STRATEGIC: 0.25,
            AbstractionLevel.META_COGNITIVE: 0.30,
        }

        weighted_sum = 0.0
        total_weight = 0.0
        for level, lr in level_results.items():
            w = weights.get(level, 0.15)
            weighted_sum += lr.snr_score * w
            total_weight += w

        return round(weighted_sum / max(total_weight, 0.001), 4)

    def _compute_compound_learning(
        self,
        level_results: Dict[AbstractionLevel, LevelCycleResult],
    ) -> float:
        """
        Compute compound learning delta across all levels.

        Golden Gem: Compound Learning Rate — small improvements at
        many levels compound into large system-level improvements.
        """
        if not level_results:
            return 0.0

        deltas = [lr.learning_delta for lr in level_results.values()]
        positive = [d for d in deltas if d > 0]

        if not positive:
            return 0.0

        # Compound: product of (1 + delta) - 1
        compound = 1.0
        for d in positive:
            compound *= 1.0 + d

        return round(compound - 1.0, 4)

    # ─── Public API: Status & Telemetry ────────────────────────────────

    def get_level_state(self, level: AbstractionLevel) -> Dict[str, Any]:
        """Get the current state of a specific level."""
        return self._level_states.get(level, {})

    def get_hierarchy_status(self) -> Dict[str, Any]:
        """Get comprehensive hierarchy status."""
        return {
            "version": self.VERSION,
            "codename": self.CODENAME,
            "status": self._status.value,
            "cycle_count": self._cycle_count,
            "total_resonance_events": self._total_resonance_events,
            "levels": {
                level.name: {
                    "snr_score": state.get("snr_score", 0),
                    "cycle_count": state.get("cycle_count", 0),
                    "learning_velocity": state.get("learning_velocity", 0),
                    "cumulative_learning": state.get("cumulative_learning", 0),
                }
                for level, state in self._level_states.items()
            },
            "bridge_metrics": self._bridge.get_bridge_metrics(),
            "meta_level": (self._meta_level.get_status() if self._meta_level else None),
            "compound_snr_trajectory": [r.compound_snr for r in self._history[-10:]],
        }

    def get_compound_learning_rate(self) -> float:
        """Get the current compound learning rate across all levels."""
        if not self._history:
            return 0.0
        return self._history[-1].compound_learning_delta

    def get_snr_trajectory(self) -> List[float]:
        """Get compound SNR trajectory across cycles."""
        return [r.compound_snr for r in self._history]

    def get_improvement_rate(self) -> float:
        """Get rate of improvement over recent cycles."""
        trajectory = self.get_snr_trajectory()
        if len(trajectory) < 2:
            return 0.0
        recent = trajectory[-5:]
        if len(recent) < 2:
            return 0.0
        return round(recent[-1] - recent[0], 4)
