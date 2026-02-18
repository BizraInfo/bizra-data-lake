"""
Hierarchical Reasoning Model — Abstraction Levels & Level Configuration

Defines the multi-level cognitive hierarchy where each level operates at
a specific abstraction granularity with its own autopoietic learning cycle.

Standing on Giants:
  - Simon (1962) — "The Architecture of Complexity" (hierarchical systems)
  - Maturana & Varela (1980) — Autopoiesis (self-producing systems)
  - Friston (2010) — Free Energy Principle (hierarchical prediction)
  - Brooks (1986) — Subsumption Architecture (layered competence)

Constitutional Alignment:
  All SNR thresholds derived from core/integration/constants.py (SSOT).
  Level-specific thresholds form a GRADIENT: lower levels tolerate more noise
  (more data), higher levels require higher signal purity (higher stakes).

  HRM PDF Table 6: Level-Specific SNR Optimization Strategies
  HRM PDF Table 2: Hierarchical Reasoning Model Abstraction Levels
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, Enum
from typing import Dict, Final, List

from core.integration.constants import (
    UNIFIED_SNR_THRESHOLD,
    SNR_THRESHOLD_T0_ELITE,
    SNR_THRESHOLD_T1_HIGH,
    SNR_THRESHOLD_T2_STANDARD,
    SNR_THRESHOLD_T3_ACCEPTABLE,
)

# ═══════════════════════════════════════════════════════════════════════════════
# ABSTRACTION LEVELS — The Cognitive Hierarchy
# ═══════════════════════════════════════════════════════════════════════════════


class AbstractionLevel(IntEnum):
    """
    Hierarchical abstraction levels from perceptual ground to meta-cognitive.

    Each level reasons at a specific granularity and temporal scale.
    Information flows bidirectionally: goals descend, evidence ascends.

    Mapping to BIZRA Four Pillars (Curry-Howard Correspondence):
      Level 0 (Perceptual) → Genesis/Sandbox (hypothesis)
      Level 1 (Operational) → Museum (verified conjecture)
      Level 2 (Tactical) → Museum→Runtime transition
      Level 3 (Strategic) → Runtime (proven theorem)
      Level N (Meta) → Adaptive Ihsān (axiom evolution)
    """

    PERCEPTUAL = 0  # Signal detection, feature extraction
    OPERATIONAL = 1  # Pattern recognition, categorization
    TACTICAL = 2  # Action selection, resource allocation
    STRATEGIC = 3  # Goal setting, planning, priorities
    META_COGNITIVE = 4  # Reasoning about reasoning itself


class TemporalScale(str, Enum):
    """Temporal scale at which each abstraction level operates."""

    IMMEDIATE = "immediate"  # L0: Real-time signal processing
    SHORT_TERM = "short_term"  # L1: Pattern recognition window
    MEDIUM_TERM = "medium_term"  # L2: Action-outcome correlation
    LONG_TERM = "long_term"  # L3: Strategic planning horizon
    EVOLUTIONARY = "evolutionary"  # LN: Architectural evolution


class BridgeNodeType(str, Enum):
    """
    GoT node types in hierarchical context.

    Bridge nodes are the highest-SNR nodes — they connect reasoning
    across multiple abstraction levels simultaneously.

    HRM PDF Table 5: GoT Node Types in Hierarchical Context
    """

    INTRA_LEVEL = "intra_level"  # Reasoning within single level
    INTER_LEVEL = "inter_level"  # Between adjacent levels
    BRIDGE = "bridge"  # Multi-scale insight (HIGHEST SNR)
    HUB = "hub"  # Integration/synthesis point
    FRONTIER = "frontier"  # Exploration boundary probe


# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL CONFIGURATION — SNR Gradient & Learning Parameters
# ═══════════════════════════════════════════════════════════════════════════════

# Level-specific SNR thresholds (the gradient)
# Lower levels tolerate more noise (more data available)
# Higher levels require higher purity (higher stakes, less data)
HRM_SNR_GRADIENT: Final[Dict[AbstractionLevel, float]] = {
    AbstractionLevel.PERCEPTUAL: UNIFIED_SNR_THRESHOLD,  # 0.85
    AbstractionLevel.OPERATIONAL: SNR_THRESHOLD_T3_ACCEPTABLE,  # 0.85
    AbstractionLevel.TACTICAL: SNR_THRESHOLD_T2_STANDARD,  # 0.90
    AbstractionLevel.STRATEGIC: SNR_THRESHOLD_T1_HIGH,  # 0.95
    AbstractionLevel.META_COGNITIVE: SNR_THRESHOLD_T0_ELITE,  # 0.98
}

# Temporal scale mapping
HRM_TEMPORAL_SCALE: Final[Dict[AbstractionLevel, TemporalScale]] = {
    AbstractionLevel.PERCEPTUAL: TemporalScale.IMMEDIATE,
    AbstractionLevel.OPERATIONAL: TemporalScale.SHORT_TERM,
    AbstractionLevel.TACTICAL: TemporalScale.MEDIUM_TERM,
    AbstractionLevel.STRATEGIC: TemporalScale.LONG_TERM,
    AbstractionLevel.META_COGNITIVE: TemporalScale.EVOLUTIONARY,
}


@dataclass(frozen=True)
class LevelConfig:
    """
    Configuration for a single hierarchical reasoning level.

    Each level has its own SNR threshold, learning rate, and noise
    tolerance — forming the SNR gradient that governs information flow.
    """

    level: AbstractionLevel
    snr_threshold: float
    temporal_scale: TemporalScale
    learning_rate_factor: float = 1.0
    noise_tolerance: float = 0.15
    max_hypotheses: int = 20
    paradox_tolerance: float = 0.3

    @property
    def level_name(self) -> str:
        """Human-readable level name."""
        return self.level.name.replace("_", " ").title()

    @property
    def level_index(self) -> int:
        """Numeric index (0-based)."""
        return int(self.level)


@dataclass
class LevelBoundary:
    """
    Boundary between adjacent abstraction levels.

    Boundaries are NOT walls — they are selectively permeable membranes
    that learn which information to pass, block, or transform.

    Golden Gem: The Permeable Boundary Principle
      Level boundaries are intelligent membranes. The membrane learns:
      which types of information should cross easily, which require
      transformation, which should be blocked.
    """

    source_level: AbstractionLevel
    target_level: AbstractionLevel
    permeability: float = 0.5  # 0.0 = sealed, 1.0 = transparent
    transform_required: bool = True  # Must information be abstracted?
    message_count: int = 0  # Telemetry: messages crossed
    blocked_count: int = 0  # Telemetry: messages blocked

    @property
    def direction(self) -> str:
        """Is this upward (evidence) or downward (goals)?"""
        if self.target_level > self.source_level:
            return "upward"
        return "downward"

    def should_pass(self, confidence: float) -> bool:
        """Decide if a message with given confidence should cross."""
        threshold = 1.0 - self.permeability
        return confidence >= threshold

    def record_crossing(self, passed: bool) -> None:
        """Record a boundary crossing attempt."""
        if passed:
            self.message_count += 1
        else:
            self.blocked_count += 1


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY — Default Level Configurations
# ═══════════════════════════════════════════════════════════════════════════════


def default_level_configs() -> List[LevelConfig]:
    """
    Create default configurations for all 5 abstraction levels.

    The SNR gradient follows the constitutional thresholds:
      L0: 0.85 (perceptual — high noise tolerance)
      L1: 0.85 (operational — pattern recognition)
      L2: 0.90 (tactical — action decisions)
      L3: 0.95 (strategic — high-stakes planning)
      LN: 0.98 (meta-cognitive — architectural decisions)

    Learning rate factors decrease with level (lower levels learn faster
    due to more data and shorter feedback loops):
      L0: 1.0, L1: 0.8, L2: 0.6, L3: 0.4, LN: 0.2
    """
    configs = []
    learning_rates = [1.0, 0.8, 0.6, 0.4, 0.2]
    noise_tolerances = [0.20, 0.18, 0.12, 0.08, 0.03]
    paradox_tolerances = [0.15, 0.20, 0.30, 0.40, 0.50]

    for level in AbstractionLevel:
        idx = int(level)
        configs.append(
            LevelConfig(
                level=level,
                snr_threshold=HRM_SNR_GRADIENT[level],
                temporal_scale=HRM_TEMPORAL_SCALE[level],
                learning_rate_factor=learning_rates[idx],
                noise_tolerance=noise_tolerances[idx],
                paradox_tolerance=paradox_tolerances[idx],
            )
        )

    return configs


def default_boundaries() -> List[LevelBoundary]:
    """
    Create default boundaries between adjacent levels.

    Permeability increases toward upper levels (higher levels are
    more selective about what enters but more generous about what exits).
    """
    levels = list(AbstractionLevel)
    boundaries = []

    for i in range(len(levels) - 1):
        # Upward boundary (evidence ascending)
        boundaries.append(
            LevelBoundary(
                source_level=levels[i],
                target_level=levels[i + 1],
                permeability=0.6 - (i * 0.1),  # Decreases upward
                transform_required=True,
            )
        )
        # Downward boundary (goals descending)
        boundaries.append(
            LevelBoundary(
                source_level=levels[i + 1],
                target_level=levels[i],
                permeability=0.7 - (i * 0.05),  # Slightly more permeable
                transform_required=True,
            )
        )

    return boundaries
