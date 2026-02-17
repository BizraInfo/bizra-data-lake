"""
Complexity Adapter: Maps MoE complexity tiers <-> HRM abstraction levels.

Provides a bidirectional bridge between the Mixture-of-Experts router's
complexity classification (TRIVIAL..FRONTIER) and the Hierarchical Reasoning
Model's abstraction levels (PERCEPTUAL..META_COGNITIVE), including the
level-specific SNR thresholds that govern each level's quality gate.

Mapping rationale:
  TRIVIAL   -> PERCEPTUAL     (signal detection, fast path)
  STANDARD  -> OPERATIONAL    (pattern matching, default tier)
  COMPLEX   -> TACTICAL       (action planning, resource-aware)
  EXPERT    -> STRATEGIC      (long-horizon, high-stakes)
  FRONTIER  -> META_COGNITIVE (reasoning about reasoning)

Standing on Giants:
  - Vaswani et al. (2017) -- Mixture-of-Experts routing
  - Simon (1962)         -- "The Architecture of Complexity" (near-decomposability)
  - Shannon (1948)       -- SNR quality gradient
  - Al-Ghazali (1095)    -- Ihsan / excellence as hard constraint

Constitutional Alignment:
  ALL thresholds imported from core/integration/constants.py (SSOT).

Created: 2026-02-17 | BIZRA Node0 | Cognitive Fusion Phase
"""

from __future__ import annotations

import logging
from typing import Dict, Final, Tuple

from core.integration.constants import (
    SNR_THRESHOLD_T0_ELITE,
    SNR_THRESHOLD_T1_HIGH,
    SNR_THRESHOLD_T2_STANDARD,
    UNIFIED_SNR_THRESHOLD,
)

logger = logging.getLogger(__name__)


# =============================================================================
# MAPPING TABLES
# =============================================================================

# MoE complexity class -> HRM abstraction level name
COMPLEXITY_TO_LEVEL: Final[Dict[str, str]] = {
    "TRIVIAL": "PERCEPTUAL",
    "STANDARD": "OPERATIONAL",
    "COMPLEX": "TACTICAL",
    "EXPERT": "STRATEGIC",
    "FRONTIER": "META_COGNITIVE",
}

# HRM abstraction level name -> inference expert tier
LEVEL_TO_TIER: Final[Dict[str, str]] = {
    "PERCEPTUAL": "NANO",
    "OPERATIONAL": "EDGE",
    "TACTICAL": "LOCAL",
    "STRATEGIC": "POOL",
    "META_COGNITIVE": "FRONTIER",
}

# Reverse mapping: expert tier -> HRM level name
TIER_TO_LEVEL: Final[Dict[str, str]] = {v: k for k, v in LEVEL_TO_TIER.items()}

# Level-specific SNR requirements (gradient)
# Lower levels tolerate more noise; higher levels demand purer signal.
SNR_REQUIREMENTS: Final[Dict[str, float]] = {
    "PERCEPTUAL": UNIFIED_SNR_THRESHOLD,       # 0.85
    "OPERATIONAL": UNIFIED_SNR_THRESHOLD,      # 0.85
    "TACTICAL": SNR_THRESHOLD_T2_STANDARD,     # 0.90
    "STRATEGIC": SNR_THRESHOLD_T1_HIGH,        # 0.95
    "META_COGNITIVE": SNR_THRESHOLD_T0_ELITE,  # 0.98
}

# Default values for unknown inputs
_DEFAULT_LEVEL: Final[str] = "OPERATIONAL"
_DEFAULT_TIER: Final[str] = "EDGE"
_DEFAULT_SNR: Final[float] = UNIFIED_SNR_THRESHOLD


# =============================================================================
# COMPLEXITY ADAPTER
# =============================================================================


class ComplexityAdapter:
    """
    Bidirectional adapter between MoE complexity classes and HRM levels.

    Stateless -- all mappings are derived from the module-level constants.
    Instantiate once and share across the pipeline.

    Example::

        adapter = ComplexityAdapter()
        level, snr = adapter.adapt("EXPERT")
        assert level == "STRATEGIC"
        assert snr == 0.95  # SNR_THRESHOLD_T1_HIGH
    """

    # -- primary conversion ----------------------------------------------------

    def adapt(self, complexity: str) -> Tuple[str, float]:
        """
        Map a MoE complexity class to an HRM level and its SNR requirement.

        Args:
            complexity: One of TRIVIAL, STANDARD, COMPLEX, EXPERT, FRONTIER.
                        Unknown values fall back to OPERATIONAL / base SNR.

        Returns:
            Tuple of (level_name, required_snr).
        """
        level = COMPLEXITY_TO_LEVEL.get(complexity, _DEFAULT_LEVEL)
        snr = SNR_REQUIREMENTS.get(level, _DEFAULT_SNR)

        if complexity not in COMPLEXITY_TO_LEVEL:
            logger.warning(
                "Unknown complexity class %r — defaulting to %s (SNR %.2f)",
                complexity,
                level,
                snr,
            )

        return level, snr

    # -- tier / level lookups --------------------------------------------------

    @staticmethod
    def level_to_tier(level: str) -> str:
        """
        Map an HRM abstraction level name to an expert tier.

        Args:
            level: HRM level name (e.g. "TACTICAL").

        Returns:
            Expert tier string (e.g. "LOCAL").
        """
        return LEVEL_TO_TIER.get(level, _DEFAULT_TIER)

    @staticmethod
    def tier_to_level(tier: str) -> str:
        """
        Reverse-map an expert tier back to an HRM level name.

        Args:
            tier: Expert tier string (e.g. "POOL").

        Returns:
            HRM level name (e.g. "STRATEGIC").
        """
        return TIER_TO_LEVEL.get(tier, _DEFAULT_LEVEL)

    # -- SNR lookup ------------------------------------------------------------

    @staticmethod
    def get_snr_requirement(level: str) -> float:
        """
        Return the minimum SNR threshold for a given HRM level.

        Args:
            level: HRM level name.

        Returns:
            Minimum SNR as a float.
        """
        return SNR_REQUIREMENTS.get(level, _DEFAULT_SNR)
