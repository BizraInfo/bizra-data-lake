"""
Node Value Engine — Unified KPI for Sovereign Nodes
=====================================================

Computes the five-factor composite value of a sovereign node.
READ-ONLY over SeedEngine state — no duplicate counters.

Composite = (Potential x Activation x Quality x Compounding x Synergy) ^ (1/5)

All factors normalized to [0, 1]. Geometric mean ensures:
- If any factor is 0, composite is 0
- If all factors are 1.0, composite is 1.0
- No factor can dominate through volume or age

Standing on Giants:
- Shannon (1948): Bounded information — each factor is a channel
- Deming (1986): SPC control limits — normalization prevents drift
- Kahneman (2011): System 1/2 — quality over volume

Phase 72 — Constitutional Kernel
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from core.integration.constants import (
    NODE_VALUE_ACTIVATION_REFERENCE,
    NODE_VALUE_COMPOUNDING_REFERENCE_DAYS,
    NODE_VALUE_STREAK_REFERENCE,
)
from core.sovereign.human_lifecycle import human_stage

# ---------------------------------------------------------------------------
# Snapshot dataclass — all factors bounded [0, 1]
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NodeValueSnapshot:
    """Immutable snapshot of a node's five-factor value."""

    potential: float  # SeedEngine sovereignty_score [0, 1]
    activation: float  # min(DAM / reference, 1.0) [0, 1]
    quality: float  # mean ihsan [0, 1]
    compounding: float  # asymptotic time+streak [0, 1]
    synergy: float  # network effect [0, 1]
    composite: float  # geometric mean [0, 1]
    tier: str
    human_stage: str
    timestamp: str


# ---------------------------------------------------------------------------
# NodeValueEngine — read-only view over SeedEngine
# ---------------------------------------------------------------------------


class NodeValueEngine:
    """Computes the unified KPI for a sovereign node.

    READ-ONLY over SeedEngine state. Does NOT maintain its own counters.
    Single source of truth = SeedEngine.
    """

    def __init__(
        self,
        seed_engine,
        genesis_timestamp: Optional[str] = None,
    ) -> None:
        self._engine = seed_engine
        self._genesis = genesis_timestamp or datetime.now(timezone.utc).isoformat()

    def compute(self) -> NodeValueSnapshot:
        """Compute the five-factor node value snapshot."""
        pot = self._engine.potential()

        # Factor 1: Potential (already 0-1)
        potential = pot.sovereignty_score

        # Factor 2: Activation (normalized 0-1)
        active_days = max(1, self._days_since_genesis())
        dam = pot.episodes_total / active_days
        activation = min(dam / NODE_VALUE_ACTIVATION_REFERENCE, 1.0)

        # Factor 3: Quality (already 0-1)
        ihsan_scores = self._engine._dimension_scores.get("ihsan", [])
        if len(ihsan_scores) > 0:
            window = ihsan_scores[-50:]
            quality = sum(window) / len(window)
        else:
            quality = 0.0

        # Factor 4: Compounding (normalized 0-1 via asymptotic curve)
        age_days = max(1.0, float(self._days_since_genesis()))
        time_factor = 1.0 - math.exp(-age_days / NODE_VALUE_COMPOUNDING_REFERENCE_DAYS)
        streak_factor = min(pot.streak / NODE_VALUE_STREAK_REFERENCE, 1.0)
        compounding = time_factor * (0.7 + 0.3 * streak_factor)

        # Factor 5: Synergy (0-1, pre-federation = 1.0)
        synergy = self._compute_network_synergy()

        # Composite: GEOMETRIC MEAN (5th root of product)
        factors = [potential, activation, quality, compounding, synergy]
        if all(f > 0 for f in factors):
            product = 1.0
            for f in factors:
                product *= f
            composite = product**0.2
        else:
            composite = 0.0

        return NodeValueSnapshot(
            potential=round(potential, 4),
            activation=round(activation, 4),
            quality=round(quality, 4),
            compounding=round(compounding, 4),
            synergy=round(synergy, 4),
            composite=round(composite, 4),
            tier=pot.tier,
            human_stage=human_stage(pot.sovereignty_score),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _compute_network_synergy(self) -> float:
        """Stub for pre-federation. Returns 1.0 until A2A is live."""
        return 1.0

    def _days_since_genesis(self) -> int:
        """Calculate days since genesis timestamp."""
        try:
            genesis_dt = datetime.fromisoformat(self._genesis)
            if genesis_dt.tzinfo is None:
                genesis_dt = genesis_dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            delta = now - genesis_dt
            return max(1, delta.days)
        except (ValueError, TypeError):
            return 1

    def health(self) -> dict:
        """Health check — provenance is explicit."""
        return {
            "engine": "node_value",
            "source": "seed_engine",
            "genesis": self._genesis,
            "has_federation": False,
        }
