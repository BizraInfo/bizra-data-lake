"""
Network Effect Estimator — PROJECTION LOGIC (not constitutional law)
====================================================================

Classification: ESTIMATOR
Constitutional status: NONE — this module contains no invariants.
Accuracy: PROJECTED — calibrated from Node0 baseline, not measured network.

All constants in this module are EMPIRICAL DEFAULTS, not constitutional
thresholds. They MUST NOT be added to constants.py or CANONICAL_THRESHOLDS.
They change as real network data arrives. They are NOT fail-closed gates.

The distinction:
  - constants.py: UNIFIED_IHSAN_THRESHOLD = 0.95 -> constitutional, immutable
  - this file: EST_TFLOPS_PER_NODE = 5.0 -> empirical, mutable, projection

Standing on Giants: Metcalfe (1993) - Reed (1999) - Shannon (1948)

Phase 72 — Constitutional Kernel
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

# For automated auditing:
_MODULE_CLASS = "ESTIMATOR"  # Values: CONSTITUTIONAL | ESTIMATOR | UTILITY

# =========================================================================
# EMPIRICAL ESTIMATION DEFAULTS (not constitutional thresholds)
# =========================================================================
# Prefix: EST_ to distinguish from constitutional constants.
# These values are projections based on Node0 hardware profile.
# They WILL change as real network telemetry arrives.

EST_REFLEXES_PER_NODE: int = 50
EST_TFLOPS_PER_NODE: float = 5.0
EST_BASELINE_LATENCY_MS: float = 100.0
EST_COST_DECAY_RATE: float = 0.15


# ---------------------------------------------------------------------------
# Network Projection dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NetworkProjection:
    """Snapshot of projected network-wide metrics at a given node count."""

    nodes: int
    skills_available: int
    compute_tflops: float
    latency_factor: float  # < 1.0 means faster than baseline
    intelligence_density: float  # cross-domain connections per node
    cost_per_node: float  # relative cost (1.0 = single node baseline)
    timestamp: str


# ---------------------------------------------------------------------------
# NetworkEffectEstimator — pure computation, no I/O
# ---------------------------------------------------------------------------


class NetworkEffectEstimator:
    """Projects network-wide metrics as a function of node count.

    Pure computation — no I/O, no network calls. Uses conservative
    empirical constants that can be tuned as real data arrives.
    """

    def __init__(
        self,
        reflexes_per_node: int = EST_REFLEXES_PER_NODE,
        tflops_per_node: float = EST_TFLOPS_PER_NODE,
        baseline_latency_ms: float = EST_BASELINE_LATENCY_MS,
    ) -> None:
        self._reflexes = reflexes_per_node
        self._tflops = tflops_per_node
        self._baseline_latency = baseline_latency_ms

    def project(self, n_nodes: int) -> NetworkProjection:
        """Project network metrics for n_nodes participants."""
        if n_nodes < 1:
            raise ValueError("n_nodes must be >= 1")

        # Skills: linear in nodes
        skills = n_nodes * self._reflexes

        # Compute: linear sum
        compute = n_nodes * self._tflops

        # Latency: improves logarithmically
        latency_factor = 1.0 / (1.0 + math.log10(n_nodes))

        # Intelligence density: log(n) per node
        if n_nodes > 1:
            intelligence_density = math.log(n_nodes)
        else:
            intelligence_density = 1.0

        # Cost per node: decreases with shared infrastructure
        cost_per_node = 1.0 / (
            1.0 + EST_COST_DECAY_RATE * math.log(max(1, n_nodes))
        )

        return NetworkProjection(
            nodes=n_nodes,
            skills_available=skills,
            compute_tflops=round(compute, 2),
            latency_factor=round(latency_factor, 4),
            intelligence_density=round(intelligence_density, 4),
            cost_per_node=round(cost_per_node, 4),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def project_milestones(self) -> list[NetworkProjection]:
        """Standard milestone projections for dashboards."""
        milestones = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 8_000_000_000]
        return [self.project(n) for n in milestones]

    def compute_moat_metrics(self, n_nodes: int) -> dict:
        """The three-pillar moat quantification.

        Each new node brings:
        1. Hardware — compute contribution
        2. Data — knowledge graph expansion
        3. Intelligence — compiled skill library growth
        """
        proj = self.project(n_nodes)
        data_connections = n_nodes * (n_nodes - 1) // 2 if n_nodes > 1 else 0
        return {
            "hardware_contribution_tflops": proj.compute_tflops,
            "data_connections": data_connections,
            "intelligence_skills": proj.skills_available,
            "moat_score": round(
                math.log(1 + proj.compute_tflops)
                * math.log(1 + proj.skills_available)
                * proj.intelligence_density,
                4,
            ),
        }
