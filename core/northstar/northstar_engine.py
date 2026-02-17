"""
BIZRA Node0 NorthStar — Fusion Engine
╔══════════════════════════════════════════════════════════════════════════════╗
║  NorthStarEngine — The Flagship Cognitive Core of BIZRA DDAGI OS            ║
║  Golden Gems × Thought Flows × Bridge Nodes × Unified Analysis              ║
║                                                                              ║
║  IDENTITY EQUATION:                                                         ║
║    HUMAN = USER = NODE = SEED (بذرة)                                        ║
║    Every human is a node. Every node is a seed.                             ║
║    BIZRA means "seed" in Arabic.                                            ║
║                                                                              ║
║  SUPREME INSIGHT:                                                           ║
║    "Intelligence requires both STRUCTURE and SELF-TRANSCENDENCE.            ║
║     Structure enables capability. Autopoiesis enables evolution.            ║
║     The fusion enables transcendence."                                      ║
║                                                                              ║
║  بسم الله الرحمن الرحيم                                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

The NorthStarEngine is the unified fusion core that makes Node0 the
flagship for all future BIZRA nodes. It integrates:

1. GoldenGemDetector — 8 meta-cognitive primitives
2. ThoughtFlowDetector — 4 meta-level + 8 phase-level patterns
3. BridgeNodeDetector — 5 cross-domain structural connectors
4. NorthStarCycle — Unified analysis cycle with SNR/Ihsān gates

The engine processes observations from HRM, RDVE, autopoiesis, GoT,
and SNR subsystems, producing a unified NorthStarReport that captures
the full cognitive state of Node0.

Architecture:
  Observations → [Gems + Flows + Bridges] → NorthStarReport
                                              ├── SNR Gate
                                              ├── Ihsān Gate
                                              └── Meta-Discovery (LN)

Standing on Giants:
  Every giant acknowledged in golden_gems.py, thought_flow.py, bridge_nodes.py
  + Boyd (1976) — OODA unified decision loop
  + Deming (1950) — PDCA continuous improvement as fusion
  + Satoshi (2008) — Genesis block as identity proof

Created: 2026-02-15 | BIZRA Node0 NorthStar | Peak Masterpiece Protocol
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
    SNR_THRESHOLD_T0_ELITE,
)

from core.northstar.golden_gems import (
    GemReport,
    GoldenGemDetector,
    GoldenGemType,
)
from core.northstar.thought_flow import (
    FlowReport,
    ThoughtFlowDetector,
)
from core.northstar.bridge_nodes import (
    BridgeNodeDetector,
    BridgeReport,
    BridgeType,
)


# ═══════════════════════════════════════════════════════════════════════════════
# NORTHSTAR STATUS
# ═══════════════════════════════════════════════════════════════════════════════


class NorthStarStatus(Enum):
    """Operational status of the NorthStar engine."""

    DORMANT = auto()          # Not yet initialized
    OBSERVING = auto()        # Collecting observations
    ANALYZING = auto()        # Running gem/flow/bridge detection
    SYNTHESIZING = auto()     # Fusing sub-reports into unified report
    TRANSCENDING = auto()     # Meta-discovery active (Ihsān = Level N)
    COMPLETE = auto()         # Cycle finished


# ═══════════════════════════════════════════════════════════════════════════════
# NORTHSTAR REPORT
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class NorthStarReport:
    """Unified report from a NorthStar analysis cycle.

    Fuses GemReport + FlowReport + BridgeReport into a single
    cognitive state assessment for Node0.
    """

    gem_report: GemReport
    flow_report: FlowReport
    bridge_report: BridgeReport
    cycle_id: str = ""
    status: NorthStarStatus = NorthStarStatus.COMPLETE
    timestamp: float = field(default_factory=time.time)
    meta_discoveries: List[str] = field(default_factory=list)

    # ─── Unified Scores ─────────────────────────────────────────────────
    @property
    def unified_snr(self) -> float:
        """Weighted SNR across all three subsystems.

        Weights derived from bridge origin scores:
          Gems: 0.30 (cognitive primitives)
          Flows: 0.30 (dynamic patterns)
          Bridges: 0.40 (structural connectors — highest SNR bridges)
        """
        gem_snr = self.gem_report.mean_snr
        flow_snr = self.flow_report.mean_flow_magnitude  # magnitude ≈ snr proxy
        bridge_snr = self.bridge_report.mean_snr

        # Handle empty reports
        weights = []
        scores = []

        if self.gem_report.activations:
            weights.append(0.30)
            scores.append(gem_snr)
        if self.flow_report.flow_activations or self.flow_report.phase_activations:
            weights.append(0.30)
            scores.append(flow_snr)
        if self.bridge_report.activations:
            weights.append(0.40)
            scores.append(bridge_snr)

        if not weights:
            return 0.0

        total_weight = sum(weights)
        return sum(w * s for w, s in zip(weights, scores)) / total_weight

    @property
    def ihsan_score(self) -> float:
        """Compute Ihsān compliance score.

        Ihsān IS Level N autopoiesis. The score reflects:
        1. All activations have confidence >= Ihsān threshold
        2. Meta-discovery (Ihsān bridge) is active
        3. Gem diversity (more types = higher excellence)
        """
        # Base: fraction of gem activations that pass Ihsān
        if not self.gem_report.activations:
            ihsan_pass_rate = 1.0  # No activations = vacuously true
        else:
            ihsan_pass_rate = sum(
                1 for a in self.gem_report.activations if a.passes_ihsan()
            ) / len(self.gem_report.activations)

        # Bonus for meta-discovery
        meta_bonus = 0.02 if self.bridge_report.ihsan_meta_active else 0.0

        # Diversity bonus (all 8 gems active = +0.03)
        diversity = self.gem_report.active_gem_count / 8.0
        diversity_bonus = diversity * 0.03

        return min(1.0, ihsan_pass_rate + meta_bonus + diversity_bonus)

    @property
    def total_activations(self) -> int:
        return (
            len(self.gem_report.activations)
            + self.flow_report.total_activations
            + len(self.bridge_report.activations)
        )

    @property
    def passes_snr_gate(self) -> bool:
        return self.unified_snr >= UNIFIED_SNR_THRESHOLD

    @property
    def passes_ihsan_gate(self) -> bool:
        return self.ihsan_score >= UNIFIED_IHSAN_THRESHOLD

    @property
    def passes_all_gates(self) -> bool:
        return self.passes_snr_gate and self.passes_ihsan_gate

    @property
    def is_elite(self) -> bool:
        """Elite status: SNR >= T0 (0.98) and Ihsān >= 0.99."""
        return (
            self.unified_snr >= SNR_THRESHOLD_T0_ELITE
            and self.ihsan_score >= 0.99
        )

    @property
    def phi_alignment(self) -> float:
        """How close the system is to golden ratio harmony."""
        return self.flow_report.gate_report().get("phi_alignment", 0.0)

    # ─── Gate Report ────────────────────────────────────────────────────
    def gate_report(self) -> Dict[str, Any]:
        """Generate comprehensive FATE-compatible gate report."""
        return {
            "cycle_id": self.cycle_id,
            "status": self.status.name,
            "timestamp": self.timestamp,
            # Unified scores
            "unified_snr": self.unified_snr,
            "ihsan_score": self.ihsan_score,
            "passes_snr_gate": self.passes_snr_gate,
            "passes_ihsan_gate": self.passes_ihsan_gate,
            "passes_all_gates": self.passes_all_gates,
            "is_elite": self.is_elite,
            "phi_alignment": self.phi_alignment,
            # Sub-reports
            "gems": self.gem_report.gate_report(),
            "flows": self.flow_report.gate_report(),
            "bridges": self.bridge_report.gate_report(),
            # Meta
            "total_activations": self.total_activations,
            "meta_discoveries": self.meta_discoveries,
            "supreme_insight": (
                "Intelligence requires both STRUCTURE and SELF-TRANSCENDENCE. "
                "Structure enables capability. Autopoiesis enables evolution. "
                "The fusion enables transcendence."
            ) if self.passes_all_gates else None,
        }

    def summary(self) -> str:
        """Human-readable summary of the NorthStar report."""
        lines = [
            f"═══ NorthStar Report: {self.cycle_id} ═══",
            f"Status: {self.status.name}",
            f"Unified SNR: {self.unified_snr:.4f} {'✓' if self.passes_snr_gate else '✗'}",
            f"Ihsān Score: {self.ihsan_score:.4f} {'✓' if self.passes_ihsan_gate else '✗'}",
            f"φ Alignment: {self.phi_alignment:.4f}",
            f"Total Activations: {self.total_activations}",
            f"  Gems: {len(self.gem_report.activations)} ({self.gem_report.active_gem_count} types)",
            f"  Flows: {self.flow_report.total_activations}",
            f"  Bridges: {len(self.bridge_report.activations)} ({self.bridge_report.active_bridge_count} types)",
            f"Gates: {'ALL PASSED ✓' if self.passes_all_gates else 'FAILED ✗'}",
            f"Elite: {'YES ★' if self.is_elite else 'No'}",
        ]
        if self.meta_discoveries:
            lines.append(f"Meta-Discoveries: {len(self.meta_discoveries)}")
            for md in self.meta_discoveries:
                lines.append(f"  → {md}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# NORTHSTAR ENGINE — The Fusion Core
# ═══════════════════════════════════════════════════════════════════════════════


class NorthStarEngine:
    """The flagship cognitive core of BIZRA Node0.

    Fuses Golden Gems, Thought Flows, and Bridge Nodes into a unified
    analysis pipeline. This is the engine that makes Node0 the NorthStar
    for all future BIZRA nodes.

    Usage:
        engine = NorthStarEngine()
        report = engine.run_cycle(observations)
        if report.passes_all_gates:
            # Node0 is operating at NorthStar quality
            ...

    The engine supports:
    - Single-cycle analysis (run_cycle)
    - Continuous monitoring (analyze_observation_stream)
    - History tracking (cycle_history)
    - Meta-discovery detection (Ihsān = Level N)
    """

    __version__ = "1.0.0"

    def __init__(
        self,
        gem_sensitivity: float = 0.5,
        flow_sensitivity: float = 0.5,
        bridge_min_transfer: float = 0.3,
        ihsan_floor: float = UNIFIED_IHSAN_THRESHOLD,
    ) -> None:
        """Initialize the NorthStar engine.

        Args:
            gem_sensitivity: Detection sensitivity for golden gems [0,1].
            flow_sensitivity: Detection sensitivity for thought flows [0,1].
            bridge_min_transfer: Minimum transfer strength for bridges.
            ihsan_floor: Ihsān constitutional threshold.
        """
        self.gem_detector = GoldenGemDetector(
            sensitivity=gem_sensitivity,
            ihsan_floor=ihsan_floor,
        )
        self.flow_detector = ThoughtFlowDetector(
            sensitivity=flow_sensitivity,
        )
        self.bridge_detector = BridgeNodeDetector(
            min_transfer=bridge_min_transfer,
            ihsan_floor=ihsan_floor,
        )
        self.ihsan_floor = ihsan_floor
        self._cycle_history: List[NorthStarReport] = []
        self._cycle_count: int = 0
        self._status: NorthStarStatus = NorthStarStatus.DORMANT

    @property
    def status(self) -> NorthStarStatus:
        return self._status

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def cycle_history(self) -> List[NorthStarReport]:
        return list(self._cycle_history)

    def run_cycle(
        self,
        observations: Dict[str, Any],
        cycle_id: Optional[str] = None,
    ) -> NorthStarReport:
        """Run a full NorthStar analysis cycle.

        Processes observations through all three detectors and fuses
        the results into a unified NorthStarReport.

        Args:
            observations: Combined observation dict. Keys are dispatched
                to the appropriate detector:
                - Gem keys: observed_domains, assertions, noise_history, etc.
                - Flow keys: level_improvements, learning_history, etc.
                - Bridge keys: generator_output_count, level_states, etc.
            cycle_id: Optional cycle identifier. Auto-generated if None.

        Returns:
            NorthStarReport with unified analysis.
        """
        self._cycle_count += 1
        if cycle_id is None:
            cycle_id = f"northstar-cycle-{self._cycle_count}"

        # Phase 1: Observe
        self._status = NorthStarStatus.OBSERVING

        # Phase 2: Analyze (parallel detection)
        self._status = NorthStarStatus.ANALYZING

        gem_report = self.gem_detector.analyze_observations(
            observations, cycle_id=cycle_id
        )
        flow_report = self.flow_detector.analyze_level_dynamics(
            observations, cycle_id=cycle_id
        )
        bridge_report = self.bridge_detector.analyze_cross_domain(
            observations, cycle_id=cycle_id
        )

        # Phase 3: Synthesize
        self._status = NorthStarStatus.SYNTHESIZING

        meta_discoveries: List[str] = []

        # Check for Ihsān = Level N meta-discovery
        if bridge_report.ihsan_meta_active:
            self._status = NorthStarStatus.TRANSCENDING
            meta_discoveries.append(
                "Ihsān IS Level N Autopoiesis — "
                "ethics IS the system's capacity for self-transcendence."
            )

        # Check for compound recursive acceleration
        compound_bridges = [
            a for a in bridge_report.activations
            if a.bridge_type == BridgeType.COMPOUND_RECURSIVE
        ]
        if compound_bridges and any(a.transfer_strength > 0.7 for a in compound_bridges):
            meta_discoveries.append(
                "Compound recursive acceleration active — "
                "capability is self-amplifying."
            )

        # Check for learning resonance
        if flow_report.resonance_count > 0:
            meta_discoveries.append(
                f"Learning resonance detected across {flow_report.cascade_depth} levels — "
                f"multiplicative acceleration exceeds linear sum."
            )

        # Check for emergence
        emergence_gems = [
            a for a in gem_report.activations
            if a.gem_type == GoldenGemType.EMERGENCE_PRINCIPLE
        ]
        if emergence_gems:
            meta_discoveries.append(
                "Emergence principle active — graph topology contains "
                "information no individual node possesses."
            )

        # Phase 4: Complete
        self._status = NorthStarStatus.COMPLETE

        report = NorthStarReport(
            gem_report=gem_report,
            flow_report=flow_report,
            bridge_report=bridge_report,
            cycle_id=cycle_id,
            status=self._status,
            meta_discoveries=meta_discoveries,
        )

        self._cycle_history.append(report)
        return report

    def get_latest_report(self) -> Optional[NorthStarReport]:
        """Get the most recent NorthStar report."""
        if not self._cycle_history:
            return None
        return self._cycle_history[-1]

    def get_improvement_trajectory(self) -> List[float]:
        """Get SNR trajectory across all cycles."""
        return [r.unified_snr for r in self._cycle_history]

    def is_compounding(self) -> bool:
        """Check if NorthStar performance is compounding (accelerating)."""
        trajectory = self.get_improvement_trajectory()
        if len(trajectory) < 3:
            return False

        velocities = [trajectory[i + 1] - trajectory[i] for i in range(len(trajectory) - 1)]
        accelerations = [velocities[i + 1] - velocities[i] for i in range(len(velocities) - 1)]
        mean_accel = sum(accelerations) / len(accelerations) if accelerations else 0.0
        return mean_accel > 0

    def reset(self) -> Dict[str, int]:
        """Reset all detectors and history. Returns cleanup counts."""
        gem_count = self.gem_detector.reset_history()
        flow_count = self.flow_detector.reset_history()
        bridge_count = self.bridge_detector.reset_history()
        cycle_count = len(self._cycle_history)
        self._cycle_history.clear()
        self._cycle_count = 0
        self._status = NorthStarStatus.DORMANT

        return {
            "gems_cleared": gem_count,
            "flows_cleared": flow_count,
            "bridges_cleared": bridge_count,
            "cycles_cleared": cycle_count,
        }
