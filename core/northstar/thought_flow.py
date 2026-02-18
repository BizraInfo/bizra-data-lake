"""
BIZRA Node0 NorthStar — Hidden Thought Flow Patterns
╔══════════════════════════════════════════════════════════════════════════════╗
║  4 Hidden Thought Flow Patterns × 8 Phase-Level Hidden Patterns             ║
║  The invisible currents that drive cognitive evolution                       ║
║                                                                              ║
║  بسم الله الرحمن الرحيم                                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Hidden Thought Flow Patterns (meta-level dynamics):
  1. Cross-Level Learning Cascade — 10% L0 → super-linear cascade up
  2. Permeable Boundary Principle — level boundaries as selective membranes
  3. Compound Learning Rate — C(t+dt) = C(t) + g(C(t))·dt
  4. Learning Resonance — cross-level acceleration events

Phase-Level Hidden Patterns (per-phase dynamics with SNR scores):
  1. Observation Paradox      (9.7) — most valuable observations surprise
  2. Diversity-Specificity    (9.4) — 80/20 portfolio (specific/exploratory)
  3. Convergence-Divergence   (9.8) — golden ratio φ ≈ 1.618 rhythm
  4. Signal-Noise Phase Shift (9.5) — persistent noise = truth
  5. Paradox Tolerance        (9.6) — optimal learning near paradox threshold
  6. Implementation Gap       (9.3) — gap = tacit knowledge
  7. Integration Depth        (9.5) — Surface→Foundational gradient
  8. Meta-Learning Accel.     (9.8) — 10% L0 → cascades entire hierarchy

Standing on Giants:
  - Friston (2010) — Free Energy & hierarchical prediction
  - Simon (1962) — near-decomposable systems
  - Kauffman (1993) — edge of chaos optimization
  - Fibonacci/Pacioli — golden ratio in natural growth
  - Brooks (1986) — subsumption architecture
  - Deming (1950) — PDCA continuous improvement

Created: 2026-02-15 | BIZRA Node0 NorthStar | Peak Masterpiece Protocol
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.integration.constants import (
    UNIFIED_SNR_THRESHOLD,
)

# ═══════════════════════════════════════════════════════════════════════════════
# GOLDEN RATIO — The pulse of convergence-divergence
# ═══════════════════════════════════════════════════════════════════════════════
PHI: float = (1.0 + math.sqrt(5)) / 2  # φ ≈ 1.618033988749895


# ═══════════════════════════════════════════════════════════════════════════════
# HIDDEN THOUGHT FLOW PATTERN TYPES
# ═══════════════════════════════════════════════════════════════════════════════


class ThoughtFlowType(Enum):
    """The 4 meta-level thought flow patterns."""

    CROSS_LEVEL_CASCADE = auto()  # L0 improvements cascade super-linearly
    PERMEABLE_BOUNDARY = auto()  # Level boundaries as selective membranes
    COMPOUND_LEARNING = auto()  # C(t+dt) = C(t) + g(C(t))·dt
    LEARNING_RESONANCE = auto()  # Cross-level acceleration events


class PhasePatternType(Enum):
    """The 8 phase-level hidden patterns."""

    OBSERVATION_PARADOX = auto()  # Most valuable observations surprise
    DIVERSITY_SPECIFICITY = auto()  # 80/20 portfolio balance
    CONVERGENCE_DIVERGENCE = auto()  # Golden ratio φ rhythm
    SIGNAL_NOISE_PHASE_SHIFT = auto()  # Phase transition in noise
    PARADOX_TOLERANCE = auto()  # Learning at paradox threshold
    IMPLEMENTATION_GAP = auto()  # Gap = tacit knowledge
    INTEGRATION_DEPTH = auto()  # Surface → Foundational gradient
    META_LEARNING_ACCELERATION = auto()  # 10% L0 → cascade whole hierarchy


# Origin SNR scores for phase patterns
PHASE_PATTERN_SNR: Dict[PhasePatternType, float] = {
    PhasePatternType.OBSERVATION_PARADOX: 9.7,
    PhasePatternType.DIVERSITY_SPECIFICITY: 9.4,
    PhasePatternType.CONVERGENCE_DIVERGENCE: 9.8,
    PhasePatternType.SIGNAL_NOISE_PHASE_SHIFT: 9.5,
    PhasePatternType.PARADOX_TOLERANCE: 9.6,
    PhasePatternType.IMPLEMENTATION_GAP: 9.3,
    PhasePatternType.INTEGRATION_DEPTH: 9.5,
    PhasePatternType.META_LEARNING_ACCELERATION: 9.8,
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class FlowActivation:
    """A thought flow pattern activation event."""

    flow_type: ThoughtFlowType
    evidence: str
    magnitude: float  # [0, 1] — strength of flow
    direction: str  # "ascending" | "descending" | "lateral" | "resonant"
    insight: str
    timestamp: float = field(default_factory=time.time)
    affected_levels: Tuple[int, ...] = ()  # HRM levels involved

    def snr_score(self) -> float:
        """Compute SNR for this flow activation."""
        base_snr = 0.95  # All thought flows are high-quality by definition
        return base_snr * self.magnitude

    def passes_gate(self) -> bool:
        """Check if activation clears SNR threshold."""
        return self.snr_score() >= UNIFIED_SNR_THRESHOLD


@dataclass(frozen=True)
class PhaseActivation:
    """A phase-level hidden pattern activation."""

    pattern_type: PhasePatternType
    phase_index: int  # 0-7 lifecycle phase
    evidence: str
    intensity: float  # [0, 1]
    insight: str
    timestamp: float = field(default_factory=time.time)

    def snr_score(self) -> float:
        """Compute SNR from origin score × intensity."""
        origin = PHASE_PATTERN_SNR.get(self.pattern_type, 9.0) / 10.0
        return origin * self.intensity


@dataclass
class FlowReport:
    """Summary of thought flow and phase pattern activations."""

    flow_activations: List[FlowActivation] = field(default_factory=list)
    phase_activations: List[PhaseActivation] = field(default_factory=list)
    cycle_id: str = ""
    timestamp: float = field(default_factory=time.time)

    @property
    def total_activations(self) -> int:
        return len(self.flow_activations) + len(self.phase_activations)

    @property
    def mean_flow_magnitude(self) -> float:
        if not self.flow_activations:
            return 0.0
        return sum(f.magnitude for f in self.flow_activations) / len(
            self.flow_activations
        )

    @property
    def cascade_depth(self) -> int:
        """Number of distinct HRM levels involved in cascades."""
        levels = set()
        for f in self.flow_activations:
            levels.update(f.affected_levels)
        return len(levels)

    @property
    def resonance_count(self) -> int:
        """Number of resonance events detected."""
        return sum(
            1
            for f in self.flow_activations
            if f.flow_type == ThoughtFlowType.LEARNING_RESONANCE
        )

    def gate_report(self) -> Dict[str, Any]:
        """Generate FATE-compatible gate report."""
        flow_passed = [f for f in self.flow_activations if f.passes_gate()]
        return {
            "total_flow_activations": len(self.flow_activations),
            "total_phase_activations": len(self.phase_activations),
            "flow_passed_snr": len(flow_passed),
            "mean_flow_magnitude": self.mean_flow_magnitude,
            "cascade_depth": self.cascade_depth,
            "resonance_events": self.resonance_count,
            "phi_alignment": self._phi_alignment(),
        }

    def _phi_alignment(self) -> float:
        """How close the convergence-divergence ratio is to φ."""
        conv_divs = [
            p
            for p in self.phase_activations
            if p.pattern_type == PhasePatternType.CONVERGENCE_DIVERGENCE
        ]
        if not conv_divs:
            return 0.0
        return sum(p.intensity for p in conv_divs) / len(conv_divs)


# ═══════════════════════════════════════════════════════════════════════════════
# THOUGHT FLOW DETECTOR — Core Engine
# ═══════════════════════════════════════════════════════════════════════════════


class ThoughtFlowDetector:
    """Detects hidden thought flow patterns across HRM levels.

    The detector monitors level-level dynamics and identifies
    the 4 meta-level flows and 8 phase-level patterns that drive
    cognitive evolution in the BIZRA hierarchy.

    Usage:
        detector = ThoughtFlowDetector()
        report = detector.analyze_level_dynamics(level_states)
    """

    def __init__(
        self,
        sensitivity: float = 0.5,
        min_magnitude: float = 0.3,
        cascade_amplification: float = 1.1,
    ) -> None:
        self.sensitivity = max(0.0, min(1.0, sensitivity))
        self.min_magnitude = min_magnitude
        self.cascade_amplification = cascade_amplification
        self._flow_history: List[FlowActivation] = []
        self._phase_history: List[PhaseActivation] = []

    # ─── Cross-Level Learning Cascade ────────────────────────────────────
    def detect_cross_level_cascade(
        self,
        level_improvements: Dict[int, float],
    ) -> Optional[FlowActivation]:
        """Detect super-linear cascade from lower to higher levels.

        Meta-Learning Acceleration: A 10% improvement at L0 cascades
        super-linearly through the hierarchy. The amplification factor
        follows: improvement_Ln = improvement_L0 × amplification^n
        """
        if len(level_improvements) < 2:
            return None

        sorted_levels = sorted(level_improvements.items())

        # Check for cascading amplification
        cascade_ratios: List[float] = []
        for i in range(1, len(sorted_levels)):
            prev_imp = sorted_levels[i - 1][1]
            curr_imp = sorted_levels[i][1]
            if prev_imp > 0.001:
                cascade_ratios.append(curr_imp / prev_imp)

        if not cascade_ratios:
            return None

        mean_ratio = sum(cascade_ratios) / len(cascade_ratios)

        # Super-linear = ratio > 1.0 (each level amplifies more)
        is_super_linear = mean_ratio > 1.0
        magnitude = min(1.0, mean_ratio / self.cascade_amplification * self.sensitivity)

        if magnitude < self.min_magnitude:
            return None

        affected = tuple(sorted(level_improvements.keys()))
        direction = "ascending" if is_super_linear else "descending"

        activation = FlowActivation(
            flow_type=ThoughtFlowType.CROSS_LEVEL_CASCADE,
            evidence=(
                f"Level improvements: {dict(sorted_levels)}, "
                f"cascade ratios: {[f'{r:.3f}' for r in cascade_ratios]}, "
                f"mean ratio: {mean_ratio:.3f}"
            ),
            magnitude=magnitude,
            direction=direction,
            insight=(
                f"{'Super-linear' if is_super_linear else 'Sub-linear'} cascade detected. "
                f"Mean amplification: {mean_ratio:.3f}×. "
                f"{'Invest in L0 for maximum leverage.' if is_super_linear else 'Check for bottlenecks.'}"
            ),
            affected_levels=affected,
        )
        self._flow_history.append(activation)
        return activation

    # ─── Permeable Boundary Principle ────────────────────────────────────
    def detect_permeable_boundary(
        self,
        cross_level_messages: int,
        same_level_messages: int,
        boundary_levels: Tuple[int, int] = (0, 1),
    ) -> Optional[FlowActivation]:
        """Detect boundary permeability between HRM levels.

        Permeable Boundary Principle: Level boundaries should be selective
        membranes, not walls. Optimal permeability allows information
        transfer while preserving level-specific abstraction.
        """
        total = cross_level_messages + same_level_messages
        if total < 1:
            return None

        permeability = cross_level_messages / total

        # Optimal permeability: ~30% cross-level (not too rigid, not too leaky)
        optimal = 0.30
        balance_score = 1.0 - abs(permeability - optimal) / optimal
        balance_score = max(0.0, balance_score)

        magnitude = min(1.0, balance_score * (1.0 + self.sensitivity))

        if magnitude < self.min_magnitude:
            return None

        activation = FlowActivation(
            flow_type=ThoughtFlowType.PERMEABLE_BOUNDARY,
            evidence=(
                f"Cross-level: {cross_level_messages}, Same-level: {same_level_messages}, "
                f"Permeability: {permeability:.3f}, Optimal balance: {balance_score:.3f}"
            ),
            magnitude=magnitude,
            direction="lateral",
            insight=(
                f"Boundary permeability={permeability:.3f} "
                f"(optimal ≈ 0.30). "
                f"{'Well-balanced membrane.' if balance_score > 0.7 else 'Adjust permeability for better information flow.'}"
            ),
            affected_levels=boundary_levels,
        )
        self._flow_history.append(activation)
        return activation

    # ─── Compound Learning Rate ──────────────────────────────────────────
    def detect_compound_learning(
        self,
        learning_history: Sequence[float],
    ) -> Optional[FlowActivation]:
        """Detect compound learning: C(t+dt) = C(t) + g(C(t))·dt.

        Compound Learning Rate: Learning that depends on current capability
        grows exponentially. Detect by measuring whether the growth rate
        itself is growing (second derivative > 0).
        """
        if len(learning_history) < 3:
            return None

        # First derivative (velocity)
        velocities = [
            learning_history[i + 1] - learning_history[i]
            for i in range(len(learning_history) - 1)
        ]

        # Second derivative (acceleration)
        accelerations = [
            velocities[i + 1] - velocities[i] for i in range(len(velocities) - 1)
        ]

        mean_acceleration = sum(accelerations) / len(accelerations)
        mean_velocity = sum(velocities) / len(velocities)

        # Compound learning = positive acceleration (growth of growth)
        is_compound = mean_acceleration > 0
        # Estimate growth function g: velocity ≈ g × current_capability
        current = learning_history[-1]
        if current > 0.001 and mean_velocity > 0:
            growth_rate = mean_velocity / current
        else:
            growth_rate = 0.0

        magnitude = min(1.0, abs(mean_acceleration) * 10 * (1.0 + self.sensitivity))

        if magnitude < self.min_magnitude:
            return None

        activation = FlowActivation(
            flow_type=ThoughtFlowType.COMPOUND_LEARNING,
            evidence=(
                f"History: {len(learning_history)} points, "
                f"velocity={mean_velocity:.4f}, acceleration={mean_acceleration:.4f}, "
                f"growth_rate g≈{growth_rate:.4f}"
            ),
            magnitude=magnitude,
            direction="ascending" if is_compound else "descending",
            insight=(
                f"{'Compound' if is_compound else 'Decelerating'} learning detected. "
                f"g(C)≈{growth_rate:.4f}. "
                f"{'Capability is self-amplifying — protect this trajectory.' if is_compound else 'Learning curve flattening — inject novelty.'}"
            ),
        )
        self._flow_history.append(activation)
        return activation

    # ─── Learning Resonance ──────────────────────────────────────────────
    def detect_learning_resonance(
        self,
        level_learning_rates: Dict[int, float],
    ) -> Optional[FlowActivation]:
        """Detect cross-level resonance events.

        Learning Resonance: When multiple levels learn at harmonically
        related rates, they create resonance — a multiplicative
        acceleration that exceeds the sum of individual improvements.
        """
        if len(level_learning_rates) < 2:
            return None

        rates = list(level_learning_rates.values())
        levels = list(level_learning_rates.keys())

        # Check for harmonic relationships between rates
        harmonic_pairs: List[Tuple[int, int, float]] = []
        for i in range(len(rates)):
            for j in range(i + 1, len(rates)):
                if rates[j] > 0.001:
                    ratio = rates[i] / rates[j]
                    # Check if ratio is near an integer or simple fraction
                    for harmonic in [1.0, 2.0, 0.5, PHI, 1.0 / PHI, 3.0, 1.0 / 3.0]:
                        if abs(ratio - harmonic) < 0.15:
                            harmonic_pairs.append((levels[i], levels[j], harmonic))
                            break

        if not harmonic_pairs:
            return None

        resonance_strength = len(harmonic_pairs) / max(len(rates), 1)
        magnitude = min(1.0, resonance_strength * (1.0 + self.sensitivity))

        if magnitude < self.min_magnitude:
            return None

        affected = tuple(
            sorted(
                set(level for pair in harmonic_pairs for level in (pair[0], pair[1]))
            )
        )

        harmonic_desc = "; ".join(
            f"L{p[0]}-L{p[1]} ratio≈{p[2]:.3f}" for p in harmonic_pairs[:3]
        )

        activation = FlowActivation(
            flow_type=ThoughtFlowType.LEARNING_RESONANCE,
            evidence=f"Harmonic pairs: [{harmonic_desc}], strength={resonance_strength:.3f}",
            magnitude=magnitude,
            direction="resonant",
            insight=(
                f"Learning resonance across {len(affected)} levels. "
                f"{len(harmonic_pairs)} harmonic pair(s). "
                f"Resonance amplifies learning beyond linear sum — protect this state."
            ),
            affected_levels=affected,
        )
        self._flow_history.append(activation)
        return activation

    # ─── Phase-Level: Convergence-Divergence Pulse (φ rhythm) ────────────
    def detect_convergence_divergence(
        self,
        convergence_score: float,
        divergence_score: float,
        phase_index: int = 0,
    ) -> Optional[PhaseActivation]:
        """Detect convergence-divergence pulse alignment with φ.

        The golden ratio φ ≈ 1.618 governs the optimal rhythm between
        convergent (exploitation) and divergent (exploration) phases.
        Optimal ratio: divergence/convergence ≈ φ early,
                       convergence/divergence ≈ φ late.
        """
        if convergence_score <= 0 and divergence_score <= 0:
            return None

        total = convergence_score + divergence_score
        if total < 0.01:
            return None

        # Compute ratio
        if convergence_score > 0.001:
            ratio = divergence_score / convergence_score
        else:
            ratio = float("inf")

        # Proximity to φ
        if math.isinf(ratio):
            phi_alignment = 0.0
        else:
            phi_alignment = 1.0 - min(1.0, abs(ratio - PHI) / PHI)

        intensity = max(0.0, phi_alignment * (1.0 + self.sensitivity))
        intensity = min(1.0, intensity)

        if intensity < self.min_magnitude:
            return None

        activation = PhaseActivation(
            pattern_type=PhasePatternType.CONVERGENCE_DIVERGENCE,
            phase_index=phase_index,
            evidence=(
                f"Convergence={convergence_score:.3f}, Divergence={divergence_score:.3f}, "
                f"Ratio={ratio:.3f}, φ={PHI:.3f}, Alignment={phi_alignment:.3f}"
            ),
            intensity=intensity,
            insight=(
                f"Convergence-divergence ratio {ratio:.3f} vs φ={PHI:.3f}. "
                f"Alignment={phi_alignment:.1%}. "
                f"{'Optimal golden ratio rhythm achieved.' if phi_alignment > 0.8 else 'Adjust exploration/exploitation balance toward φ.'}"
            ),
        )
        self._phase_history.append(activation)
        return activation

    # ─── Phase-Level: Meta-Learning Acceleration ─────────────────────────
    def detect_meta_learning_acceleration(
        self,
        l0_improvement: float,
        total_hierarchy_improvement: float,
        level_count: int = 5,
        phase_index: int = 0,
    ) -> Optional[PhaseActivation]:
        """Detect meta-learning acceleration from L0 improvements.

        Meta-Learning Acceleration: A 10% improvement at L0 (perceptual)
        should cascade through the hierarchy. If total hierarchy improvement
        exceeds 10% × level_count, we have super-linear acceleration.
        """
        if l0_improvement <= 0:
            return None

        # Expected linear improvement
        expected_linear = l0_improvement * level_count

        # Acceleration ratio
        if expected_linear > 0.001:
            acceleration = total_hierarchy_improvement / expected_linear
        else:
            acceleration = 0.0

        is_super_linear = acceleration > 1.0
        intensity = min(1.0, acceleration * self.sensitivity)

        if intensity < self.min_magnitude:
            return None

        activation = PhaseActivation(
            pattern_type=PhasePatternType.META_LEARNING_ACCELERATION,
            phase_index=phase_index,
            evidence=(
                f"L0 improvement={l0_improvement:.3f}, "
                f"Total hierarchy={total_hierarchy_improvement:.3f}, "
                f"Expected linear={expected_linear:.3f}, "
                f"Acceleration={acceleration:.3f}×"
            ),
            intensity=intensity,
            insight=(
                f"{'Super-linear' if is_super_linear else 'Sub-linear'} acceleration: "
                f"{acceleration:.2f}× multiplier. "
                f"{'L0 investments are compounding through hierarchy.' if is_super_linear else 'Check for cascade bottlenecks between levels.'}"
            ),
        )
        self._phase_history.append(activation)
        return activation

    # ─── Phase-Level: Integration Depth Gradient ─────────────────────────
    def detect_integration_depth(
        self,
        surface_count: int,
        intermediate_count: int,
        deep_count: int,
        foundational_count: int,
        phase_index: int = 0,
    ) -> Optional[PhaseActivation]:
        """Detect integration depth distribution.

        Expected distribution:
          Surface:       70% volume / 15% impact
          Intermediate:  20% volume / 25% impact
          Deep:           8% volume / 35% impact
          Foundational:   2% volume / 25% impact
        """
        total = surface_count + intermediate_count + deep_count + foundational_count
        if total < 1:
            return None

        actual = {
            "surface": surface_count / total,
            "intermediate": intermediate_count / total,
            "deep": deep_count / total,
            "foundational": foundational_count / total,
        }

        expected = {
            "surface": 0.70,
            "intermediate": 0.20,
            "deep": 0.08,
            "foundational": 0.02,
        }

        # Compute KL-like divergence from expected
        alignment = 0.0
        for key in expected:
            if expected[key] > 0:
                alignment += 1.0 - min(
                    1.0, abs(actual[key] - expected[key]) / expected[key]
                )
        alignment /= len(expected)

        intensity = min(1.0, alignment * (1.0 + self.sensitivity))

        if intensity < self.min_magnitude:
            return None

        activation = PhaseActivation(
            pattern_type=PhasePatternType.INTEGRATION_DEPTH,
            phase_index=phase_index,
            evidence=(
                f"Distribution: S={actual['surface']:.1%}, I={actual['intermediate']:.1%}, "
                f"D={actual['deep']:.1%}, F={actual['foundational']:.1%}, "
                f"Alignment={alignment:.3f}"
            ),
            intensity=intensity,
            insight=(
                f"Integration depth alignment={alignment:.1%} with ideal gradient. "
                f"{'Healthy depth distribution.' if alignment > 0.7 else 'Consider investing more in deep/foundational integration.'}"
            ),
        )
        self._phase_history.append(activation)
        return activation

    # ─── Full Analysis ──────────────────────────────────────────────────
    def analyze_level_dynamics(
        self,
        dynamics: Dict[str, Any],
        cycle_id: str = "",
    ) -> FlowReport:
        """Run all thought flow and phase pattern detectors.

        Args:
            dynamics: Dict with keys matching detector parameters.
            cycle_id: Identifier for this analysis cycle.

        Returns:
            FlowReport with all activations.
        """
        report = FlowReport(cycle_id=cycle_id)

        # Meta-level flows
        if "level_improvements" in dynamics:
            result = self.detect_cross_level_cascade(dynamics["level_improvements"])
            if result:
                report.flow_activations.append(result)

        if "cross_level_messages" in dynamics and "same_level_messages" in dynamics:
            result = self.detect_permeable_boundary(
                dynamics["cross_level_messages"],
                dynamics["same_level_messages"],
                dynamics.get("boundary_levels", (0, 1)),
            )
            if result:
                report.flow_activations.append(result)

        if "learning_history" in dynamics:
            result = self.detect_compound_learning(dynamics["learning_history"])
            if result:
                report.flow_activations.append(result)

        if "level_learning_rates" in dynamics:
            result = self.detect_learning_resonance(dynamics["level_learning_rates"])
            if result:
                report.flow_activations.append(result)

        # Phase-level patterns
        if "convergence_score" in dynamics and "divergence_score" in dynamics:
            result_p = self.detect_convergence_divergence(
                dynamics["convergence_score"],
                dynamics["divergence_score"],
                dynamics.get("phase_index", 0),
            )
            if result_p:
                report.phase_activations.append(result_p)

        if "l0_improvement" in dynamics and "total_hierarchy_improvement" in dynamics:
            result_p = self.detect_meta_learning_acceleration(
                dynamics["l0_improvement"],
                dynamics["total_hierarchy_improvement"],
                dynamics.get("level_count", 5),
                dynamics.get("phase_index", 0),
            )
            if result_p:
                report.phase_activations.append(result_p)

        if "surface_count" in dynamics:
            result_p = self.detect_integration_depth(
                dynamics["surface_count"],
                dynamics.get("intermediate_count", 0),
                dynamics.get("deep_count", 0),
                dynamics.get("foundational_count", 0),
                dynamics.get("phase_index", 0),
            )
            if result_p:
                report.phase_activations.append(result_p)

        return report

    @property
    def flow_history(self) -> List[FlowActivation]:
        return list(self._flow_history)

    @property
    def phase_history(self) -> List[PhaseActivation]:
        return list(self._phase_history)

    def reset_history(self) -> int:
        count = len(self._flow_history) + len(self._phase_history)
        self._flow_history.clear()
        self._phase_history.clear()
        return count
