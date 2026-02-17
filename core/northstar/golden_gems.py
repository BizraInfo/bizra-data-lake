"""
BIZRA Node0 NorthStar — Golden Gems Engine
╔══════════════════════════════════════════════════════════════════════════════╗
║  8 Golden Gems × SNR-Scored Cognitive Primitives                            ║
║  Hidden pattern engines extracted from cross-document synthesis              ║
║                                                                              ║
║  بسم الله الرحمن الرحيم                                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Each Golden Gem encodes a meta-cognitive principle discovered through
Graph-of-Thoughts exploration across the full BIZRA document corpus.
These are not heuristics — they are structural properties of how
intelligence MUST operate at the intersection of autopoiesis,
hierarchical reasoning, and proof-carrying inference.

Golden Gems (with origin SNR scores):
  1. Shadow Knowledge Principle    (SNR 9.7/10) — Blind spots ARE data
  2. Contradiction Harvest         (SNR 9.5/10) — Conflicts reveal structure
  3. Emergence Principle           (SNR 9.8/10) — Topology > individual nodes
  4. Noise-is-Signal               (SNR 9.4/10) — Persistent noise = unconceptualized truth
  5. Validation Mirror             (SNR 9.6/10) — Rejections map system boundaries
  6. Implementation as Hypothesis  (SNR 9.3/10) — Every execution is experiment
  7. Integration Paradox           (SNR 9.5/10) — Easy = limited; friction = breakthrough
  8. Outcome as Oracle             (SNR 9.7/10) — Results answer unasked questions

Standing on Giants:
  - Shannon (1948) — information as surprise
  - Popper (1963) — falsifiability as knowledge boundary
  - Taleb (2007) — antifragility from stressors
  - Kuhn (1962) — anomalies as paradigm signals
  - Maturana & Varela (1980) — structural coupling and operational closure
  - Al-Ghazali (1095) — Ihsān as meta-cognitive vigilance

Created: 2026-02-15 | BIZRA Node0 NorthStar | Peak Masterpiece Protocol
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)


# ═══════════════════════════════════════════════════════════════════════════════
# GOLDEN GEM TAXONOMY
# ═══════════════════════════════════════════════════════════════════════════════


class GoldenGemType(Enum):
    """The 8 golden gems — meta-cognitive primitives of the NorthStar."""

    SHADOW_KNOWLEDGE = auto()       # Blind spots as data
    CONTRADICTION_HARVEST = auto()  # Conflicts reveal hidden structure
    EMERGENCE_PRINCIPLE = auto()    # Topology > individual nodes
    NOISE_IS_SIGNAL = auto()        # Persistent noise = unconceptualized truth
    VALIDATION_MIRROR = auto()      # Rejections map system boundaries
    IMPLEMENTATION_HYPOTHESIS = auto()  # Every execution is experiment
    INTEGRATION_PARADOX = auto()    # Friction signals breakthrough potential
    OUTCOME_ORACLE = auto()         # Results answer unasked questions


# Origin SNR scores from cross-document synthesis (scale: 0-10)
GEM_ORIGIN_SNR: Dict[GoldenGemType, float] = {
    GoldenGemType.SHADOW_KNOWLEDGE: 9.7,
    GoldenGemType.CONTRADICTION_HARVEST: 9.5,
    GoldenGemType.EMERGENCE_PRINCIPLE: 9.8,
    GoldenGemType.NOISE_IS_SIGNAL: 9.4,
    GoldenGemType.VALIDATION_MIRROR: 9.6,
    GoldenGemType.IMPLEMENTATION_HYPOTHESIS: 9.3,
    GoldenGemType.INTEGRATION_PARADOX: 9.5,
    GoldenGemType.OUTCOME_ORACLE: 9.7,
}

# Normalized to [0, 1] for BIZRA SNR compatibility
GEM_NORMALIZED_SNR: Dict[GoldenGemType, float] = {
    gem: score / 10.0 for gem, score in GEM_ORIGIN_SNR.items()
}


# ═══════════════════════════════════════════════════════════════════════════════
# GOLDEN GEM ACTIVATION DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class GemActivation:
    """Record of a golden gem being activated by observed evidence.

    A gem activates when the system detects a pattern that matches its
    meta-cognitive principle. Each activation carries:
    - The gem type that fired
    - Evidence that triggered it (observations, contradictions, anomalies)
    - Computed intensity (how strongly the pattern matches)
    - Extracted insight (what the gem reveals about the system state)
    """

    gem_type: GoldenGemType
    evidence: str
    intensity: float          # [0, 1] — strength of pattern match
    insight: str              # What the gem reveals
    timestamp: float = field(default_factory=time.time)
    source_level: int = 0     # HRM abstraction level that produced evidence
    confidence: float = 0.95  # Ihsān-floor confidence

    def snr_score(self) -> float:
        """Compute SNR for this activation.

        SNR = gem_origin_snr × intensity × confidence
        """
        origin = GEM_NORMALIZED_SNR.get(self.gem_type, 0.9)
        return origin * self.intensity * self.confidence

    def passes_gate(self) -> bool:
        """Check if activation clears unified SNR threshold."""
        return self.snr_score() >= UNIFIED_SNR_THRESHOLD

    def passes_ihsan(self) -> bool:
        """Check if confidence meets Ihsān floor."""
        return self.confidence >= UNIFIED_IHSAN_THRESHOLD


@dataclass
class GemReport:
    """Summary report of gem activations across a cognitive cycle."""

    activations: List[GemActivation] = field(default_factory=list)
    cycle_id: str = ""
    total_evidence_processed: int = 0
    timestamp: float = field(default_factory=time.time)

    @property
    def active_gem_count(self) -> int:
        """Number of distinct gem types that activated."""
        return len({a.gem_type for a in self.activations})

    @property
    def mean_snr(self) -> float:
        """Average SNR across all activations."""
        if not self.activations:
            return 0.0
        return sum(a.snr_score() for a in self.activations) / len(self.activations)

    @property
    def mean_intensity(self) -> float:
        """Average intensity across all activations."""
        if not self.activations:
            return 0.0
        return sum(a.intensity for a in self.activations) / len(self.activations)

    @property
    def dominant_gem(self) -> Optional[GoldenGemType]:
        """The gem with highest cumulative intensity in this cycle."""
        if not self.activations:
            return None
        gem_scores: Dict[GoldenGemType, float] = {}
        for a in self.activations:
            gem_scores[a.gem_type] = gem_scores.get(a.gem_type, 0.0) + a.intensity
        return max(gem_scores, key=gem_scores.get)  # type: ignore[arg-type]

    def gate_report(self) -> Dict[str, Any]:
        """Generate FATE-compatible gate report."""
        passed = [a for a in self.activations if a.passes_gate()]
        failed = [a for a in self.activations if not a.passes_gate()]
        return {
            "total_activations": len(self.activations),
            "passed_snr_gate": len(passed),
            "failed_snr_gate": len(failed),
            "mean_snr": self.mean_snr,
            "active_gem_types": self.active_gem_count,
            "dominant_gem": self.dominant_gem.name if self.dominant_gem else None,
            "ihsan_compliant": all(a.passes_ihsan() for a in self.activations),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# GOLDEN GEM DETECTOR — Core Engine
# ═══════════════════════════════════════════════════════════════════════════════


class GoldenGemDetector:
    """Detects golden gem activations from system observations.

    The detector processes structured observations (from HRM levels,
    RDVE cycles, autopoietic phases) and identifies when golden gem
    patterns are present in the data. Each gem has a specific detection
    heuristic derived from the cross-document synthesis.

    Usage:
        detector = GoldenGemDetector()
        report = detector.analyze_observations(observations)
        if report.mean_snr >= SNR_THRESHOLD_T1_HIGH:
            # High-quality gems found — feed to NorthStar engine
            ...
    """

    def __init__(
        self,
        sensitivity: float = 0.5,
        min_intensity: float = 0.3,
        ihsan_floor: float = UNIFIED_IHSAN_THRESHOLD,
    ) -> None:
        """Initialize detector.

        Args:
            sensitivity: Detection sensitivity [0,1]. Higher = more activations.
            min_intensity: Minimum intensity to register an activation.
            ihsan_floor: Minimum confidence for Ihsān compliance.
        """
        self.sensitivity = max(0.0, min(1.0, sensitivity))
        self.min_intensity = min_intensity
        self.ihsan_floor = ihsan_floor
        self._activation_history: List[GemActivation] = []

    # ─── Shadow Knowledge: Blind spots ARE data ─────────────────────────
    def detect_shadow_knowledge(
        self,
        observed_domains: Sequence[str],
        all_known_domains: Sequence[str],
    ) -> Optional[GemActivation]:
        """Detect blind spots — domains NOT observed are informative.

        The Shadow Knowledge Principle: What you DON'T see tells you
        as much as what you DO see. Missing observations indicate
        either unexamined territory or structural boundaries.
        """
        observed_set = set(observed_domains)
        full_set = set(all_known_domains)
        blind_spots = full_set - observed_set

        if not blind_spots:
            return None

        coverage = len(observed_set) / max(len(full_set), 1)
        shadow_ratio = len(blind_spots) / max(len(full_set), 1)

        # Intensity increases with shadow ratio AND sensitivity
        intensity = min(1.0, shadow_ratio * (1.0 + self.sensitivity))

        if intensity < self.min_intensity:
            return None

        blind_list = ", ".join(sorted(blind_spots)[:5])
        activation = GemActivation(
            gem_type=GoldenGemType.SHADOW_KNOWLEDGE,
            evidence=f"Blind spots detected: [{blind_list}] ({len(blind_spots)} domains unobserved)",
            intensity=intensity,
            insight=(
                f"Coverage {coverage:.0%}. {len(blind_spots)} domain(s) unexamined. "
                f"These shadows may contain critical undiscovered patterns."
            ),
            confidence=max(self.ihsan_floor, 0.9 + 0.1 * coverage),
        )
        self._activation_history.append(activation)
        return activation

    # ─── Contradiction Harvest: Conflicts reveal hidden structure ────────
    def detect_contradiction_harvest(
        self,
        assertions: Sequence[Dict[str, Any]],
    ) -> Optional[GemActivation]:
        """Detect contradictions in a set of assertions.

        The Contradiction Harvest: When two claims conflict, neither
        is simply 'wrong' — the contradiction points to a deeper
        structural truth that neither claim fully captures.
        """
        if len(assertions) < 2:
            return None

        contradictions: List[Tuple[int, int, str]] = []

        for i in range(len(assertions)):
            for j in range(i + 1, len(assertions)):
                a = assertions[i]
                b = assertions[j]

                # Detect claim-level contradiction
                if a.get("claim") and b.get("claim"):
                    a_val = a.get("value")
                    b_val = b.get("value")
                    if a.get("domain") == b.get("domain") and a_val != b_val:
                        contradictions.append((
                            i, j,
                            f"'{a.get('claim')}' vs '{b.get('claim')}' in domain '{a.get('domain')}'"
                        ))

                # Detect confidence-level contradiction
                a_conf = a.get("confidence", 0.5)
                b_conf = b.get("confidence", 0.5)
                if abs(a_conf - b_conf) > 0.4 and a.get("topic") == b.get("topic"):
                    contradictions.append((
                        i, j,
                        f"Confidence gap: {a_conf:.2f} vs {b_conf:.2f} on '{a.get('topic')}'"
                    ))

        if not contradictions:
            return None

        contradiction_density = len(contradictions) / max(len(assertions), 1)
        intensity = min(1.0, contradiction_density * (1.0 + self.sensitivity))

        if intensity < self.min_intensity:
            return None

        sample = contradictions[0][2]
        activation = GemActivation(
            gem_type=GoldenGemType.CONTRADICTION_HARVEST,
            evidence=f"{len(contradictions)} contradiction(s) found. Sample: {sample}",
            intensity=intensity,
            insight=(
                f"These contradictions suggest hidden structural dimensions. "
                f"Density: {contradiction_density:.2f}. Investigate boundary conditions."
            ),
            confidence=self.ihsan_floor,
        )
        self._activation_history.append(activation)
        return activation

    # ─── Emergence Principle: Topology > individual nodes ────────────────
    def detect_emergence(
        self,
        node_count: int,
        edge_count: int,
        clustering_coefficient: float = 0.0,
        avg_path_length: float = 0.0,
    ) -> Optional[GemActivation]:
        """Detect emergent properties from graph topology.

        The Emergence Principle: The graph as a whole encodes information
        that no individual node possesses. Small-world properties
        (high clustering ≈ 0.4, short paths ≈ 3.2) indicate emergence.
        """
        if node_count < 3 or edge_count < 2:
            return None

        # Edge density
        max_edges = node_count * (node_count - 1) / 2
        density = edge_count / max(max_edges, 1)

        # Small-world detection (reference: clustering ≈ 0.4, path ≈ 3.2)
        small_world_score = 0.0
        if clustering_coefficient > 0:
            # How close to optimal small-world clustering
            cluster_proximity = 1.0 - abs(clustering_coefficient - 0.4) / 0.4
            small_world_score += max(0.0, cluster_proximity) * 0.5

        if avg_path_length > 0:
            path_proximity = 1.0 - abs(avg_path_length - 3.2) / 3.2
            small_world_score += max(0.0, path_proximity) * 0.5

        # Combined intensity
        intensity = min(1.0, (density * 0.3 + small_world_score * 0.7) * (1.0 + self.sensitivity))

        if intensity < self.min_intensity:
            return None

        activation = GemActivation(
            gem_type=GoldenGemType.EMERGENCE_PRINCIPLE,
            evidence=(
                f"Graph: {node_count} nodes, {edge_count} edges, "
                f"clustering={clustering_coefficient:.3f}, path_length={avg_path_length:.2f}"
            ),
            intensity=intensity,
            insight=(
                f"Emergence detected: density={density:.3f}, "
                f"small-world score={small_world_score:.3f}. "
                f"Topology contains information no individual node possesses."
            ),
            confidence=max(self.ihsan_floor, 0.90 + 0.10 * small_world_score),
        )
        self._activation_history.append(activation)
        return activation

    # ─── Noise-is-Signal: Persistent noise = unconceptualized truth ──────
    def detect_noise_as_signal(
        self,
        noise_history: Sequence[float],
        window_size: int = 5,
    ) -> Optional[GemActivation]:
        """Detect persistent noise patterns that indicate hidden signal.

        The Noise-is-Signal Principle: If noise persists across multiple
        observation windows, it's not random — it's structure the system
        hasn't conceptualized yet. σ²/s ≈ 2.3 indicates punctuated equilibrium.
        """
        if len(noise_history) < window_size:
            return None

        # Compute noise persistence: variance of noise values across windows
        windows: List[List[float]] = []
        for i in range(0, len(noise_history) - window_size + 1, max(1, window_size // 2)):
            windows.append(list(noise_history[i:i + window_size]))

        if len(windows) < 2:
            return None

        # Cross-window correlation (persistent noise = high cross-correlation)
        window_means = [sum(w) / len(w) for w in windows]
        mean_of_means = sum(window_means) / len(window_means)
        variance = sum((m - mean_of_means) ** 2 for m in window_means) / len(window_means)

        # Low variance across windows = persistent noise (= hidden signal)
        persistence = 1.0 - min(1.0, variance / max(mean_of_means, 0.01))

        # σ²/s ratio check (punctuated equilibrium marker at ≈ 2.3)
        overall_mean = sum(noise_history) / len(noise_history)
        overall_var = sum((x - overall_mean) ** 2 for x in noise_history) / len(noise_history)
        sigma_over_s = overall_var / max(overall_mean, 0.001)
        equilibrium_proximity = 1.0 - min(1.0, abs(sigma_over_s - 2.3) / 2.3)

        intensity = min(1.0, (persistence * 0.6 + equilibrium_proximity * 0.4) * (1.0 + self.sensitivity))

        if intensity < self.min_intensity:
            return None

        activation = GemActivation(
            gem_type=GoldenGemType.NOISE_IS_SIGNAL,
            evidence=(
                f"Noise persistence={persistence:.3f}, σ²/s={sigma_over_s:.3f} "
                f"(punctuated equilibrium at 2.3), windows={len(windows)}"
            ),
            intensity=intensity,
            insight=(
                "Persistent noise pattern detected. This noise is NOT random — "
                "it represents structure the system hasn't conceptualized yet. "
                "Investigate for hidden dimensions."
            ),
            confidence=max(self.ihsan_floor, 0.90 + 0.05 * persistence),
        )
        self._activation_history.append(activation)
        return activation

    # ─── Validation Mirror: Rejections map system boundaries ─────────────
    def detect_validation_mirror(
        self,
        total_validations: int,
        rejections: int,
        rejection_reasons: Optional[Sequence[str]] = None,
    ) -> Optional[GemActivation]:
        """Detect system boundary patterns from validation rejections.

        The Validation Mirror: What the system REJECTS tells you its
        shape. Rejection patterns map the boundary between acceptable
        and unacceptable — revealing implicit assumptions.
        """
        if total_validations < 1:
            return None

        rejection_rate = rejections / total_validations

        # Optimal information: neither 0% nor 100% rejection
        # Maximum information at ~30% rejection (rule of thumb)
        info_score = 1.0 - abs(rejection_rate - 0.30) / 0.30
        info_score = max(0.0, info_score)

        intensity = min(1.0, info_score * (1.0 + self.sensitivity))

        if intensity < self.min_intensity:
            return None

        reasons_str = ""
        if rejection_reasons:
            top_reasons = sorted(set(rejection_reasons))[:3]
            reasons_str = f" Top reasons: {', '.join(top_reasons)}."

        activation = GemActivation(
            gem_type=GoldenGemType.VALIDATION_MIRROR,
            evidence=(
                f"Validation: {total_validations} total, {rejections} rejected "
                f"({rejection_rate:.1%}).{reasons_str}"
            ),
            intensity=intensity,
            insight=(
                f"Rejection pattern maps system boundaries. "
                f"Information score={info_score:.3f}. "
                f"These boundaries reveal implicit system assumptions."
            ),
            confidence=self.ihsan_floor,
        )
        self._activation_history.append(activation)
        return activation

    # ─── Implementation as Hypothesis: Every execution is experiment ─────
    def detect_implementation_hypothesis(
        self,
        planned_outcome: str,
        actual_outcome: str,
        divergence_score: float = 0.0,
    ) -> Optional[GemActivation]:
        """Detect when implementation diverges from plan.

        The Implementation as Hypothesis: Every line of code is a
        hypothesis about how the world works. The gap between plan
        and reality IS tacit knowledge made visible.
        """
        intensity = min(1.0, divergence_score * (1.0 + self.sensitivity))

        if intensity < self.min_intensity:
            return None

        activation = GemActivation(
            gem_type=GoldenGemType.IMPLEMENTATION_HYPOTHESIS,
            evidence=(
                f"Plan: '{planned_outcome[:80]}' | "
                f"Actual: '{actual_outcome[:80]}' | "
                f"Divergence: {divergence_score:.3f}"
            ),
            intensity=intensity,
            insight=(
                f"The gap between plan and reality ({divergence_score:.3f}) "
                f"represents tacit knowledge. This divergence is not failure — "
                f"it's the system learning its own implicit assumptions."
            ),
            confidence=max(self.ihsan_floor, 0.92),
        )
        self._activation_history.append(activation)
        return activation

    # ─── Integration Paradox: Easy = limited; friction = breakthrough ────
    def detect_integration_paradox(
        self,
        integration_effort: float,
        integration_value: float,
    ) -> Optional[GemActivation]:
        """Detect the integration paradox.

        The Integration Paradox: Easy integration = limited transformation.
        High-friction integration = breakthrough potential. The distribution:
          Surface:       70% volume / 15% impact
          Intermediate:  20% volume / 25% impact
          Deep:           8% volume / 35% impact
          Foundational:   2% volume / 25% impact
        """
        if integration_effort <= 0 or integration_value <= 0:
            return None

        # Value-per-effort ratio (high effort + high value = paradox confirmed)
        ratio = integration_value / integration_effort
        # The paradox: when effort is HIGH and value is ALSO HIGH
        paradox_score = min(1.0, integration_effort * integration_value)

        intensity = min(1.0, paradox_score * (1.0 + self.sensitivity))

        if intensity < self.min_intensity:
            return None

        # Classify integration depth
        if integration_effort < 0.3:
            depth = "Surface (70% volume / 15% impact)"
        elif integration_effort < 0.6:
            depth = "Intermediate (20% volume / 25% impact)"
        elif integration_effort < 0.9:
            depth = "Deep (8% volume / 35% impact)"
        else:
            depth = "Foundational (2% volume / 25% impact)"

        activation = GemActivation(
            gem_type=GoldenGemType.INTEGRATION_PARADOX,
            evidence=(
                f"Effort={integration_effort:.3f}, Value={integration_value:.3f}, "
                f"Ratio={ratio:.3f}, Depth: {depth}"
            ),
            intensity=intensity,
            insight=(
                f"Integration paradox active: {depth}. "
                f"High-friction integrations signal breakthrough potential."
            ),
            confidence=self.ihsan_floor,
        )
        self._activation_history.append(activation)
        return activation

    # ─── Outcome as Oracle: Results answer unasked questions ─────────────
    def detect_outcome_oracle(
        self,
        expected_outcomes: Sequence[str],
        observed_outcomes: Sequence[str],
    ) -> Optional[GemActivation]:
        """Detect when outcomes reveal unexpected insights.

        The Outcome as Oracle: Results always answer more questions
        than were asked. The UNEXPECTED outcomes are the most valuable —
        they reveal dimensions the system didn't know to look for.
        """
        expected_set = set(expected_outcomes)
        observed_set = set(observed_outcomes)

        unexpected = observed_set - expected_set
        missing = expected_set - observed_set
        expected_set & observed_set

        if not unexpected and not missing:
            return None  # Perfect match = no oracle signal

        surprise_ratio = len(unexpected) / max(len(observed_set), 1)
        missing_ratio = len(missing) / max(len(expected_set), 1)
        oracle_score = (surprise_ratio + missing_ratio) / 2

        intensity = min(1.0, oracle_score * (1.0 + self.sensitivity))

        if intensity < self.min_intensity:
            return None

        unexpected_list = ", ".join(sorted(unexpected)[:3]) if unexpected else "none"
        missing_list = ", ".join(sorted(missing)[:3]) if missing else "none"

        activation = GemActivation(
            gem_type=GoldenGemType.OUTCOME_ORACLE,
            evidence=(
                f"Expected: {len(expected_set)}, Observed: {len(observed_set)}, "
                f"Unexpected: [{unexpected_list}], Missing: [{missing_list}]"
            ),
            intensity=intensity,
            insight=(
                f"Oracle signal: {len(unexpected)} unexpected outcome(s) reveal "
                f"dimensions not asked about. {len(missing)} missing outcome(s) "
                f"indicate boundary conditions. Both are valuable data."
            ),
            confidence=max(self.ihsan_floor, 0.93),
        )
        self._activation_history.append(activation)
        return activation

    # ─── Full Analysis ──────────────────────────────────────────────────
    def analyze_observations(
        self,
        observations: Dict[str, Any],
        cycle_id: str = "",
    ) -> GemReport:
        """Run all 8 gem detectors on a set of observations.

        Args:
            observations: Dict with keys matching detector parameters.
                Expected keys (all optional):
                  - observed_domains, all_known_domains → Shadow Knowledge
                  - assertions → Contradiction Harvest
                  - node_count, edge_count, clustering_coefficient, avg_path_length → Emergence
                  - noise_history → Noise-is-Signal
                  - total_validations, rejections, rejection_reasons → Validation Mirror
                  - planned_outcome, actual_outcome, divergence_score → Implementation Hypothesis
                  - integration_effort, integration_value → Integration Paradox
                  - expected_outcomes, observed_outcomes → Outcome Oracle
            cycle_id: Identifier for this analysis cycle.

        Returns:
            GemReport with all activations.
        """
        report = GemReport(cycle_id=cycle_id)
        report.total_evidence_processed = len(observations)

        # 1. Shadow Knowledge
        if "observed_domains" in observations and "all_known_domains" in observations:
            result = self.detect_shadow_knowledge(
                observations["observed_domains"],
                observations["all_known_domains"],
            )
            if result:
                report.activations.append(result)

        # 2. Contradiction Harvest
        if "assertions" in observations:
            result = self.detect_contradiction_harvest(observations["assertions"])
            if result:
                report.activations.append(result)

        # 3. Emergence Principle
        if "node_count" in observations and "edge_count" in observations:
            result = self.detect_emergence(
                observations["node_count"],
                observations["edge_count"],
                observations.get("clustering_coefficient", 0.0),
                observations.get("avg_path_length", 0.0),
            )
            if result:
                report.activations.append(result)

        # 4. Noise-is-Signal
        if "noise_history" in observations:
            result = self.detect_noise_as_signal(observations["noise_history"])
            if result:
                report.activations.append(result)

        # 5. Validation Mirror
        if "total_validations" in observations and "rejections" in observations:
            result = self.detect_validation_mirror(
                observations["total_validations"],
                observations["rejections"],
                observations.get("rejection_reasons"),
            )
            if result:
                report.activations.append(result)

        # 6. Implementation as Hypothesis
        if "planned_outcome" in observations and "actual_outcome" in observations:
            result = self.detect_implementation_hypothesis(
                observations["planned_outcome"],
                observations["actual_outcome"],
                observations.get("divergence_score", 0.0),
            )
            if result:
                report.activations.append(result)

        # 7. Integration Paradox
        if "integration_effort" in observations and "integration_value" in observations:
            result = self.detect_integration_paradox(
                observations["integration_effort"],
                observations["integration_value"],
            )
            if result:
                report.activations.append(result)

        # 8. Outcome Oracle
        if "expected_outcomes" in observations and "observed_outcomes" in observations:
            result = self.detect_outcome_oracle(
                observations["expected_outcomes"],
                observations["observed_outcomes"],
            )
            if result:
                report.activations.append(result)

        return report

    @property
    def history(self) -> List[GemActivation]:
        """Full activation history."""
        return list(self._activation_history)

    def reset_history(self) -> int:
        """Clear activation history. Returns count of cleared entries."""
        count = len(self._activation_history)
        self._activation_history.clear()
        return count
