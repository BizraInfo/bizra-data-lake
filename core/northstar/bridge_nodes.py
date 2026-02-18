"""
BIZRA Node0 NorthStar — Cross-Document Bridge Nodes
╔══════════════════════════════════════════════════════════════════════════════╗
║  5 Bridge Nodes × Cross-Domain Integration Architecture                     ║
║  The structural connectors that make BIZRA more than sum of parts           ║
║                                                                              ║
║  بسم الله الرحمن الرحيم                                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Cross-Document Bridge Nodes (with origin SNR scores):
  Bridge 1: Autopoiesis ↔ RDVE Resonance          (SNR 0.97)
    PAT Agent = Generator, SAT Agent = Verifier, Kernel = Meta-Cognitive Optimizer
  Bridge 2: HRM ↔ BIZRA Four Pillars               (SNR 0.98)
    Curry-Howard: L0=Genesis, L1-2=Museum, L3=Runtime, LN=Ihsan
  Bridge 3: SNR ↔ Shannon Channel Duality           (SNR 0.96)
    Noise classification → Shannon's 4 categories
  Bridge 4: GoT ↔ Sacred Geometry Knowledge Graph   (SNR 0.95)
    Small-world: clustering ≈ 0.4, path ≈ 3.2
  Bridge 5: Compound Learning ↔ Recursive Accel.    (SNR 0.99)
    C(t+dt) = C(t) + g(C(t))·dt ≈ 2.3 punctuated equilibrium

Meta-Discovery:
  "Ihsān IS Level N Autopoiesis — ethics IS the system's self-transcendence"

Supreme Insight:
  "Intelligence requires both STRUCTURE and SELF-TRANSCENDENCE.
   Structure enables capability. Autopoiesis enables evolution.
   The fusion enables transcendence."

Standing on Giants:
  - Curry & Howard (1969/1980) — propositions as types
  - Shannon (1948) — channel capacity theorem
  - Watts & Strogatz (1998) — small-world networks
  - Gould & Eldredge (1972) — punctuated equilibrium
  - Al-Ghazali (1095) — Ihsān as meta-ethical compass

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

# ═══════════════════════════════════════════════════════════════════════════════
# BRIDGE NODE TAXONOMY
# ═══════════════════════════════════════════════════════════════════════════════


class BridgeType(Enum):
    """The 5 cross-document bridge nodes."""

    AUTOPOIESIS_RDVE = auto()  # Autopoiesis ↔ RDVE Resonance
    HRM_FOUR_PILLARS = auto()  # HRM ↔ Four Pillars (Curry-Howard)
    SNR_SHANNON_CHANNEL = auto()  # SNR ↔ Shannon Channel Duality
    GOT_SACRED_GEOMETRY = auto()  # GoT ↔ Sacred Geometry / Small-World
    COMPOUND_RECURSIVE = auto()  # Compound Learning ↔ Recursive Acceleration


# Origin SNR scores from cross-document synthesis
BRIDGE_ORIGIN_SNR: Dict[BridgeType, float] = {
    BridgeType.AUTOPOIESIS_RDVE: 0.97,
    BridgeType.HRM_FOUR_PILLARS: 0.98,
    BridgeType.SNR_SHANNON_CHANNEL: 0.96,
    BridgeType.GOT_SACRED_GEOMETRY: 0.95,
    BridgeType.COMPOUND_RECURSIVE: 0.99,
}


# ═══════════════════════════════════════════════════════════════════════════════
# BRIDGE ROLE MAPPINGS — How each bridge connects BIZRA subsystems
# ═══════════════════════════════════════════════════════════════════════════════

# Bridge 1: PAT/SAT ↔ RDVE mapping
AUTOPOIESIS_RDVE_ROLES: Dict[str, str] = {
    "pat_agent": "generator",
    "sat_agent": "verifier",
    "node0_kernel": "meta_cognitive_optimizer",
    "autopoietic_cycle": "rdve_observe_generate_verify",
    "emergence": "interdisciplinary_transfer",
}

# Bridge 2: HRM Level ↔ Four Pillars mapping (Curry-Howard)
HRM_PILLAR_MAP: Dict[str, str] = {
    "L0_perceptual": "genesis_sandbox",  # Pillar 4: Genesis Cutoff
    "L1_operational": "museum",  # Pillar 2: Museum (The Ark)
    "L2_tactical": "museum",  # Pillar 2: Museum (The Ark)
    "L3_strategic": "runtime",  # Pillar 1: Runtime (The Fortress)
    "LN_meta": "adaptive_ihsan",  # Ihsān as Level N autopoiesis
}

# Bridge 3: SNR Noise ↔ Shannon Classification
SHANNON_NOISE_MAP: Dict[str, str] = {
    "measurement_noise": "channel_noise",  # Physical noise
    "model_noise": "source_coding_loss",  # Encoding imperfection
    "structural_noise": "capacity_mismatch",  # Channel capacity < required
    "epistemological_noise": "fundamental_limit",  # Shannon limit approach
}

# Bridge 4: GoT Node ↔ Sacred Geometry / Small-World Constants
GOT_TOPOLOGY_CONSTANTS: Dict[str, float] = {
    "optimal_clustering": 0.4,  # Small-world clustering coefficient
    "optimal_path_length": 3.2,  # Average shortest path
    "hub_threshold": 0.8,  # Degree centrality for hub classification
    "bridge_threshold": 0.6,  # Betweenness centrality for bridge
    "frontier_threshold": 0.3,  # Low centrality = frontier node
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class BridgeActivation:
    """Record of a bridge node being traversed."""

    bridge_type: BridgeType
    source_domain: str  # Where the signal originates
    target_domain: str  # Where the signal transfers to
    transfer_strength: float  # [0, 1] — how much signal crosses
    evidence: str
    insight: str
    timestamp: float = field(default_factory=time.time)

    def snr_score(self) -> float:
        """SNR = origin_snr × transfer_strength."""
        origin = BRIDGE_ORIGIN_SNR.get(self.bridge_type, 0.95)
        return origin * self.transfer_strength

    def passes_gate(self) -> bool:
        return self.snr_score() >= UNIFIED_SNR_THRESHOLD

    def passes_elite(self) -> bool:
        return self.snr_score() >= SNR_THRESHOLD_T0_ELITE

    def is_ihsan_bridge(self) -> bool:
        """Is this the Ihsān = Level N meta-autopoiesis bridge?"""
        return (
            self.bridge_type == BridgeType.HRM_FOUR_PILLARS
            and "ihsan" in self.target_domain.lower()
        )


@dataclass
class BridgeReport:
    """Summary of bridge node activations."""

    activations: List[BridgeActivation] = field(default_factory=list)
    cycle_id: str = ""
    timestamp: float = field(default_factory=time.time)

    @property
    def active_bridge_count(self) -> int:
        return len({a.bridge_type for a in self.activations})

    @property
    def mean_transfer_strength(self) -> float:
        if not self.activations:
            return 0.0
        return sum(a.transfer_strength for a in self.activations) / len(
            self.activations
        )

    @property
    def mean_snr(self) -> float:
        if not self.activations:
            return 0.0
        return sum(a.snr_score() for a in self.activations) / len(self.activations)

    @property
    def strongest_bridge(self) -> Optional[BridgeType]:
        if not self.activations:
            return None
        best = max(self.activations, key=lambda a: a.snr_score())
        return best.bridge_type

    @property
    def ihsan_meta_active(self) -> bool:
        """Is the Ihsān = Level N meta-autopoiesis bridge active?"""
        return any(a.is_ihsan_bridge() for a in self.activations)

    def gate_report(self) -> Dict[str, Any]:
        passed = [a for a in self.activations if a.passes_gate()]
        elite = [a for a in self.activations if a.passes_elite()]
        return {
            "total_activations": len(self.activations),
            "passed_snr_gate": len(passed),
            "elite_bridges": len(elite),
            "active_bridge_types": self.active_bridge_count,
            "mean_snr": self.mean_snr,
            "mean_transfer_strength": self.mean_transfer_strength,
            "strongest_bridge": (
                self.strongest_bridge.name if self.strongest_bridge else None
            ),
            "ihsan_meta_active": self.ihsan_meta_active,
            "supreme_insight": (
                (
                    "Structure enables capability. Autopoiesis enables evolution. "
                    "The fusion enables transcendence."
                )
                if self.ihsan_meta_active
                else None
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# BRIDGE NODE DETECTOR — Core Engine
# ═══════════════════════════════════════════════════════════════════════════════


class BridgeNodeDetector:
    """Detects cross-domain bridge activations in the BIZRA architecture.

    Each bridge connects two major subsystems and enables information
    transfer that neither system could achieve alone. The detector
    monitors for conditions that indicate a bridge is being traversed.

    Usage:
        detector = BridgeNodeDetector()
        report = detector.analyze_cross_domain(domain_data)
    """

    def __init__(
        self,
        min_transfer: float = 0.3,
        ihsan_floor: float = UNIFIED_IHSAN_THRESHOLD,
    ) -> None:
        self.min_transfer = min_transfer
        self.ihsan_floor = ihsan_floor
        self._history: List[BridgeActivation] = []

    # ─── Bridge 1: Autopoiesis ↔ RDVE Resonance ─────────────────────────
    def detect_autopoiesis_rdve(
        self,
        generator_output_count: int,
        verifier_accept_count: int,
        meta_optimizer_adjustments: int = 0,
    ) -> Optional[BridgeActivation]:
        """Detect PAT=Generator ↔ SAT=Verifier bridge.

        The RDVE bifurcated architecture maps directly to PAT/SAT:
        PAT agents GENERATE, SAT agents VERIFY, Node0 kernel OPTIMIZES.
        """
        if generator_output_count < 1:
            return None

        acceptance_rate = verifier_accept_count / generator_output_count
        optimization_rate = meta_optimizer_adjustments / max(generator_output_count, 1)

        # Bridge strength = balance between generation and verification
        balance = 1.0 - abs(acceptance_rate - 0.5) / 0.5
        transfer = min(1.0, balance * 0.7 + optimization_rate * 0.3)

        if transfer < self.min_transfer:
            return None

        activation = BridgeActivation(
            bridge_type=BridgeType.AUTOPOIESIS_RDVE,
            source_domain="autopoiesis",
            target_domain="rdve",
            transfer_strength=transfer,
            evidence=(
                f"Generator outputs={generator_output_count}, "
                f"Verifier accepts={verifier_accept_count} ({acceptance_rate:.1%}), "
                f"Meta-optimizer adjustments={meta_optimizer_adjustments}"
            ),
            insight=(
                f"PAT→Generator, SAT→Verifier bridge active. "
                f"Balance={balance:.3f}. "
                f"{'Healthy generator-verifier resonance.' if balance > 0.5 else 'Imbalanced — adjust generation/verification ratio.'}"
            ),
        )
        self._history.append(activation)
        return activation

    # ─── Bridge 2: HRM ↔ Four Pillars (Curry-Howard) ────────────────────
    def detect_hrm_pillars(
        self,
        level_states: Dict[str, str],
    ) -> Optional[BridgeActivation]:
        """Detect HRM ↔ Four Pillars mapping (Curry-Howard correspondence).

        L0=Genesis/Sandbox, L1-2=Museum, L3=Runtime, LN=Adaptive Ihsān.
        The isomorphism: propositions (HRM levels) ≅ types (Pillar gates).
        """
        if not level_states:
            return None

        # Count how many levels map correctly to their pillar
        correct_mappings = 0
        total = len(level_states)
        ihsan_detected = False

        for level, state in level_states.items():
            expected_pillar = HRM_PILLAR_MAP.get(level)
            if expected_pillar and expected_pillar in state.lower():
                correct_mappings += 1
            if level == "LN_meta" and "ihsan" in state.lower():
                ihsan_detected = True

        alignment = correct_mappings / max(total, 1)
        transfer = min(1.0, alignment + (0.2 if ihsan_detected else 0.0))

        if transfer < self.min_transfer:
            return None

        target = "adaptive_ihsan" if ihsan_detected else "four_pillars"
        activation = BridgeActivation(
            bridge_type=BridgeType.HRM_FOUR_PILLARS,
            source_domain="hrm",
            target_domain=target,
            transfer_strength=transfer,
            evidence=(
                f"Level-Pillar alignment={alignment:.1%} "
                f"({correct_mappings}/{total} correct), "
                f"Ihsān=LevelN: {ihsan_detected}"
            ),
            insight=(
                f"Curry-Howard isomorphism: {alignment:.0%} alignment. "
                + (
                    "META-DISCOVERY: Ihsān IS Level N autopoiesis — "
                    "ethics IS the system's capacity for self-transcendence."
                    if ihsan_detected
                    else "Strengthen LN→Ihsān mapping for meta-autopoiesis."
                )
            ),
        )
        self._history.append(activation)
        return activation

    # ─── Bridge 3: SNR ↔ Shannon Channel Duality ────────────────────────
    def detect_snr_shannon(
        self,
        noise_classifications: Dict[str, int],
    ) -> Optional[BridgeActivation]:
        """Detect SNR ↔ Shannon noise type mapping.

        Maps BIZRA noise types to Shannon's information-theoretic categories:
          measurement → channel_noise
          model → source_coding_loss
          structural → capacity_mismatch
          epistemological → fundamental_limit
        """
        if not noise_classifications:
            return None

        mapped = 0
        total = sum(noise_classifications.values())

        for noise_type, count in noise_classifications.items():
            if noise_type in SHANNON_NOISE_MAP:
                mapped += count

        coverage = mapped / max(total, 1)
        transfer = min(1.0, coverage)

        if transfer < self.min_transfer:
            return None

        shannon_mapping = {
            noise_type: SHANNON_NOISE_MAP.get(noise_type, "unmapped")
            for noise_type in noise_classifications
        }

        activation = BridgeActivation(
            bridge_type=BridgeType.SNR_SHANNON_CHANNEL,
            source_domain="snr_engine",
            target_domain="shannon_theory",
            transfer_strength=transfer,
            evidence=(
                f"Noise types mapped: {mapped}/{total} ({coverage:.1%}), "
                f"Mapping: {shannon_mapping}"
            ),
            insight=(
                f"Shannon channel duality: {coverage:.0%} noise classification coverage. "
                f"Each noise type maps to a specific information-theoretic category."
            ),
        )
        self._history.append(activation)
        return activation

    # ─── Bridge 4: GoT ↔ Sacred Geometry / Small-World ──────────────────
    def detect_got_topology(
        self,
        clustering_coefficient: float,
        avg_path_length: float,
        hub_count: int = 0,
        bridge_count: int = 0,
        frontier_count: int = 0,
    ) -> Optional[BridgeActivation]:
        """Detect GoT knowledge graph small-world properties.

        Optimal small-world: clustering ≈ 0.4, path_length ≈ 3.2.
        Node types: Hub (high degree), Bridge (high betweenness), Frontier (low centrality).
        """
        # Proximity to optimal small-world values
        cluster_proximity = 1.0 - min(1.0, abs(clustering_coefficient - 0.4) / 0.4)
        path_proximity = 1.0 - min(1.0, abs(avg_path_length - 3.2) / 3.2)

        small_world_score = cluster_proximity * 0.5 + path_proximity * 0.5
        transfer = min(1.0, small_world_score)

        if transfer < self.min_transfer:
            return None

        hub_count + bridge_count + frontier_count
        activation = BridgeActivation(
            bridge_type=BridgeType.GOT_SACRED_GEOMETRY,
            source_domain="got_explorer",
            target_domain="knowledge_topology",
            transfer_strength=transfer,
            evidence=(
                f"Clustering={clustering_coefficient:.3f} (optimal 0.4), "
                f"Path={avg_path_length:.2f} (optimal 3.2), "
                f"Nodes: {hub_count} hubs, {bridge_count} bridges, {frontier_count} frontiers"
            ),
            insight=(
                f"Small-world score={small_world_score:.3f}. "
                f"{'Graph exhibits sacred geometry — optimal knowledge topology.' if small_world_score > 0.7 else 'Graph topology needs rebalancing toward small-world properties.'}"
            ),
        )
        self._history.append(activation)
        return activation

    # ─── Bridge 5: Compound Learning ↔ Recursive Acceleration ───────────
    def detect_compound_recursive(
        self,
        learning_rates: List[float],
        sigma_squared_over_s: float = 0.0,
    ) -> Optional[BridgeActivation]:
        """Detect compound learning with punctuated equilibrium.

        C(t+dt) = C(t) + g(C(t))·dt
        σ²/s ≈ 2.3 indicates punctuated equilibrium mode.
        This is the HIGHEST SNR bridge (0.99).
        """
        if len(learning_rates) < 3:
            return None

        # Check for compounding: second derivative > 0
        velocities = [
            learning_rates[i + 1] - learning_rates[i]
            for i in range(len(learning_rates) - 1)
        ]
        accelerations = [
            velocities[i + 1] - velocities[i] for i in range(len(velocities) - 1)
        ]

        mean_accel = sum(accelerations) / len(accelerations) if accelerations else 0.0
        is_compound = mean_accel > 0

        # Punctuated equilibrium check
        equilibrium_proximity = 1.0 - min(1.0, abs(sigma_squared_over_s - 2.3) / 2.3)

        transfer = min(
            1.0, ((0.5 if is_compound else 0.2) + equilibrium_proximity * 0.5)
        )

        if transfer < self.min_transfer:
            return None

        activation = BridgeActivation(
            bridge_type=BridgeType.COMPOUND_RECURSIVE,
            source_domain="learning_dynamics",
            target_domain="recursive_acceleration",
            transfer_strength=transfer,
            evidence=(
                f"Compounding={'yes' if is_compound else 'no'}, "
                f"acceleration={mean_accel:.5f}, "
                f"σ²/s={sigma_squared_over_s:.3f} (optimal 2.3), "
                f"equilibrium_proximity={equilibrium_proximity:.3f}"
            ),
            insight=(
                f"{'Compound recursive acceleration ACTIVE' if is_compound else 'Learning not yet compounding'}. "
                f"σ²/s proximity to punctuated equilibrium: {equilibrium_proximity:.1%}. "
                f"{'System at criticality — maximum learning potential.' if equilibrium_proximity > 0.7 else 'Push toward σ²/s ≈ 2.3 for punctuated equilibrium.'}"
            ),
        )
        self._history.append(activation)
        return activation

    # ─── Full Analysis ──────────────────────────────────────────────────
    def analyze_cross_domain(
        self,
        domain_data: Dict[str, Any],
        cycle_id: str = "",
    ) -> BridgeReport:
        """Run all 5 bridge detectors on cross-domain data.

        Args:
            domain_data: Dict with keys matching detector parameters.
            cycle_id: Identifier for this analysis cycle.

        Returns:
            BridgeReport with all activations.
        """
        report = BridgeReport(cycle_id=cycle_id)

        # Bridge 1
        if "generator_output_count" in domain_data:
            result = self.detect_autopoiesis_rdve(
                domain_data["generator_output_count"],
                domain_data.get("verifier_accept_count", 0),
                domain_data.get("meta_optimizer_adjustments", 0),
            )
            if result:
                report.activations.append(result)

        # Bridge 2
        if "level_states" in domain_data:
            result = self.detect_hrm_pillars(domain_data["level_states"])
            if result:
                report.activations.append(result)

        # Bridge 3
        if "noise_classifications" in domain_data:
            result = self.detect_snr_shannon(domain_data["noise_classifications"])
            if result:
                report.activations.append(result)

        # Bridge 4
        if "clustering_coefficient" in domain_data and "avg_path_length" in domain_data:
            result = self.detect_got_topology(
                domain_data["clustering_coefficient"],
                domain_data["avg_path_length"],
                domain_data.get("hub_count", 0),
                domain_data.get("bridge_count", 0),
                domain_data.get("frontier_count", 0),
            )
            if result:
                report.activations.append(result)

        # Bridge 5
        if "learning_rates" in domain_data:
            result = self.detect_compound_recursive(
                domain_data["learning_rates"],
                domain_data.get("sigma_squared_over_s", 0.0),
            )
            if result:
                report.activations.append(result)

        return report

    @property
    def history(self) -> List[BridgeActivation]:
        return list(self._history)

    def reset_history(self) -> int:
        count = len(self._history)
        self._history.clear()
        return count
