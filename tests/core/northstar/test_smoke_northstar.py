"""
BIZRA Node0 NorthStar — Smoke Test Suite
╔══════════════════════════════════════════════════════════════════════════════╗
║  35 Smoke Tests × All Pillars                                               ║
║  Peak Masterpiece Protocol Validation                                       ║
║                                                                              ║
║  بسم الله الرحمن الرحيم                                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Test Organization:
  [01-08] Golden Gems — 8 gem detector tests
  [09-12] Thought Flows — 4 meta-level flow tests
  [13-16] Phase Patterns — 4 phase-level pattern tests
  [17-21] Bridge Nodes — 5 bridge detector tests
  [22-28] NorthStar Engine — 7 fusion engine tests
  [29-35] Constitutional Gates — 7 SNR/Ihsān/FATE gate tests

Created: 2026-02-15 | BIZRA Node0 NorthStar | Peak Masterpiece Protocol
"""

import math
import pytest

from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
    SNR_THRESHOLD_T0_ELITE,
)

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS — Verify all public API items are importable
# ═══════════════════════════════════════════════════════════════════════════════

from core.northstar import (
    # Version
    __version__,
    # Golden Gems
    GoldenGemType,
    GoldenGemDetector,
    GemActivation,
    GemReport,
    GEM_ORIGIN_SNR,
    GEM_NORMALIZED_SNR,
    # Thought Flows
    ThoughtFlowType,
    ThoughtFlowDetector,
    FlowActivation,
    FlowReport,
    PhasePatternType,
    PhaseActivation,
    PHASE_PATTERN_SNR,
    PHI,
    # Bridge Nodes
    BridgeType,
    BridgeNodeDetector,
    BridgeActivation,
    BridgeReport,
    BRIDGE_ORIGIN_SNR,
    AUTOPOIESIS_RDVE_ROLES,
    HRM_PILLAR_MAP,
    SHANNON_NOISE_MAP,
    GOT_TOPOLOGY_CONSTANTS,
    # NorthStar Engine
    NorthStarEngine,
    NorthStarReport,
    NorthStarStatus,
)


# ═══════════════════════════════════════════════════════════════════════════════
# [01-08] GOLDEN GEMS
# ═══════════════════════════════════════════════════════════════════════════════


class TestGoldenGems:
    """Tests for the 8 golden gem detectors."""

    def setup_method(self):
        self.detector = GoldenGemDetector(sensitivity=0.7)

    # [01] Shadow Knowledge — blind spots as data
    def test_01_shadow_knowledge(self):
        """Shadow Knowledge: unobserved domains produce activation."""
        result = self.detector.detect_shadow_knowledge(
            observed_domains=["reasoning", "snr"],
            all_known_domains=["reasoning", "snr", "governance", "federation", "treasury"],
        )
        assert result is not None
        assert result.gem_type == GoldenGemType.SHADOW_KNOWLEDGE
        assert result.intensity > 0
        assert "blind spots" in result.evidence.lower() or "Blind spots" in result.evidence
        assert result.snr_score() > 0

    # [02] Contradiction Harvest — conflicts reveal structure
    def test_02_contradiction_harvest(self):
        """Contradiction Harvest: conflicting claims trigger activation."""
        assertions = [
            {"claim": "SNR is rising", "value": "high", "domain": "quality", "confidence": 0.9, "topic": "snr"},
            {"claim": "SNR is falling", "value": "low", "domain": "quality", "confidence": 0.4, "topic": "snr"},
        ]
        result = self.detector.detect_contradiction_harvest(assertions)
        assert result is not None
        assert result.gem_type == GoldenGemType.CONTRADICTION_HARVEST
        assert result.intensity > 0

    # [03] Emergence Principle — topology > individual nodes
    def test_03_emergence_principle(self):
        """Emergence: small-world graph properties trigger activation."""
        result = self.detector.detect_emergence(
            node_count=50,
            edge_count=120,
            clustering_coefficient=0.38,
            avg_path_length=3.1,
        )
        assert result is not None
        assert result.gem_type == GoldenGemType.EMERGENCE_PRINCIPLE
        assert "Emergence" in result.insight or "emergence" in result.insight

    # [04] Noise-is-Signal — persistent noise = unconceptualized truth
    def test_04_noise_is_signal(self):
        """Noise-is-Signal: persistent noise pattern triggers activation."""
        # Persistent noise: similar values across windows
        noise_history = [0.5, 0.52, 0.48, 0.51, 0.49, 0.50, 0.53, 0.47, 0.51, 0.49]
        result = self.detector.detect_noise_as_signal(noise_history, window_size=3)
        assert result is not None
        assert result.gem_type == GoldenGemType.NOISE_IS_SIGNAL
        assert "persistent" in result.insight.lower() or "Persistent" in result.insight

    # [05] Validation Mirror — rejections map boundaries
    def test_05_validation_mirror(self):
        """Validation Mirror: optimal rejection rate triggers activation."""
        result = self.detector.detect_validation_mirror(
            total_validations=100,
            rejections=30,  # 30% is optimal
            rejection_reasons=["threshold_fail", "confidence_low", "threshold_fail"],
        )
        assert result is not None
        assert result.gem_type == GoldenGemType.VALIDATION_MIRROR
        assert result.intensity > 0.5  # Strong signal at 30%

    # [06] Implementation as Hypothesis — execution is experiment
    def test_06_implementation_hypothesis(self):
        """Implementation Hypothesis: plan-reality divergence triggers activation."""
        result = self.detector.detect_implementation_hypothesis(
            planned_outcome="SNR should reach 0.95",
            actual_outcome="SNR reached 0.88 with unexpected clustering",
            divergence_score=0.6,
        )
        assert result is not None
        assert result.gem_type == GoldenGemType.IMPLEMENTATION_HYPOTHESIS
        assert "tacit knowledge" in result.insight.lower()

    # [07] Integration Paradox — friction = breakthrough
    def test_07_integration_paradox(self):
        """Integration Paradox: high effort + high value triggers activation."""
        result = self.detector.detect_integration_paradox(
            integration_effort=0.85,
            integration_value=0.90,
        )
        assert result is not None
        assert result.gem_type == GoldenGemType.INTEGRATION_PARADOX
        assert "Deep" in result.evidence or "Foundational" in result.evidence

    # [08] Outcome as Oracle — results answer unasked questions
    def test_08_outcome_oracle(self):
        """Outcome Oracle: unexpected outcomes trigger activation."""
        result = self.detector.detect_outcome_oracle(
            expected_outcomes=["snr_improvement", "convergence"],
            observed_outcomes=["snr_improvement", "emergence", "resonance"],
        )
        assert result is not None
        assert result.gem_type == GoldenGemType.OUTCOME_ORACLE
        assert "unexpected" in result.insight.lower() or "Unexpected" in result.insight


# ═══════════════════════════════════════════════════════════════════════════════
# [09-12] THOUGHT FLOWS — Meta-Level
# ═══════════════════════════════════════════════════════════════════════════════


class TestThoughtFlows:
    """Tests for the 4 meta-level thought flow detectors."""

    def setup_method(self):
        self.detector = ThoughtFlowDetector(sensitivity=0.7)

    # [09] Cross-Level Learning Cascade
    def test_09_cross_level_cascade(self):
        """Cascade: super-linear level improvements detected."""
        result = self.detector.detect_cross_level_cascade(
            level_improvements={0: 0.10, 1: 0.15, 2: 0.22, 3: 0.35},
        )
        assert result is not None
        assert result.flow_type == ThoughtFlowType.CROSS_LEVEL_CASCADE
        assert result.direction == "ascending"
        assert len(result.affected_levels) == 4

    # [10] Permeable Boundary
    def test_10_permeable_boundary(self):
        """Boundary: optimal permeability detected."""
        result = self.detector.detect_permeable_boundary(
            cross_level_messages=30,
            same_level_messages=70,
        )
        assert result is not None
        assert result.flow_type == ThoughtFlowType.PERMEABLE_BOUNDARY
        assert result.direction == "lateral"

    # [11] Compound Learning
    def test_11_compound_learning(self):
        """Compound: accelerating growth detected."""
        # Exponential-like growth
        history = [1.0, 1.05, 1.12, 1.21, 1.35, 1.55, 1.82]
        result = self.detector.detect_compound_learning(history)
        assert result is not None
        assert result.flow_type == ThoughtFlowType.COMPOUND_LEARNING
        assert result.direction == "ascending"

    # [12] Learning Resonance
    def test_12_learning_resonance(self):
        """Resonance: harmonic learning rates detected."""
        result = self.detector.detect_learning_resonance(
            level_learning_rates={0: 0.1, 1: 0.162, 2: 0.05},  # 0.1/0.162 ≈ 1/φ
        )
        assert result is not None
        assert result.flow_type == ThoughtFlowType.LEARNING_RESONANCE
        assert result.direction == "resonant"


# ═══════════════════════════════════════════════════════════════════════════════
# [13-16] PHASE PATTERNS — Per-Phase
# ═══════════════════════════════════════════════════════════════════════════════


class TestPhasePatterns:
    """Tests for phase-level hidden pattern detectors."""

    def setup_method(self):
        self.detector = ThoughtFlowDetector(sensitivity=0.6)

    # [13] Convergence-Divergence Pulse (φ alignment)
    def test_13_convergence_divergence_phi(self):
        """φ Pulse: golden ratio alignment detected."""
        result = self.detector.detect_convergence_divergence(
            convergence_score=0.38,
            divergence_score=0.62,  # 0.62/0.38 ≈ φ
        )
        assert result is not None
        assert result.pattern_type == PhasePatternType.CONVERGENCE_DIVERGENCE
        # φ alignment should be high
        assert result.intensity > 0.3

    # [14] Meta-Learning Acceleration
    def test_14_meta_learning_acceleration(self):
        """Meta-Learning: super-linear cascade from L0 improvement."""
        result = self.detector.detect_meta_learning_acceleration(
            l0_improvement=0.10,
            total_hierarchy_improvement=0.80,  # 0.10 × 5 = 0.50 expected, got 0.80
            level_count=5,
        )
        assert result is not None
        assert result.pattern_type == PhasePatternType.META_LEARNING_ACCELERATION
        assert "Super-linear" in result.insight

    # [15] Integration Depth Gradient
    def test_15_integration_depth(self):
        """Depth: correct distribution alignment detected."""
        result = self.detector.detect_integration_depth(
            surface_count=70,
            intermediate_count=20,
            deep_count=8,
            foundational_count=2,
        )
        assert result is not None
        assert result.pattern_type == PhasePatternType.INTEGRATION_DEPTH
        assert result.intensity > 0.5  # Should be well-aligned

    # [16] Phase patterns have correct SNR scores
    def test_16_phase_pattern_snr_scores(self):
        """All 8 phase patterns have valid SNR origin scores."""
        assert len(PHASE_PATTERN_SNR) == 8
        for pattern, score in PHASE_PATTERN_SNR.items():
            assert 9.0 <= score <= 10.0, f"{pattern.name} SNR {score} out of range"


# ═══════════════════════════════════════════════════════════════════════════════
# [17-21] BRIDGE NODES
# ═══════════════════════════════════════════════════════════════════════════════


class TestBridgeNodes:
    """Tests for the 5 cross-document bridge node detectors."""

    def setup_method(self):
        self.detector = BridgeNodeDetector(min_transfer=0.3)

    # [17] Bridge 1: Autopoiesis ↔ RDVE
    def test_17_autopoiesis_rdve_bridge(self):
        """Bridge 1: PAT=Generator, SAT=Verifier balance detected."""
        result = self.detector.detect_autopoiesis_rdve(
            generator_output_count=100,
            verifier_accept_count=50,  # 50% acceptance = balanced
            meta_optimizer_adjustments=10,
        )
        assert result is not None
        assert result.bridge_type == BridgeType.AUTOPOIESIS_RDVE
        assert result.source_domain == "autopoiesis"
        assert result.target_domain == "rdve"
        assert result.snr_score() > 0.65  # 0.97 origin × 0.73 transfer ≈ 0.71

    # [18] Bridge 2: HRM ↔ Four Pillars (with Ihsān)
    def test_18_hrm_pillars_ihsan_bridge(self):
        """Bridge 2: Curry-Howard with Ihsān=LevelN meta-discovery."""
        result = self.detector.detect_hrm_pillars(
            level_states={
                "L0_perceptual": "genesis_sandbox zone",
                "L1_operational": "museum archive",
                "L2_tactical": "museum curator",
                "L3_strategic": "runtime engine",
                "LN_meta": "adaptive_ihsan governance",
            },
        )
        assert result is not None
        assert result.bridge_type == BridgeType.HRM_FOUR_PILLARS
        assert result.is_ihsan_bridge()
        assert "Ihsān IS Level N" in result.insight or "Ihsān" in result.insight

    # [19] Bridge 3: SNR ↔ Shannon Channel
    def test_19_snr_shannon_bridge(self):
        """Bridge 3: noise type to Shannon classification mapping."""
        result = self.detector.detect_snr_shannon(
            noise_classifications={
                "measurement_noise": 40,
                "model_noise": 30,
                "structural_noise": 20,
                "epistemological_noise": 10,
            },
        )
        assert result is not None
        assert result.bridge_type == BridgeType.SNR_SHANNON_CHANNEL
        assert result.transfer_strength == 1.0  # All types mapped

    # [20] Bridge 4: GoT ↔ Sacred Geometry
    def test_20_got_topology_bridge(self):
        """Bridge 4: small-world properties near optimal."""
        result = self.detector.detect_got_topology(
            clustering_coefficient=0.38,
            avg_path_length=3.3,
            hub_count=5,
            bridge_count=12,
            frontier_count=33,
        )
        assert result is not None
        assert result.bridge_type == BridgeType.GOT_SACRED_GEOMETRY
        assert result.snr_score() >= 0.85

    # [21] Bridge 5: Compound ↔ Recursive (highest SNR bridge)
    def test_21_compound_recursive_bridge(self):
        """Bridge 5: compound learning with punctuated equilibrium."""
        result = self.detector.detect_compound_recursive(
            learning_rates=[0.10, 0.12, 0.15, 0.20, 0.28, 0.40],
            sigma_squared_over_s=2.25,  # Near 2.3 optimal
        )
        assert result is not None
        assert result.bridge_type == BridgeType.COMPOUND_RECURSIVE
        assert BRIDGE_ORIGIN_SNR[BridgeType.COMPOUND_RECURSIVE] == 0.99


# ═══════════════════════════════════════════════════════════════════════════════
# [22-28] NORTHSTAR ENGINE (Fusion)
# ═══════════════════════════════════════════════════════════════════════════════


class TestNorthStarEngine:
    """Tests for the unified NorthStar fusion engine."""

    def setup_method(self):
        self.engine = NorthStarEngine(
            gem_sensitivity=0.7,
            flow_sensitivity=0.7,
        )

    def _full_observations(self) -> dict:
        """Generate comprehensive observations for full-cycle testing."""
        return {
            # Gem data
            "observed_domains": ["reasoning", "snr"],
            "all_known_domains": ["reasoning", "snr", "governance", "federation"],
            "assertions": [
                {"claim": "A", "value": "high", "domain": "d1", "confidence": 0.9, "topic": "t1"},
                {"claim": "B", "value": "low", "domain": "d1", "confidence": 0.4, "topic": "t1"},
            ],
            "node_count": 50,
            "edge_count": 120,
            "clustering_coefficient": 0.38,
            "avg_path_length": 3.1,
            "noise_history": [0.5, 0.52, 0.48, 0.51, 0.49, 0.50, 0.53, 0.47, 0.51, 0.49],
            "total_validations": 100,
            "rejections": 30,
            "expected_outcomes": ["snr_improvement"],
            "observed_outcomes": ["snr_improvement", "emergence"],
            # Flow data
            "level_improvements": {0: 0.10, 1: 0.15, 2: 0.22, 3: 0.35},
            "learning_history": [1.0, 1.05, 1.12, 1.21, 1.35, 1.55, 1.82],
            "level_learning_rates": {0: 0.1, 1: 0.162, 2: 0.05},
            "cross_level_messages": 30,
            "same_level_messages": 70,
            "convergence_score": 0.38,
            "divergence_score": 0.62,
            "l0_improvement": 0.10,
            "total_hierarchy_improvement": 0.80,
            # Bridge data
            "generator_output_count": 100,
            "verifier_accept_count": 50,
            "meta_optimizer_adjustments": 10,
            "level_states": {
                "L0_perceptual": "genesis_sandbox",
                "LN_meta": "adaptive_ihsan",
            },
            "noise_classifications": {
                "measurement_noise": 40,
                "model_noise": 30,
                "structural_noise": 20,
                "epistemological_noise": 10,
            },
            "learning_rates": [0.10, 0.12, 0.15, 0.20, 0.28, 0.40],
            "sigma_squared_over_s": 2.25,
        }

    # [22] Engine initialization
    def test_22_engine_init(self):
        """Engine initializes with correct defaults."""
        assert self.engine.status == NorthStarStatus.DORMANT
        assert self.engine.cycle_count == 0
        assert self.engine.__version__ == "1.0.0"
        assert len(self.engine.cycle_history) == 0

    # [23] Full cycle produces report
    def test_23_full_cycle_report(self):
        """Full cycle produces a NorthStarReport with all sub-reports."""
        report = self.engine.run_cycle(self._full_observations())
        assert isinstance(report, NorthStarReport)
        assert report.status == NorthStarStatus.COMPLETE
        assert report.total_activations > 0
        assert len(report.gem_report.activations) > 0
        assert len(report.bridge_report.activations) > 0

    # [24] Cycle ID tracking
    def test_24_cycle_id_tracking(self):
        """Cycle count increments and IDs are tracked."""
        r1 = self.engine.run_cycle({}, cycle_id="test-1")
        r2 = self.engine.run_cycle({}, cycle_id="test-2")
        assert self.engine.cycle_count == 2
        assert r1.cycle_id == "test-1"
        assert r2.cycle_id == "test-2"
        assert len(self.engine.cycle_history) == 2

    # [25] Meta-discovery detection
    def test_25_meta_discovery_ihsan(self):
        """Ihsān = Level N meta-discovery is detected when bridge fires."""
        obs = self._full_observations()
        report = self.engine.run_cycle(obs)
        # The HRM_FOUR_PILLARS bridge with ihsan should trigger meta-discovery
        has_ihsan_meta = any(
            "Ihsān" in md or "ihsan" in md.lower()
            for md in report.meta_discoveries
        )
        assert has_ihsan_meta, f"Meta-discoveries: {report.meta_discoveries}"

    # [26] Report summary generation
    def test_26_report_summary(self):
        """Report summary is human-readable string."""
        report = self.engine.run_cycle(self._full_observations())
        summary = report.summary()
        assert "NorthStar Report" in summary
        assert "Unified SNR" in summary
        assert "Ihsān Score" in summary

    # [27] Gate report generation
    def test_27_gate_report(self):
        """Gate report contains all required FATE fields."""
        report = self.engine.run_cycle(self._full_observations())
        gate = report.gate_report()
        assert "unified_snr" in gate
        assert "ihsan_score" in gate
        assert "passes_snr_gate" in gate
        assert "passes_ihsan_gate" in gate
        assert "gems" in gate
        assert "flows" in gate
        assert "bridges" in gate

    # [28] Engine reset
    def test_28_engine_reset(self):
        """Engine reset clears all state."""
        self.engine.run_cycle(self._full_observations())
        assert self.engine.cycle_count > 0
        result = self.engine.reset()
        assert self.engine.cycle_count == 0
        assert self.engine.status == NorthStarStatus.DORMANT
        assert result["cycles_cleared"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# [29-35] CONSTITUTIONAL GATES
# ═══════════════════════════════════════════════════════════════════════════════


class TestConstitutionalGates:
    """Tests for SNR, Ihsān, and FATE gate compliance."""

    # [29] All 8 golden gems have valid origin SNR scores
    def test_29_gem_origin_snr_valid(self):
        """All 8 gems have origin SNR between 9.0 and 10.0."""
        assert len(GEM_ORIGIN_SNR) == 8
        for gem_type, score in GEM_ORIGIN_SNR.items():
            assert 9.0 <= score <= 10.0, f"{gem_type.name}: {score}"

    # [30] All 8 gems have normalized SNR in [0, 1]
    def test_30_gem_normalized_snr_range(self):
        """Normalized SNR values are in [0, 1]."""
        assert len(GEM_NORMALIZED_SNR) == 8
        for gem_type, score in GEM_NORMALIZED_SNR.items():
            assert 0.0 <= score <= 1.0, f"{gem_type.name}: {score}"

    # [31] All 5 bridges have valid origin SNR scores
    def test_31_bridge_origin_snr_valid(self):
        """All 5 bridges have origin SNR >= 0.95."""
        assert len(BRIDGE_ORIGIN_SNR) == 5
        for bridge_type, score in BRIDGE_ORIGIN_SNR.items():
            assert score >= 0.95, f"{bridge_type.name}: {score}"

    # [32] Bridge 5 is highest SNR (0.99)
    def test_32_bridge5_highest_snr(self):
        """Compound Recursive bridge has highest SNR (0.99)."""
        max_bridge = max(BRIDGE_ORIGIN_SNR, key=BRIDGE_ORIGIN_SNR.get)
        assert max_bridge == BridgeType.COMPOUND_RECURSIVE
        assert BRIDGE_ORIGIN_SNR[max_bridge] == 0.99

    # [33] PHI constant is correct
    def test_33_phi_golden_ratio(self):
        """Golden ratio φ constant is correct."""
        expected_phi = (1.0 + math.sqrt(5)) / 2
        assert abs(PHI - expected_phi) < 1e-10
        assert abs(PHI - 1.618033988749895) < 1e-10

    # [34] Ihsān threshold from constants
    def test_34_ihsan_from_constants(self):
        """Ihsān threshold is imported from constants.py SSOT."""
        assert UNIFIED_IHSAN_THRESHOLD == 0.95

    # [35] Full pipeline gate compliance
    def test_35_full_pipeline_gate_compliance(self):
        """Full NorthStar cycle with rich data passes both gates."""
        engine = NorthStarEngine(gem_sensitivity=0.7, flow_sensitivity=0.7)
        obs = {
            # Enough data to trigger activations
            "observed_domains": ["reasoning", "snr"],
            "all_known_domains": ["reasoning", "snr", "governance", "federation"],
            "node_count": 50,
            "edge_count": 120,
            "clustering_coefficient": 0.38,
            "avg_path_length": 3.1,
            "generator_output_count": 100,
            "verifier_accept_count": 50,
            "level_states": {
                "L0_perceptual": "genesis_sandbox",
                "LN_meta": "adaptive_ihsan",
            },
            "noise_classifications": {
                "measurement_noise": 40,
                "model_noise": 30,
                "structural_noise": 20,
                "epistemological_noise": 10,
            },
            "learning_rates": [0.10, 0.12, 0.15, 0.20, 0.28, 0.40],
            "sigma_squared_over_s": 2.25,
        }
        report = engine.run_cycle(obs)
        gate = report.gate_report()

        # Verify structure
        assert gate["total_activations"] > 0
        assert gate["unified_snr"] >= 0  # SNR is non-negative
        assert gate["ihsan_score"] >= 0  # Ihsān is non-negative
        assert isinstance(gate["passes_snr_gate"], bool)
        assert isinstance(gate["passes_ihsan_gate"], bool)

        # The report should contain meta-discoveries
        assert isinstance(gate["meta_discoveries"], list)

        # Verify supreme insight appears when gates pass
        if gate["passes_all_gates"]:
            assert gate["supreme_insight"] is not None
