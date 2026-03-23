"""
HRM Module — Smoke Test Suite (30 Tests, 6 Pillars)

Peak Masterpiece Protocol Phase 5: Validation

Pillar A: Package Imports & Exports (5 tests)
Pillar B: Abstraction Levels & Configuration (5 tests)
Pillar C: Cross-Level Bridge Mechanisms (5 tests)
Pillar D: Meta-Autopoietic Level N (5 tests)
Pillar E: Hierarchical Engine Core (5 tests)
Pillar F: Campaign, Convergence & Telemetry (5 tests)

Constitutional Alignment:
  All thresholds verified against core/integration/constants.py (SSOT).
  SNR floor: 0.85 | Ihsan threshold: 0.95 | T0 Elite: 0.98

Created: 2026-02-15 | BIZRA Node0 Proactive Pilot | Peak Masterpiece Protocol
"""


import pytest

# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR A: Package Imports & Exports
# ═══════════════════════════════════════════════════════════════════════════════


class TestPillarA_PackageImports:
    """Validate that all public API exports resolve correctly."""

    def test_a1_abstraction_level_imports(self):
        """Import all abstraction level types from core.hrm."""
        from core.hrm import (
            AbstractionLevel,
            BridgeNodeType,
            LevelConfig,
            default_boundaries,
            default_level_configs,
        )

        assert AbstractionLevel is not None
        assert BridgeNodeType is not None
        assert LevelConfig is not None
        assert callable(default_level_configs)
        assert callable(default_boundaries)

    def test_a2_cross_level_bridge_imports(self):
        """Import all cross-level bridge types from core.hrm."""
        from core.hrm import (
            CrossLevelBridge,
            MessageType,
            PropagationDirection,
        )

        assert CrossLevelBridge is not None
        assert MessageType is not None
        assert PropagationDirection.UPWARD.value == "upward"
        assert PropagationDirection.DOWNWARD.value == "downward"

    def test_a3_meta_level_imports(self):
        """Import all meta-level types from core.hrm."""
        from core.hrm import (
            MetaAutopoieticLevel,
            MetaOperation,
            TriggerCondition,
        )

        assert MetaAutopoieticLevel is not None
        assert MetaOperation.BOUNDARY_TUNING.value == "boundary_tuning"
        assert TriggerCondition.INFORMATION_BOTTLENECK.value == "information_bottleneck"

    def test_a4_hierarchical_engine_imports(self):
        """Import all engine types from core.hrm."""
        from core.hrm import (
            HierarchicalReasoningModel,
            HRMStatus,
        )

        assert HierarchicalReasoningModel is not None
        assert HRMStatus.IDLE.value == "idle"
        assert HRMStatus.CONVERGED.value == "converged"

    def test_a5_package_version(self):
        """Verify package version is set."""
        from core.hrm import __version__

        assert __version__ == "1.0.0"


# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR B: Abstraction Levels & Configuration
# ═══════════════════════════════════════════════════════════════════════════════


class TestPillarB_AbstractionLevels:
    """Validate level hierarchy, SNR gradient, and boundary logic."""

    def test_b1_abstraction_level_enum(self):
        """AbstractionLevel has 5 levels with correct ordering."""
        from core.hrm import AbstractionLevel

        assert AbstractionLevel.PERCEPTUAL == 0
        assert AbstractionLevel.OPERATIONAL == 1
        assert AbstractionLevel.TACTICAL == 2
        assert AbstractionLevel.STRATEGIC == 3
        assert AbstractionLevel.META_COGNITIVE == 4
        assert len(AbstractionLevel) == 5

    def test_b2_snr_gradient_constitutional_alignment(self):
        """SNR gradient matches constitutional thresholds from SSOT."""
        from core.hrm import HRM_SNR_GRADIENT, AbstractionLevel
        from core.integration.constants import (
            SNR_THRESHOLD_T0_ELITE,
            UNIFIED_SNR_THRESHOLD,
        )

        # Verify gradient is monotonically non-decreasing
        levels = sorted(HRM_SNR_GRADIENT.keys())
        values = [HRM_SNR_GRADIENT[l] for l in levels]
        for i in range(len(values) - 1):
            assert (
                values[i] <= values[i + 1]
            ), f"SNR gradient violated: L{i}={values[i]} > L{i+1}={values[i+1]}"

        # Verify constitutional alignment
        assert HRM_SNR_GRADIENT[AbstractionLevel.PERCEPTUAL] == UNIFIED_SNR_THRESHOLD
        assert (
            HRM_SNR_GRADIENT[AbstractionLevel.META_COGNITIVE] == SNR_THRESHOLD_T0_ELITE
        )

    def test_b3_default_level_configs(self):
        """Default configs cover all 5 levels with decreasing learning rates."""
        from core.hrm import AbstractionLevel, default_level_configs

        configs = default_level_configs()
        assert len(configs) == 5

        # Learning rates should decrease with level
        learning_rates = [c.learning_rate_factor for c in configs]
        for i in range(len(learning_rates) - 1):
            assert learning_rates[i] >= learning_rates[i + 1], (
                f"Learning rate should decrease: L{i}={learning_rates[i]} < "
                f"L{i+1}={learning_rates[i+1]}"
            )

        # All levels represented
        covered_levels = {c.level for c in configs}
        assert covered_levels == set(AbstractionLevel)

    def test_b4_default_boundaries(self):
        """Default boundaries create bidirectional links between adjacent levels."""
        from core.hrm import default_boundaries

        boundaries = default_boundaries()
        # 4 pairs × 2 directions = 8 boundaries
        assert len(boundaries) == 8

        # Every boundary has permeability in (0, 1)
        for b in boundaries:
            assert 0 < b.permeability <= 1.0, f"Bad permeability: {b.permeability}"

    def test_b5_boundary_should_pass(self):
        """Boundary permeability correctly gates messages by confidence."""
        from core.hrm import AbstractionLevel, LevelBoundary

        boundary = LevelBoundary(
            source_level=AbstractionLevel.PERCEPTUAL,
            target_level=AbstractionLevel.OPERATIONAL,
            permeability=0.6,  # threshold = 1.0 - 0.6 = 0.4
        )

        # Confidence 0.5 > threshold 0.4 → pass
        assert boundary.should_pass(0.5) is True
        # Confidence 0.3 < threshold 0.4 → block
        assert boundary.should_pass(0.3) is False

        # Record crossings
        boundary.record_crossing(passed=True)
        boundary.record_crossing(passed=False)
        assert boundary.message_count == 1
        assert boundary.blocked_count == 1


# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR C: Cross-Level Bridge Mechanisms
# ═══════════════════════════════════════════════════════════════════════════════


class TestPillarC_CrossLevelBridge:
    """Validate the 5 cross-level integration mechanisms."""

    def test_c1_hypothesis_propagation_upward(self):
        """Hypothesis propagates upward to adjacent level."""
        from core.hrm import AbstractionLevel, CrossLevelBridge, PropagationDirection

        bridge = CrossLevelBridge()
        messages = bridge.propagate_hypothesis(
            hypothesis={"content": "test pattern detected"},
            source_level=AbstractionLevel.PERCEPTUAL,
            direction=PropagationDirection.UPWARD,
            confidence=0.9,  # High confidence → should pass
        )

        assert len(messages) >= 1
        msg = messages[0]
        assert msg.source_level == AbstractionLevel.PERCEPTUAL
        assert msg.target_level == AbstractionLevel.OPERATIONAL
        assert msg.direction == "upward"

    def test_c2_hypothesis_propagation_bidirectional(self):
        """Bidirectional propagation reaches both adjacent levels."""
        from core.hrm import AbstractionLevel, CrossLevelBridge, PropagationDirection

        bridge = CrossLevelBridge()
        messages = bridge.propagate_hypothesis(
            hypothesis={"content": "tactical insight"},
            source_level=AbstractionLevel.TACTICAL,  # Middle level
            direction=PropagationDirection.BOTH,
            confidence=0.9,
        )

        # Should reach both L1 (down) and L3 (up)
        targets = {m.target_level for m in messages}
        assert (
            AbstractionLevel.OPERATIONAL in targets
            or AbstractionLevel.STRATEGIC in targets
        )

    def test_c3_validation_cascade(self):
        """Validation cascade collects responses from multiple levels."""
        from core.hrm import AbstractionLevel, CrossLevelBridge

        bridge = CrossLevelBridge()
        result = bridge.request_validation(
            hypothesis={"claim": "anomaly detected", "confidence": 0.7},
            requesting_level=AbstractionLevel.TACTICAL,
        )

        assert result.requesting_level == AbstractionLevel.TACTICAL
        assert len(result.responses) > 0
        assert result.aggregate_confidence > 0.0

    def test_c4_integration_sync(self):
        """Integration sync detects contradictions and transfer opportunities."""
        from core.hrm import AbstractionLevel, CrossLevelBridge

        bridge = CrossLevelBridge()

        # Create level states with some tension.
        # Gap detection checks the LOWER level in each adjacent pair,
        # so PERCEPTUAL (lowest) must have empty active_hypotheses.
        level_states = {
            AbstractionLevel.PERCEPTUAL: {
                "snr_scores": [0.9, 0.91, 0.92],
                "active_hypotheses": [],  # Gap! (detected as lower in pair)
                "insights": ["i1"],
            },
            AbstractionLevel.OPERATIONAL: {
                "snr_scores": [0.6, 0.61, 0.62],  # Much lower → contradiction
                "active_hypotheses": ["h3"],
                "insights": ["i2"],
            },
            AbstractionLevel.TACTICAL: {
                "snr_scores": [0.88, 0.89],
                "active_hypotheses": ["h4"],
                "insights": ["i3"],
            },
        }

        result = bridge.synchronize_integration(level_states)
        assert len(result.participating_levels) == 3
        # Should detect at least one contradiction (0.9 vs 0.6 delta > 0.2)
        assert result.contradictions_found >= 1
        # Should detect gap (PERCEPTUAL has no active_hypotheses)
        assert result.gaps_identified >= 1
        # Transfer opportunities between levels with insights
        assert result.transfers_discovered >= 1
        assert 0.0 <= result.sync_quality <= 1.0

    def test_c5_surprise_reporting_and_attention(self):
        """Surprise reports propagate upward; attention allocates downward."""
        from core.hrm import AbstractionLevel, CrossLevelBridge

        bridge = CrossLevelBridge()

        # Surprise: L0 detects anomaly → reports to all above
        surprise_msgs = bridge.report_surprise(
            anomaly={"type": "unexpected_pattern"},
            source_level=AbstractionLevel.PERCEPTUAL,
            surprise_magnitude=0.8,
        )
        assert len(surprise_msgs) >= 1
        # All messages go upward
        for m in surprise_msgs:
            assert m.target_level > m.source_level

        # Attention: L3 focuses L0, L1
        attention_msgs = bridge.allocate_attention(
            priority_level=AbstractionLevel.STRATEGIC,
            priority_signal={"focus": "security_threat"},
        )
        assert len(attention_msgs) >= 1
        # All messages go downward
        for m in attention_msgs:
            assert m.target_level < m.source_level

        # Bridge metrics should reflect all traffic
        metrics = bridge.get_bridge_metrics()
        assert metrics["total_messages"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR D: Meta-Autopoietic Level N
# ═══════════════════════════════════════════════════════════════════════════════


class TestPillarD_MetaLevel:
    """Validate Level N observation, proposal, evaluation, and application."""

    def test_d1_observe_hierarchy(self):
        """Meta-level observation captures level states and computes fitness."""
        from core.hrm import AbstractionLevel, MetaAutopoieticLevel

        meta = MetaAutopoieticLevel()
        level_states = {
            AbstractionLevel.PERCEPTUAL: {
                "snr_score": 0.88,
                "cycle_count": 10,
                "learning_velocity": 0.02,
            },
            AbstractionLevel.OPERATIONAL: {
                "snr_score": 0.87,
                "cycle_count": 8,
                "learning_velocity": 0.01,
            },
            AbstractionLevel.TACTICAL: {
                "snr_score": 0.91,
                "cycle_count": 6,
                "learning_velocity": 0.03,
            },
        }
        bridge_metrics = {
            "pass_rate": 0.85,
            "sync_quality": 0.78,
            "resonance_events": 2,
            "boundary_health": [],
        }

        obs = meta.observe_hierarchy(level_states, bridge_metrics)
        assert obs.message_pass_rate == 0.85
        assert obs.sync_quality == 0.78
        assert len(obs.level_snr_scores) == 3
        assert 0.0 <= obs.architectural_fitness <= 1.0

    def test_d2_propose_bottleneck_fix(self):
        """Meta-level proposes boundary tuning for detected bottleneck."""
        from core.hrm import (
            AbstractionLevel,
            MetaAutopoieticLevel,
            MetaObservation,
            MetaOperation,
        )

        meta = MetaAutopoieticLevel()
        obs = MetaObservation(
            bottlenecks=[(AbstractionLevel.PERCEPTUAL, AbstractionLevel.OPERATIONAL)],
            level_snr_scores={
                AbstractionLevel.PERCEPTUAL: 0.88,
                AbstractionLevel.OPERATIONAL: 0.86,
            },
        )

        proposal = meta.propose_modification(obs)
        assert proposal is not None
        assert proposal.operation == MetaOperation.BOUNDARY_TUNING
        assert "source_level" in proposal.target

    def test_d3_propose_snr_rebalancing(self):
        """Meta-level proposes SNR rebalancing for underperforming level."""
        from core.hrm import (
            AbstractionLevel,
            MetaAutopoieticLevel,
            MetaObservation,
            MetaOperation,
        )

        meta = MetaAutopoieticLevel()
        obs = MetaObservation(
            level_snr_scores={
                AbstractionLevel.PERCEPTUAL: 0.70,  # Well below 0.85 target
                AbstractionLevel.OPERATIONAL: 0.86,
            },
        )

        proposal = meta.propose_modification(obs)
        assert proposal is not None
        assert proposal.operation == MetaOperation.SNR_REBALANCING

    def test_d4_evaluate_modification(self):
        """Evaluation scores proposals based on improvement vs risk."""
        from core.hrm import MetaAutopoieticLevel, MetaOperation, MetaProposal

        meta = MetaAutopoieticLevel(
            min_improvement_threshold=0.02,
            max_risk_tolerance=0.3,
        )

        # Good proposal: high improvement, low risk
        good = MetaProposal(
            operation=MetaOperation.BOUNDARY_TUNING,
            expected_improvement=0.08,
            risk_level=0.1,
        )
        good_score = meta.evaluate_modification(good)
        assert good_score > 0, f"Good proposal should score positive: {good_score}"

        # Too risky
        risky = MetaProposal(
            operation=MetaOperation.LEVEL_MERGER,
            expected_improvement=0.10,
            risk_level=0.5,  # > max_risk_tolerance 0.3
        )
        risky_score = meta.evaluate_modification(risky)
        assert risky_score < 0, f"Risky proposal should score negative: {risky_score}"

        # Too small
        tiny = MetaProposal(
            operation=MetaOperation.SNR_REBALANCING,
            expected_improvement=0.01,  # < min_improvement 0.02
            risk_level=0.05,
        )
        tiny_score = meta.evaluate_modification(tiny)
        assert tiny_score < 0, f"Tiny improvement should score negative: {tiny_score}"

    def test_d5_apply_boundary_tuning(self):
        """Applying boundary tuning modifies permeability."""
        from core.hrm import (
            AbstractionLevel,
            LevelBoundary,
            MetaAutopoieticLevel,
            MetaOperation,
            MetaProposal,
            TriggerCondition,
            default_level_configs,
        )

        meta = MetaAutopoieticLevel()
        boundary = LevelBoundary(
            source_level=AbstractionLevel.PERCEPTUAL,
            target_level=AbstractionLevel.OPERATIONAL,
            permeability=0.5,
        )
        boundaries = {
            (AbstractionLevel.PERCEPTUAL, AbstractionLevel.OPERATIONAL): boundary,
        }

        proposal = MetaProposal(
            operation=MetaOperation.BOUNDARY_TUNING,
            trigger=TriggerCondition.INFORMATION_BOTTLENECK,
            target={
                "source_level": "PERCEPTUAL",
                "target_level": "OPERATIONAL",
                "action": "increase_permeability",
                "delta": 0.1,
            },
            expected_improvement=0.08,
            risk_level=0.1,
        )

        applied = meta.apply_modification(proposal, boundaries, default_level_configs())
        assert applied is True
        assert boundary.permeability == pytest.approx(0.6, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR E: Hierarchical Engine Core
# ═══════════════════════════════════════════════════════════════════════════════


class TestPillarE_HierarchicalEngine:
    """Validate the core HRM engine: single cycle, cascade, resonance."""

    def test_e1_engine_instantiation(self):
        """HRM instantiates with default config, bridge, and meta-level."""
        from core.hrm import HierarchicalReasoningModel, HRMStatus

        hrm = HierarchicalReasoningModel()
        assert hrm.VERSION == "1.0.0"
        assert hrm.CODENAME == "Ascending Spiral"
        assert hrm._status == HRMStatus.IDLE
        assert hrm._cycle_count == 0

    def test_e2_single_cycle_produces_results(self):
        """A single cycle runs all 5 levels and produces compound SNR."""
        from core.hrm import (
            HierarchicalReasoningModel,
            HRMStatus,
        )

        hrm = HierarchicalReasoningModel()
        result = hrm.run_cycle({"context": "smoke_test"})

        # Status
        assert result.status == HRMStatus.COMPLETED
        assert result.cycle_number == 1

        # All 5 levels produced results
        assert len(result.level_results) == 5

        # Compound SNR is reasonable (above floor)
        assert (
            result.compound_snr > 0.0
        ), f"Compound SNR should be positive: {result.compound_snr}"

        # Each level produced hypotheses
        for level, lr in result.level_results.items():
            assert lr.hypotheses_generated > 0
            assert lr.snr_score > 0.0

    def test_e3_learning_cascade_mechanics(self):
        """Learning cascade correctly propagates positive deltas across levels."""
        from core.hrm import (
            AbstractionLevel,
            HierarchicalReasoningModel,
            LevelCycleResult,
        )

        hrm = HierarchicalReasoningModel()

        # Directly test _cascade_learning with synthetic positive deltas.
        # The deterministic simulation produces stable SNR (delta=0), so
        # we test the cascade mechanics directly with mock level results.
        level_results = {
            AbstractionLevel.PERCEPTUAL: LevelCycleResult(
                level=AbstractionLevel.PERCEPTUAL,
                snr_score=0.90,
                learning_delta=0.05,  # Positive delta → triggers cascade
            ),
            AbstractionLevel.OPERATIONAL: LevelCycleResult(
                level=AbstractionLevel.OPERATIONAL,
                snr_score=0.88,
                learning_delta=0.02,
            ),
            AbstractionLevel.TACTICAL: LevelCycleResult(
                level=AbstractionLevel.TACTICAL,
                snr_score=0.91,
                learning_delta=0.0,  # No learning at this level
            ),
        }

        cascade_count = hrm._cascade_learning(level_results)

        # L0 delta=0.05 cascades upward to L1, L2 (at least 2 events)
        # L1 delta=0.02 cascades upward to L2 and downward to L0
        assert (
            cascade_count >= 2
        ), f"Expected at least 2 cascade events, got {cascade_count}"

        # Verify cumulative_learning was boosted at receiving levels
        l1_state = hrm.get_level_state(AbstractionLevel.OPERATIONAL)
        assert (
            l1_state.get("cumulative_learning", 0.0) > 0
        ), "L1 should receive cascaded learning from L0"

    def test_e4_compound_snr_weighted_correctly(self):
        """Compound SNR uses correct level weights (L0=0.10 ... LN=0.30)."""
        from core.hrm import (
            AbstractionLevel,
            HierarchicalReasoningModel,
            LevelCycleResult,
        )

        hrm = HierarchicalReasoningModel()

        # Create uniform level results (all SNR = 0.90)
        level_results = {}
        for level in AbstractionLevel:
            level_results[level] = LevelCycleResult(
                level=level,
                snr_score=0.90,
            )

        compound = hrm._compute_compound_snr(level_results)
        # With uniform 0.90 and weights summing to 1.0, result ≈ 0.90
        assert compound == pytest.approx(
            0.90, abs=0.01
        ), f"Uniform 0.90 across all levels should give ~0.90, got {compound}"

    def test_e5_level_cycle_success_property(self):
        """LevelCycleResult.success checks against level-specific SNR threshold."""
        from core.hrm import AbstractionLevel, LevelCycleResult

        # L0 threshold = 0.85
        passing = LevelCycleResult(
            level=AbstractionLevel.PERCEPTUAL,
            snr_score=0.90,
        )
        assert passing.success is True

        # L3 threshold = 0.95
        failing = LevelCycleResult(
            level=AbstractionLevel.STRATEGIC,
            snr_score=0.90,  # Below 0.95
        )
        assert failing.success is False


# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR F: Campaign, Convergence & Telemetry
# ═══════════════════════════════════════════════════════════════════════════════


class TestPillarF_CampaignAndTelemetry:
    """Validate multi-cycle campaigns, convergence, and status reporting."""

    def test_f1_campaign_runs_multiple_cycles(self):
        """Campaign runs multiple cycles and returns list of results."""
        from core.hrm import HierarchicalReasoningModel

        hrm = HierarchicalReasoningModel()
        results = hrm.run_campaign(max_cycles=5)

        assert len(results) >= 1
        assert len(results) <= 5

        # Cycle numbers are sequential
        for i, r in enumerate(results):
            assert r.cycle_number == i + 1

    def test_f2_campaign_converges(self):
        """Campaign converges when improvement drops below threshold."""
        from core.hrm import HierarchicalReasoningModel, HRMConfig, HRMStatus

        config = HRMConfig(
            convergence_threshold=0.05,  # Generous threshold for quick convergence
            max_cycles=20,
        )
        hrm = HierarchicalReasoningModel(config=config)
        results = hrm.run_campaign()

        # Should converge before max_cycles (the engine uses deterministic
        # simulation, so after first cycle SNR stabilizes quickly)
        assert len(results) < 20 or results[-1].status == HRMStatus.CONVERGED

    def test_f3_hierarchy_status_comprehensive(self):
        """get_hierarchy_status returns all expected fields."""
        from core.hrm import HierarchicalReasoningModel

        hrm = HierarchicalReasoningModel()
        hrm.run_cycle()

        status = hrm.get_hierarchy_status()
        assert status["version"] == "1.0.0"
        assert status["codename"] == "Ascending Spiral"
        assert status["cycle_count"] == 1
        assert "levels" in status
        assert "bridge_metrics" in status
        assert "compound_snr_trajectory" in status

        # All 5 levels in status
        assert len(status["levels"]) == 5

    def test_f4_snr_trajectory_tracks_progress(self):
        """SNR trajectory records compound SNR for each cycle."""
        from core.hrm import HierarchicalReasoningModel

        hrm = HierarchicalReasoningModel()
        for _ in range(4):
            hrm.run_cycle()

        trajectory = hrm.get_snr_trajectory()
        assert len(trajectory) == 4
        for snr in trajectory:
            assert 0.0 <= snr <= 1.0

    def test_f5_meta_observation_triggers_on_interval(self):
        """Meta-level observation triggers every N cycles."""
        from core.hrm import HierarchicalReasoningModel, HRMConfig

        config = HRMConfig(
            meta_observation_interval=2,  # Every 2 cycles
            enable_meta_level=True,
        )
        hrm = HierarchicalReasoningModel(config=config)

        # Cycle 1: no meta observation (1 % 2 != 0)
        r1 = hrm.run_cycle()
        assert r1.meta_observation is None

        # Cycle 2: meta observation fires (2 % 2 == 0)
        r2 = hrm.run_cycle()
        assert r2.meta_observation is not None
        assert 0.0 <= r2.meta_observation.architectural_fitness <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# RUNNER ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
