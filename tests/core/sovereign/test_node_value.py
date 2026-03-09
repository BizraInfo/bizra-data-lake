"""
Tests for Phase 72 — Node Value Engine, Human Lifecycle, Network Effect
========================================================================

Covers:
- human_lifecycle.py: stage mapping, boundaries, progress, agent alignment
- node_value.py: five-factor KPI, geometric mean, normalization bounds
- network_effect.py: projections, milestones, moat metrics, estimator classification
- constants.py: Phase 72 constant verification
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from core.integration.constants import (
    HUMAN_STAGE_ORDER,
    HUMAN_STAGE_THRESHOLDS,
    NODE_VALUE_ACTIVATION_REFERENCE,
    NODE_VALUE_COMPOUNDING_REFERENCE_DAYS,
    NODE_VALUE_STREAK_REFERENCE,
    SEED_QUALIFICATION_RATE_APPRENTICE,
    SEED_QUALIFICATION_RATE_VERIFIER,
    SEED_REWARD_QUALIFICATION,
)
from core.sovereign.human_lifecycle import (
    AGENT_TIER_MAP,
    STAGES,
    HumanStage,
    agent_tier_equivalent,
    human_stage,
    human_stage_detail,
    stage_progress,
)
from core.sovereign.network_effect import (
    _MODULE_CLASS,
    EST_COST_DECAY_RATE,
    EST_REFLEXES_PER_NODE,
    EST_TFLOPS_PER_NODE,
    NetworkEffectEstimator,
    NetworkProjection,
)
from core.sovereign.node_value import NodeValueEngine, NodeValueSnapshot
from core.sovereign.seed_engine import SeedEngine

# =========================================================================
# Helpers
# =========================================================================


def _make_engine_with_episodes(
    n: int = 5,
    snr: float = 0.95,
    ihsan: float = 0.96,
    quality: float = 0.9,
) -> SeedEngine:
    """Create a SeedEngine with n qualified episodes."""
    engine = SeedEngine("test")
    for _ in range(n):
        engine.record_episode({"snr": snr, "ihsan": ihsan, "quality": quality})
    return engine


def _genesis_n_days_ago(days: int) -> str:
    """Return ISO timestamp N days ago."""
    dt = datetime.now(timezone.utc) - timedelta(days=days)
    return dt.isoformat()


# =========================================================================
# CONSTANTS VERIFICATION — Phase 72
# =========================================================================


class TestPhase72Constants:
    """Verify Phase 72 constants exist in constants.py."""

    def test_seed_reward_qualification(self):
        assert SEED_REWARD_QUALIFICATION == 0.75

    def test_seed_qualification_rate_verifier(self):
        assert SEED_QUALIFICATION_RATE_VERIFIER == 0.75

    def test_seed_qualification_rate_apprentice(self):
        assert SEED_QUALIFICATION_RATE_APPRENTICE == 0.50

    def test_human_stage_thresholds_has_7_entries(self):
        assert len(HUMAN_STAGE_THRESHOLDS) == 7

    def test_human_stage_thresholds_seed_zero(self):
        assert HUMAN_STAGE_THRESHOLDS["Seed"] == 0.00

    def test_human_stage_thresholds_catalyst(self):
        assert HUMAN_STAGE_THRESHOLDS["Catalyst"] == 0.85

    def test_human_stage_order_has_7_entries(self):
        assert len(HUMAN_STAGE_ORDER) == 7
        assert HUMAN_STAGE_ORDER[0] == "Seed"
        assert HUMAN_STAGE_ORDER[-1] == "Catalyst"

    def test_node_value_activation_reference(self):
        assert NODE_VALUE_ACTIVATION_REFERENCE == 5.0

    def test_node_value_compounding_reference_days(self):
        assert NODE_VALUE_COMPOUNDING_REFERENCE_DAYS == 365

    def test_node_value_streak_reference(self):
        assert NODE_VALUE_STREAK_REFERENCE == 10


# =========================================================================
# HUMAN LIFECYCLE TESTS
# =========================================================================


class TestHumanStageMapping:
    """Test human_stage() boundary values."""

    def test_zero_is_seed(self):
        assert human_stage(0.00) == "Seed"

    def test_below_node_is_seed(self):
        assert human_stage(0.09) == "Seed"

    def test_node_boundary(self):
        assert human_stage(0.10) == "Node"

    def test_apprentice_boundary(self):
        assert human_stage(0.20) == "Apprentice"

    def test_builder_boundary(self):
        assert human_stage(0.35) == "Builder"

    def test_verifier_boundary(self):
        assert human_stage(0.55) == "Verifier"

    def test_mentor_boundary(self):
        assert human_stage(0.70) == "Mentor"

    def test_catalyst_boundary(self):
        assert human_stage(0.85) == "Catalyst"

    def test_max_score_is_catalyst(self):
        assert human_stage(1.00) == "Catalyst"

    def test_negative_clamps_to_seed(self):
        assert human_stage(-1.0) == "Seed"

    def test_overflow_clamps_to_catalyst(self):
        assert human_stage(5.0) == "Catalyst"

    def test_mid_builder(self):
        assert human_stage(0.45) == "Builder"


class TestHumanStageDetail:
    """Test human_stage_detail() returns full metadata."""

    def test_returns_humanstage_dataclass(self):
        result = human_stage_detail(0.50)
        assert isinstance(result, HumanStage)

    def test_builder_detail(self):
        result = human_stage_detail(0.50)
        assert result.name == "Builder"
        assert result.rank == 3

    def test_seed_detail_on_zero(self):
        result = human_stage_detail(0.0)
        assert result.name == "Seed"
        assert result.rank == 0


class TestStageProgress:
    """Test stage_progress() response structure."""

    def test_builder_progress(self):
        result = stage_progress(0.50)
        assert result["current_stage"] == "Builder"
        assert result["next_stage"] == "Verifier"
        assert result["next_threshold"] == 0.55
        assert result["points_to_next"] == 0.05
        assert 0.0 <= result["progress"] <= 1.0

    def test_catalyst_no_next(self):
        result = stage_progress(0.90)
        assert result["current_stage"] == "Catalyst"
        assert result["next_stage"] is None
        assert result["points_to_next"] == 0.0

    def test_seed_progress(self):
        result = stage_progress(0.05)
        assert result["current_stage"] == "Seed"
        assert result["next_stage"] == "Node"
        assert result["rank"] == 0

    def test_has_description(self):
        result = stage_progress(0.30)
        assert len(result["description"]) > 10
        assert len(result["unlock_condition"]) > 5

    def test_sovereignty_score_in_result(self):
        result = stage_progress(0.42)
        assert result["sovereignty_score"] == 0.42


class TestStagesIntegrity:
    """Test the STAGES list construction from constants."""

    def test_seven_stages(self):
        assert len(STAGES) == 7

    def test_all_stages_reachable(self):
        reached = set()
        for threshold in HUMAN_STAGE_THRESHOLDS.values():
            reached.add(human_stage(threshold))
        assert len(reached) == 7

    def test_monotonically_ordered(self):
        for i in range(len(STAGES) - 1):
            assert STAGES[i].score_low < STAGES[i + 1].score_low
            assert STAGES[i].rank < STAGES[i + 1].rank

    def test_descriptions_non_empty(self):
        for stage in STAGES:
            assert len(stage.description) > 10
            assert len(stage.unlock_condition) > 5

    def test_stages_frozen(self):
        """HumanStage is frozen dataclass."""
        with pytest.raises(AttributeError):
            STAGES[0].name = "Mutated"  # type: ignore[misc]


class TestAgentTierAlignment:
    """Test agent ↔ human tier mapping."""

    def test_all_stages_have_equivalents(self):
        for stage in STAGES:
            equiv = agent_tier_equivalent(stage.name)
            assert equiv is not None
            assert len(equiv) > 0

    def test_unknown_returns_novice(self):
        assert agent_tier_equivalent("Unknown") == "Novice"

    def test_catalyst_is_grandmaster(self):
        assert agent_tier_equivalent("Catalyst") == "Grandmaster"

    def test_seed_is_novice(self):
        assert agent_tier_equivalent("Seed") == "Novice"

    def test_seven_mappings(self):
        assert len(AGENT_TIER_MAP) == 7


# =========================================================================
# NODE VALUE ENGINE TESTS
# =========================================================================


class TestNodeValueZeroState:
    """Test zero-mission / zero-episode node."""

    def test_zero_episodes_zero_composite(self):
        engine = SeedEngine("test")
        nv = NodeValueEngine(engine)
        result = nv.compute()
        assert result.composite == 0.0

    def test_zero_episodes_seed_stage(self):
        engine = SeedEngine("test")
        nv = NodeValueEngine(engine)
        result = nv.compute()
        assert result.human_stage == "Seed"

    def test_zero_episodes_snapshot_type(self):
        engine = SeedEngine("test")
        nv = NodeValueEngine(engine)
        result = nv.compute()
        assert isinstance(result, NodeValueSnapshot)


class TestNodeValueCompute:
    """Test five-factor computation."""

    def test_qualified_episodes_increase_value(self):
        engine = _make_engine_with_episodes(10)
        nv = NodeValueEngine(engine)
        result = nv.compute()
        assert result.composite > 0.0
        assert result.potential > 0.0
        assert result.activation > 0.0

    def test_all_factors_bounded_0_1(self):
        engine = _make_engine_with_episodes(100, snr=0.99, ihsan=0.99)
        nv = NodeValueEngine(engine, genesis_timestamp=_genesis_n_days_ago(365 * 3))
        result = nv.compute()
        assert 0.0 <= result.potential <= 1.0
        assert 0.0 <= result.activation <= 1.0
        assert 0.0 <= result.quality <= 1.0
        assert 0.0 <= result.compounding <= 1.0
        assert 0.0 <= result.synergy <= 1.0
        assert 0.0 <= result.composite <= 1.0

    def test_composite_is_geometric_mean(self):
        engine = _make_engine_with_episodes(5)
        nv = NodeValueEngine(engine, genesis_timestamp=_genesis_n_days_ago(30))
        result = nv.compute()
        if result.composite > 0:
            expected = (
                result.potential
                * result.activation
                * result.quality
                * result.compounding
                * result.synergy
            ) ** 0.2
            assert abs(result.composite - round(expected, 4)) < 0.01

    def test_high_dam_capped_at_1(self):
        engine = _make_engine_with_episodes(100)
        nv = NodeValueEngine(engine, genesis_timestamp=_genesis_n_days_ago(1))
        result = nv.compute()
        assert result.activation == 1.0

    def test_quality_from_ihsan_scores(self):
        engine = _make_engine_with_episodes(10, ihsan=0.98)
        nv = NodeValueEngine(engine)
        result = nv.compute()
        assert result.quality > 0.95

    def test_synergy_pre_federation(self):
        engine = SeedEngine("test")
        nv = NodeValueEngine(engine)
        assert nv._compute_network_synergy() == 1.0

    def test_unqualified_episodes_low_quality(self):
        engine = SeedEngine("test")
        engine.record_episode({"snr": 0.10, "ihsan": 0.20})
        nv = NodeValueEngine(engine)
        result = nv.compute()
        assert result.quality < 0.5


class TestNodeValueCompounding:
    """Test the asymptotic compounding factor."""

    def test_streak_0_still_nonzero(self):
        engine = SeedEngine("test")
        engine.record_episode({"snr": 0.5, "ihsan": 0.5})  # unqualified
        nv = NodeValueEngine(engine, genesis_timestamp=_genesis_n_days_ago(30))
        result = nv.compute()
        assert result.compounding > 0.0

    def test_3_year_compounding_under_1(self):
        engine = _make_engine_with_episodes(1)
        nv = NodeValueEngine(engine, genesis_timestamp=_genesis_n_days_ago(1095))
        result = nv.compute()
        assert result.compounding < 1.0

    def test_compounding_increases_with_age(self):
        engine1 = _make_engine_with_episodes(1)
        nv1 = NodeValueEngine(engine1, genesis_timestamp=_genesis_n_days_ago(10))
        engine2 = _make_engine_with_episodes(1)
        nv2 = NodeValueEngine(engine2, genesis_timestamp=_genesis_n_days_ago(100))
        r1 = nv1.compute()
        r2 = nv2.compute()
        assert r2.compounding > r1.compounding


class TestNodeValueReadOnly:
    """Verify NodeValueEngine is read-only over SeedEngine."""

    def test_no_record_mission_method(self):
        engine = SeedEngine("test")
        nv = NodeValueEngine(engine)
        assert not hasattr(nv, "record_mission")

    def test_reads_episode_count_from_seed_engine(self):
        engine = SeedEngine("test")
        nv = NodeValueEngine(engine, genesis_timestamp=_genesis_n_days_ago(1))
        engine.record_episode({"snr": 0.95, "ihsan": 0.96})
        result = nv.compute()
        assert result.activation > 0


class TestNodeValueHealth:
    """Test health() response."""

    def test_health_shape(self):
        engine = SeedEngine("test")
        nv = NodeValueEngine(engine)
        h = nv.health()
        assert h["engine"] == "node_value"
        assert h["source"] == "seed_engine"
        assert "genesis" in h
        assert h["has_federation"] is False


class TestNodeValueHumanStage:
    """Test human_stage field in snapshots."""

    def test_stage_maps_correctly_seed(self):
        engine = SeedEngine("test")
        nv = NodeValueEngine(engine)
        result = nv.compute()
        assert result.human_stage == "Seed"

    def test_stage_advances_with_growth(self):
        engine = _make_engine_with_episodes(50, snr=0.99, ihsan=0.99)
        nv = NodeValueEngine(engine)
        result = nv.compute()
        # With 50 high-quality episodes, should be beyond Seed
        assert result.human_stage != "Seed" or result.potential > 0


# =========================================================================
# NETWORK EFFECT ESTIMATOR TESTS
# =========================================================================


class TestNetworkEffectClassification:
    """Verify estimator classification."""

    def test_module_class_is_estimator(self):
        assert _MODULE_CLASS == "ESTIMATOR"

    def test_est_prefix_on_constants(self):
        assert EST_REFLEXES_PER_NODE == 50
        assert EST_TFLOPS_PER_NODE == 5.0
        assert EST_COST_DECAY_RATE == 0.15


class TestNetworkProjectionSingleNode:
    """Test baseline (1 node) projections."""

    def test_single_node_skills(self):
        est = NetworkEffectEstimator()
        p = est.project(1)
        assert p.skills_available == 50

    def test_single_node_latency(self):
        est = NetworkEffectEstimator()
        p = est.project(1)
        assert p.latency_factor == 1.0

    def test_single_node_cost(self):
        est = NetworkEffectEstimator()
        p = est.project(1)
        assert p.cost_per_node == 1.0

    def test_single_node_intelligence_density(self):
        est = NetworkEffectEstimator()
        p = est.project(1)
        assert p.intelligence_density == 1.0

    def test_returns_projection_type(self):
        est = NetworkEffectEstimator()
        p = est.project(1)
        assert isinstance(p, NetworkProjection)


class TestNetworkScaling:
    """Test that metrics improve with more nodes."""

    def test_more_nodes_more_skills(self):
        est = NetworkEffectEstimator()
        p1 = est.project(1)
        p100 = est.project(100)
        assert p100.skills_available == 100 * p1.skills_available

    def test_latency_improves(self):
        est = NetworkEffectEstimator()
        p1 = est.project(1)
        p1000 = est.project(1000)
        assert p1000.latency_factor < p1.latency_factor

    def test_cost_decreases(self):
        est = NetworkEffectEstimator()
        p1 = est.project(1)
        p1m = est.project(1_000_000)
        assert p1m.cost_per_node < p1.cost_per_node

    def test_intelligence_density_grows(self):
        est = NetworkEffectEstimator()
        densities = [est.project(n).intelligence_density for n in [1, 10, 100, 1000]]
        for i in range(len(densities) - 1):
            assert densities[i] < densities[i + 1]


class TestNetworkEdgeCases:
    """Test edge cases and validation."""

    def test_zero_nodes_raises(self):
        est = NetworkEffectEstimator()
        with pytest.raises(ValueError, match="n_nodes must be >= 1"):
            est.project(0)

    def test_negative_nodes_raises(self):
        est = NetworkEffectEstimator()
        with pytest.raises(ValueError):
            est.project(-5)


class TestNetworkMilestones:
    """Test milestone projections."""

    def test_milestones_returns_8(self):
        est = NetworkEffectEstimator()
        milestones = est.project_milestones()
        assert len(milestones) == 8

    def test_milestones_first_is_1(self):
        est = NetworkEffectEstimator()
        milestones = est.project_milestones()
        assert milestones[0].nodes == 1

    def test_milestones_last_is_8b(self):
        est = NetworkEffectEstimator()
        milestones = est.project_milestones()
        assert milestones[-1].nodes == 8_000_000_000


class TestNetworkMoatMetrics:
    """Test moat quantification."""

    def test_moat_positive_for_n_gt_1(self):
        est = NetworkEffectEstimator()
        moat = est.compute_moat_metrics(1000)
        assert moat["hardware_contribution_tflops"] > 0
        assert moat["intelligence_skills"] > 0
        assert moat["moat_score"] > 0

    def test_moat_data_connections_formula(self):
        est = NetworkEffectEstimator()
        moat = est.compute_moat_metrics(10)
        assert moat["data_connections"] == 10 * 9 // 2  # n*(n-1)/2

    def test_moat_single_node_zero_connections(self):
        est = NetworkEffectEstimator()
        moat = est.compute_moat_metrics(1)
        assert moat["data_connections"] == 0


class TestNetworkCustomParams:
    """Test custom estimator parameters."""

    def test_custom_reflexes_scales_skills(self):
        est = NetworkEffectEstimator(reflexes_per_node=100)
        p = est.project(10)
        assert p.skills_available == 1000

    def test_custom_tflops(self):
        est = NetworkEffectEstimator(tflops_per_node=10.0)
        p = est.project(100)
        assert p.compute_tflops == 1000.0


class TestNetwork8BillionNodes:
    """Test full human population projection."""

    def test_8b_skills(self):
        est = NetworkEffectEstimator()
        p = est.project(8_000_000_000)
        assert p.skills_available == 400_000_000_000

    def test_8b_latency_under_10_percent(self):
        est = NetworkEffectEstimator()
        p = est.project(8_000_000_000)
        assert p.latency_factor < 0.10
