"""Tests for core.sovereign.network_effect — Metcalfe network projection.

Pure computation tests. No I/O, no mocks needed.
Module is classified as ESTIMATOR — no constitutional thresholds.
"""

import math

import pytest

from core.sovereign.network_effect import (
    EST_COST_DECAY_RATE,
    EST_REFLEXES_PER_NODE,
    EST_TFLOPS_PER_NODE,
    NetworkEffectEstimator,
    NetworkProjection,
)


@pytest.fixture
def estimator() -> NetworkEffectEstimator:
    return NetworkEffectEstimator()


class TestProjection:
    """project() computes correct metrics."""

    def test_single_node(self, estimator: NetworkEffectEstimator):
        p = estimator.project(1)
        assert isinstance(p, NetworkProjection)
        assert p.nodes == 1
        assert p.skills_available == EST_REFLEXES_PER_NODE
        assert p.compute_tflops == EST_TFLOPS_PER_NODE
        assert p.latency_factor == 1.0  # log10(1) = 0 → 1/(1+0) = 1
        assert p.intelligence_density == 1.0  # special case for n=1

    def test_ten_nodes(self, estimator: NetworkEffectEstimator):
        p = estimator.project(10)
        assert p.skills_available == 10 * EST_REFLEXES_PER_NODE
        assert p.compute_tflops == round(10 * EST_TFLOPS_PER_NODE, 2)
        assert p.latency_factor == round(1.0 / (1.0 + math.log10(10)), 4)  # 0.5
        assert p.intelligence_density == round(math.log(10), 4)

    def test_zero_nodes_raises(self, estimator: NetworkEffectEstimator):
        with pytest.raises(ValueError, match="n_nodes must be >= 1"):
            estimator.project(0)

    def test_negative_nodes_raises(self, estimator: NetworkEffectEstimator):
        with pytest.raises(ValueError, match="n_nodes must be >= 1"):
            estimator.project(-5)

    def test_latency_decreases_with_nodes(self, estimator: NetworkEffectEstimator):
        lat_1 = estimator.project(1).latency_factor
        lat_10 = estimator.project(10).latency_factor
        lat_1000 = estimator.project(1000).latency_factor
        assert lat_1 > lat_10 > lat_1000

    def test_cost_decreases_with_nodes(self, estimator: NetworkEffectEstimator):
        cost_1 = estimator.project(1).cost_per_node
        cost_100 = estimator.project(100).cost_per_node
        cost_10000 = estimator.project(10000).cost_per_node
        assert cost_1 > cost_100 > cost_10000

    def test_intelligence_density_grows_logarithmically(
        self, estimator: NetworkEffectEstimator
    ):
        d_10 = estimator.project(10).intelligence_density
        d_100 = estimator.project(100).intelligence_density
        d_1000 = estimator.project(1000).intelligence_density
        # log growth: each 10x should add roughly the same amount
        delta_1 = d_100 - d_10
        delta_2 = d_1000 - d_100
        assert delta_1 == pytest.approx(delta_2, abs=0.01)

    def test_projection_has_timestamp(self, estimator: NetworkEffectEstimator):
        p = estimator.project(5)
        assert isinstance(p.timestamp, str)
        assert "T" in p.timestamp  # ISO format

    def test_projection_is_frozen(self, estimator: NetworkEffectEstimator):
        p = estimator.project(10)
        with pytest.raises(AttributeError):
            p.nodes = 999  # type: ignore[misc]

    def test_cost_formula(self, estimator: NetworkEffectEstimator):
        n = 100
        p = estimator.project(n)
        expected = 1.0 / (1.0 + EST_COST_DECAY_RATE * math.log(n))
        assert p.cost_per_node == round(expected, 4)


class TestCustomParams:
    """Constructor allows overriding empirical defaults."""

    def test_custom_reflexes(self):
        est = NetworkEffectEstimator(reflexes_per_node=100)
        p = est.project(5)
        assert p.skills_available == 500

    def test_custom_tflops(self):
        est = NetworkEffectEstimator(tflops_per_node=10.0)
        p = est.project(3)
        assert p.compute_tflops == 30.0


class TestMilestones:
    """project_milestones() returns standard dashboard projections."""

    def test_returns_list(self, estimator: NetworkEffectEstimator):
        milestones = estimator.project_milestones()
        assert isinstance(milestones, list)
        assert len(milestones) == 8

    def test_milestone_nodes(self, estimator: NetworkEffectEstimator):
        milestones = estimator.project_milestones()
        expected = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 8_000_000_000]
        assert [m.nodes for m in milestones] == expected

    def test_milestones_are_monotonic_compute(self, estimator: NetworkEffectEstimator):
        milestones = estimator.project_milestones()
        computes = [m.compute_tflops for m in milestones]
        assert computes == sorted(computes)


class TestMoatMetrics:
    """compute_moat_metrics() quantifies the three-pillar moat."""

    def test_single_node_zero_connections(self, estimator: NetworkEffectEstimator):
        moat = estimator.compute_moat_metrics(1)
        assert moat["data_connections"] == 0

    def test_ten_nodes_connections(self, estimator: NetworkEffectEstimator):
        moat = estimator.compute_moat_metrics(10)
        assert moat["data_connections"] == 10 * 9 // 2  # 45

    def test_moat_score_positive(self, estimator: NetworkEffectEstimator):
        moat = estimator.compute_moat_metrics(100)
        assert moat["moat_score"] > 0

    def test_moat_score_grows_with_nodes(self, estimator: NetworkEffectEstimator):
        m10 = estimator.compute_moat_metrics(10)["moat_score"]
        m100 = estimator.compute_moat_metrics(100)["moat_score"]
        m1000 = estimator.compute_moat_metrics(1000)["moat_score"]
        assert m10 < m100 < m1000

    def test_returns_expected_keys(self, estimator: NetworkEffectEstimator):
        moat = estimator.compute_moat_metrics(50)
        expected_keys = {
            "hardware_contribution_tflops",
            "data_connections",
            "intelligence_skills",
            "moat_score",
        }
        assert set(moat.keys()) == expected_keys
