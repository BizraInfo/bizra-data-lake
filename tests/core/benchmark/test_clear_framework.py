"""
Tests for CLEAR Framework — Multi-Dimensional Agent Evaluation
================================================================

Standing on the Shoulders of Giants:
- HAL (2025): Holistic Agent Leaderboard multi-VM evaluation
- ABC (2025): Agentic Benchmark Checklist anti-overestimation
- Shannon (1948): Information-theoretic efficiency bounds

إحسان — Excellence in all things.
"""

import time
import pytest

from core.benchmark.clear_framework import (
    CLEARFramework,
    CLEARDimension,
    CLEARMetrics,
    MetricWeight,
    EvaluationContext,
    AgenticBenchmarkChecklist,
    CostMetrics,
    LatencyMetrics,
    EfficacyMetrics,
    AssuranceMetrics,
    ReliabilityMetrics,
)


# ═══════════════════════════════════════════════════════════════════════════════
# MetricWeight
# ═══════════════════════════════════════════════════════════════════════════════


class TestMetricWeight:

    def test_default_sums_to_one(self):
        w = MetricWeight()
        total = w.cost + w.latency + w.efficacy + w.assurance + w.reliability
        assert abs(total - 1.0) < 0.001

    def test_custom_valid(self):
        w = MetricWeight(cost=0.2, latency=0.2, efficacy=0.2, assurance=0.2, reliability=0.2)
        assert abs(sum(w.as_dict().values()) - 1.0) < 0.001

    def test_invalid_sum_raises(self):
        with pytest.raises(ValueError):
            MetricWeight(cost=0.5, latency=0.5, efficacy=0.5, assurance=0.5, reliability=0.5)

    def test_as_dict(self):
        d = MetricWeight().as_dict()
        assert set(d.keys()) == {"cost", "latency", "efficacy", "assurance", "reliability"}


# ═══════════════════════════════════════════════════════════════════════════════
# Individual Metric Dataclasses
# ═══════════════════════════════════════════════════════════════════════════════


class TestMetricClasses:

    def test_cost_score(self):
        m = CostMetrics(total_tokens=50_000, cost_usd=0.5)
        s = m.score(budget_tokens=100_000, budget_usd=1.0)
        assert 0.0 <= s <= 1.0

    def test_cost_score_zero(self):
        m = CostMetrics()
        assert m.score() == pytest.approx(1.0)

    def test_latency_score(self):
        m = LatencyMetrics(time_to_first_token_ms=250, total_completion_ms=5000)
        s = m.score()
        assert 0.0 <= s <= 1.0

    def test_efficacy_score(self):
        m = EfficacyMetrics(accuracy=0.95, task_completion_rate=0.90, goal_achievement=0.85, partial_credit=0.80)
        s = m.score()
        assert 0.0 <= s <= 1.0

    def test_assurance_perfect(self):
        m = AssuranceMetrics(safety_violations=0, hallucination_rate=0.0, reproducibility=1.0, graceful_failures=5)
        s = m.score()
        assert s > 0.8

    def test_assurance_violations_drop_score(self):
        clean = AssuranceMetrics(safety_violations=0, reproducibility=1.0)
        dirty = AssuranceMetrics(safety_violations=3, reproducibility=1.0)
        assert clean.score() > dirty.score()

    def test_reliability_score(self):
        m = ReliabilityMetrics(consistency_across_runs=0.9, recovery_rate=0.8, runs_completed=5, runs_failed=1)
        s = m.score()
        assert 0.0 <= s <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# CLEARMetrics
# ═══════════════════════════════════════════════════════════════════════════════


class TestCLEARMetrics:

    def test_default_score(self):
        m = CLEARMetrics()
        # All zeroed sub-metrics → score should be computable (no crash)
        s = m.compute_overall_score()
        assert isinstance(s, float)

    def test_dimension_scores(self):
        m = CLEARMetrics()
        ds = m.dimension_scores()
        assert set(ds.keys()) == {"cost", "latency", "efficacy", "assurance", "reliability"}

    def test_to_dict(self):
        m = CLEARMetrics(task_id="t1", agent_id="a1")
        d = m.to_dict()
        assert d["task_id"] == "t1"
        assert "overall_score" in d
        assert "dimension_scores" in d

    def test_weighted_score_varies(self):
        m = CLEARMetrics()
        m.efficacy.accuracy = 1.0
        m.efficacy.task_completion_rate = 1.0
        w1 = MetricWeight(cost=0.05, latency=0.05, efficacy=0.80, assurance=0.05, reliability=0.05)
        w2 = MetricWeight(cost=0.80, latency=0.05, efficacy=0.05, assurance=0.05, reliability=0.05)
        s1 = m.compute_overall_score(w1)
        s2 = m.compute_overall_score(w2)
        assert s1 != s2


# ═══════════════════════════════════════════════════════════════════════════════
# AgenticBenchmarkChecklist (ABC)
# ═══════════════════════════════════════════════════════════════════════════════


class TestABC:

    def test_all_passed(self):
        abc = AgenticBenchmarkChecklist()
        config = {check[0]: True for check in abc.CHECKS}
        passed, score, failed = abc.validate(config)
        assert passed is True
        assert score == 1.0
        assert failed == []

    def test_some_failed(self):
        abc = AgenticBenchmarkChecklist()
        config = {check[0]: True for check in abc.CHECKS[:5]}
        passed, score, failed = abc.validate(config)
        assert passed is False
        assert 0.0 < score < 1.0
        assert len(failed) == 5

    def test_generate_report(self):
        abc = AgenticBenchmarkChecklist()
        abc.validate({check[0]: True for check in abc.CHECKS})
        report = abc.generate_report()
        assert "ABC" in report
        assert "PASSED" in report


# ═══════════════════════════════════════════════════════════════════════════════
# CLEARFramework
# ═══════════════════════════════════════════════════════════════════════════════


class TestCLEARFramework:

    def test_init_defaults(self):
        fw = CLEARFramework()
        assert fw.enable_abc is True
        assert fw.abc_checker is not None

    def test_init_custom_weights(self):
        w = MetricWeight(cost=0.1, latency=0.1, efficacy=0.5, assurance=0.2, reliability=0.1)
        fw = CLEARFramework(weights=w)
        assert fw.weights.efficacy == 0.5

    def test_evaluate_context_manager(self):
        fw = CLEARFramework()
        with fw.evaluate("task-1", "agent-a") as ctx:
            ctx.record_efficacy(accuracy=0.90, task_completion=0.85)
            ctx.record_cost(input_tokens=1000, output_tokens=500, cost_usd=0.02)
        metrics = fw.get_metrics("task-1")
        assert metrics is not None
        assert metrics.efficacy.accuracy == 0.90
        assert metrics.cost.total_tokens == 1500

    def test_evaluate_auto_latency(self):
        fw = CLEARFramework()
        with fw.evaluate("task-lat", "agent-b") as ctx:
            time.sleep(0.01)
            ctx.mark_first_token()
        metrics = fw.get_metrics("task-lat")
        assert metrics.latency.total_completion_ms > 0
        assert metrics.latency.time_to_first_token_ms > 0

    def test_get_metrics_missing(self):
        fw = CLEARFramework()
        assert fw.get_metrics("nonexistent") is None

    def test_compute_aggregate_empty(self):
        fw = CLEARFramework()
        agg = fw.compute_aggregate()
        assert agg["count"] == 0

    def test_compute_aggregate(self):
        fw = CLEARFramework()
        with fw.evaluate("t1", "a1") as ctx:
            ctx.record_efficacy(accuracy=0.9)
        with fw.evaluate("t2", "a1") as ctx:
            ctx.record_efficacy(accuracy=0.8)
        agg = fw.compute_aggregate()
        assert agg["count"] == 2
        assert 0 < agg["aggregate_score"] <= 1.0

    def test_validate_benchmark(self):
        fw = CLEARFramework()
        config = {check[0]: True for check in AgenticBenchmarkChecklist.CHECKS}
        passed, report = fw.validate_benchmark(config)
        assert passed is True
        assert "PASSED" in report

    def test_validate_benchmark_disabled(self):
        fw = CLEARFramework(enable_abc=False)
        passed, report = fw.validate_benchmark({})
        assert passed is True

    def test_compare_agents(self):
        fw = CLEARFramework()
        with fw.evaluate("t1", "alpha") as ctx:
            ctx.record_efficacy(accuracy=0.95)
        with fw.evaluate("t2", "beta") as ctx:
            ctx.record_efficacy(accuracy=0.80)
        comp = fw.compare_agents(["alpha", "beta"])
        assert "alpha" in comp
        assert "beta" in comp
        assert comp["alpha"]["aggregate_score"] >= comp["beta"]["aggregate_score"]

    def test_compare_unknown_agent(self):
        fw = CLEARFramework()
        comp = fw.compare_agents(["ghost"])
        assert comp["ghost"]["count"] == 0

    def test_identify_weakest_dimension(self):
        fw = CLEARFramework()
        with fw.evaluate("t1", "weak") as ctx:
            ctx.record_efficacy(accuracy=0.95, task_completion=0.90)
            ctx.record_cost(input_tokens=90_000, output_tokens=10_000, cost_usd=0.90)
        dim, score = fw.identify_weakest_dimension("weak")
        assert isinstance(dim, CLEARDimension)
        assert 0.0 <= score <= 1.0

    def test_identify_weakest_unknown(self):
        fw = CLEARFramework()
        dim, score = fw.identify_weakest_dimension("nobody")
        assert dim == CLEARDimension.EFFICACY
        assert score == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# EvaluationContext recording
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvaluationContext:

    def test_record_assurance(self):
        fw = CLEARFramework()
        with fw.evaluate("a-task", "a-agent") as ctx:
            ctx.record_assurance(safety_violations=0, hallucination_rate=0.01, reproducibility=0.99)
        m = fw.get_metrics("a-task")
        assert m.assurance.safety_violations == 0
        assert m.assurance.hallucination_rate == 0.01

    def test_record_reliability(self):
        fw = CLEARFramework()
        with fw.evaluate("r-task", "r-agent") as ctx:
            ctx.record_reliability(consistency=0.95, runs_completed=10, runs_failed=1)
        m = fw.get_metrics("r-task")
        assert m.reliability.consistency_across_runs == 0.95
        assert m.reliability.runs_completed == 10

    def test_run_hash_generated(self):
        fw = CLEARFramework()
        with fw.evaluate("hash-task", "hash-agent") as ctx:
            ctx.record_efficacy(accuracy=0.5)
        m = fw.get_metrics("hash-task")
        assert m.run_hash != ""
        assert len(m.run_hash) == 16
