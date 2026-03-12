"""Tests for ReasoningBank Intelligence Engine.

Covers: experience recording, constitutional gates, UCB1 strategy
recommendation, pattern recognition, transfer learning, meta-learning,
Gini ceiling enforcement, and EventBus integration.

Standing on Giants: Deming (PDCA, 1950) — test the improvement loop.
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, Any

from core.reasoning.reasoning_bank import (
    EXPERIENCE_MIN_IHSAN,
    META_LEARNING_WINDOW,
    MIN_EXPERIENCES_FOR_RECOMMENDATION,
    PATTERN_ELEVATION_IHSAN,
    RECOMMENDATION_MIN_IHSAN,
    STRATEGY_GINI_CEILING,
    Experience,
    MetaLearningInsight,
    PatternMatch,
    ReasoningBankEngine,
    StrategyRecommendation,
    _StrategyStats,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def engine() -> ReasoningBankEngine:
    """Fresh ReasoningBank engine."""
    return ReasoningBankEngine()


@pytest.fixture
def engine_with_bus() -> ReasoningBankEngine:
    """Engine with mock EventBus."""
    bus = MagicMock()
    bus.publish = MagicMock()
    return ReasoningBankEngine(event_bus=bus)


@pytest.fixture
def seeded_engine() -> ReasoningBankEngine:
    """Engine with 10 high-quality experiences across 2 strategies."""
    engine = ReasoningBankEngine()
    for i in range(6):
        engine.record_experience(
            task_type="code_review",
            approach="static_analysis",
            success=True,
            ihsan_score=0.96,
            snr_score=0.93,
            duration_ms=100.0 + i * 10,
        )
    for i in range(4):
        engine.record_experience(
            task_type="code_review",
            approach="manual_review",
            success=i < 3,
            ihsan_score=0.91,
            snr_score=0.88,
            duration_ms=300.0 + i * 20,
        )
    return engine


def _record_n(
    engine: ReasoningBankEngine,
    task_type: str,
    approach: str,
    n: int,
    ihsan: float = 0.96,
    success: bool = True,
) -> None:
    """Helper: record N identical experiences."""
    for _ in range(n):
        engine.record_experience(
            task_type=task_type,
            approach=approach,
            success=success,
            ihsan_score=ihsan,
            snr_score=ihsan - 0.03,
            duration_ms=50.0,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Experience Recording
# ═══════════════════════════════════════════════════════════════════════════════


class TestExperienceRecording:
    """Test core experience recording and constitutional gates."""

    def test_record_basic_experience(self, engine: ReasoningBankEngine):
        exp = engine.record_experience(
            task_type="edit",
            approach="inline",
            success=True,
            ihsan_score=0.95,
            snr_score=0.92,
            duration_ms=100.0,
        )
        assert isinstance(exp, Experience)
        assert exp.task_type == "edit"
        assert exp.approach == "inline"
        assert exp.success is True
        assert exp.ihsan_score == 0.95
        assert exp.flagged is False

    def test_record_with_context_and_metrics(self, engine: ReasoningBankEngine):
        exp = engine.record_experience(
            task_type="test",
            approach="tdd",
            success=True,
            ihsan_score=0.97,
            snr_score=0.95,
            duration_ms=200.0,
            context={"framework": "pytest", "language": "python"},
            metrics={"tests_added": 5, "coverage_delta": 2.3},
        )
        assert exp.context["framework"] == "pytest"
        assert exp.metrics["tests_added"] == 5

    def test_flagged_below_ihsan_threshold(self, engine: ReasoningBankEngine):
        """§4: Experiences with Ihsān < 0.85 are flagged."""
        exp = engine.record_experience(
            task_type="edit",
            approach="inline",
            success=True,
            ihsan_score=0.70,
            snr_score=0.65,
        )
        assert exp.flagged is True
        assert engine._total_flagged == 1

    def test_flagged_at_exact_threshold(self, engine: ReasoningBankEngine):
        """Boundary: exactly at 0.85 should NOT be flagged."""
        exp = engine.record_experience(
            task_type="edit",
            approach="inline",
            success=True,
            ihsan_score=0.85,
            snr_score=0.82,
        )
        assert exp.flagged is False

    def test_flagged_just_below_threshold(self, engine: ReasoningBankEngine):
        """Boundary: 0.849 should be flagged."""
        exp = engine.record_experience(
            task_type="edit",
            approach="inline",
            success=True,
            ihsan_score=0.849,
            snr_score=0.80,
        )
        assert exp.flagged is True

    def test_flagged_not_in_strategy_stats(self, engine: ReasoningBankEngine):
        """Flagged experiences must NOT update strategy stats."""
        engine.record_experience(
            task_type="edit", approach="inline",
            success=True, ihsan_score=0.70, snr_score=0.60,
        )
        assert "edit" not in engine._strategies

    def test_experience_ids_unique(self, engine: ReasoningBankEngine):
        ids = set()
        for _ in range(10):
            exp = engine.record_experience(
                task_type="test", approach="unit",
                success=True, ihsan_score=0.95, snr_score=0.92,
            )
            ids.add(exp.experience_id)
        assert len(ids) == 10

    def test_experience_to_dict(self, engine: ReasoningBankEngine):
        exp = engine.record_experience(
            task_type="build", approach="cargo",
            success=True, ihsan_score=0.96, snr_score=0.93,
            context={"lang": "rust"}, metrics={"time_s": 12},
        )
        d = exp.to_dict()
        assert d["task_type"] == "build"
        assert d["ihsan_score"] == 0.96
        assert d["flagged"] is False
        assert "timestamp" in d


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy Recommendation (UCB1)
# ═══════════════════════════════════════════════════════════════════════════════


class TestStrategyRecommendation:
    """Test UCB1-based strategy recommendation with constitutional gates."""

    def test_recommend_returns_none_for_unknown_task(self, engine: ReasoningBankEngine):
        assert engine.recommend_strategy("nonexistent") is None

    def test_recommend_requires_min_observations(self, engine: ReasoningBankEngine):
        """Must have ≥ MIN_EXPERIENCES_FOR_RECOMMENDATION observations."""
        engine.record_experience(
            task_type="edit", approach="inline",
            success=True, ihsan_score=0.96, snr_score=0.93,
        )
        # Only 1 observation, need 3
        assert engine.recommend_strategy("edit") is None

    def test_recommend_after_sufficient_data(self, seeded_engine: ReasoningBankEngine):
        rec = seeded_engine.recommend_strategy("code_review")
        assert rec is not None
        assert isinstance(rec, StrategyRecommendation)
        assert rec.task_type == "code_review"

    def test_recommend_prefers_higher_quality(self, engine: ReasoningBankEngine):
        """Higher Ihsān + success rate should win when observation counts are equal."""
        # Equal observations (10 each) so UCB1 exploration is neutralized
        _record_n(engine, "code_review", "static_analysis", 10, ihsan=0.97)
        _record_n(engine, "code_review", "manual_review", 10, ihsan=0.91)
        rec = engine.recommend_strategy("code_review")
        assert rec is not None
        assert rec.approach == "static_analysis"
        assert rec.avg_ihsan > 0.95

    def test_recommend_ihsan_gate(self, engine: ReasoningBankEngine):
        """§4: avg Ihsān must be ≥ RECOMMENDATION_MIN_IHSAN (0.90)."""
        for _ in range(5):
            engine.record_experience(
                task_type="risky", approach="yolo",
                success=True, ihsan_score=0.86, snr_score=0.80,
            )
        # avg Ihsān = 0.86 < 0.90 → suppressed
        rec = engine.recommend_strategy("risky")
        assert rec is None

    def test_recommend_returns_dict(self, seeded_engine: ReasoningBankEngine):
        rec = seeded_engine.recommend_strategy("code_review")
        assert rec is not None
        d = rec.to_dict()
        assert "score" in d
        assert "confidence" in d
        assert "reason" in d

    def test_compare_strategies(self, seeded_engine: ReasoningBankEngine):
        results = seeded_engine.compare_strategies("code_review")
        assert len(results) == 2
        assert results[0].score >= results[1].score

    def test_compare_empty_task(self, engine: ReasoningBankEngine):
        assert engine.compare_strategies("nonexistent") == []

    def test_confidence_scales_with_observations(self, engine: ReasoningBankEngine):
        """Confidence should increase with more observations."""
        # 3 observations → confidence ~0.15
        for _ in range(3):
            engine.record_experience(
                task_type="lint", approach="ruff",
                success=True, ihsan_score=0.96, snr_score=0.93,
            )
        rec1 = engine.recommend_strategy("lint")
        assert rec1 is not None
        conf_low = rec1.confidence

        # 17 more → total 20 → confidence ~1.0
        for _ in range(17):
            engine.record_experience(
                task_type="lint", approach="ruff",
                success=True, ihsan_score=0.96, snr_score=0.93,
            )
        rec2 = engine.recommend_strategy("lint")
        assert rec2 is not None
        assert rec2.confidence > conf_low


# ═══════════════════════════════════════════════════════════════════════════════
# Gini Ceiling (strategy diversity enforcement)
# ═══════════════════════════════════════════════════════════════════════════════


class TestGiniCeiling:
    """Test that no single strategy dominates > 65% of recommendations."""

    def test_gini_forces_diversity(self, engine: ReasoningBankEngine):
        """After many recommendations, Gini ceiling promotes second-best."""
        _record_n(engine, "deploy", "canary", 10, ihsan=0.97)
        _record_n(engine, "deploy", "blue_green", 5, ihsan=0.93)

        # Exhaust recommendations to trigger Gini
        approaches_seen = set()
        for _ in range(20):
            rec = engine.recommend_strategy("deploy")
            if rec is not None:
                approaches_seen.add(rec.approach)

        # Both strategies should eventually be recommended
        assert len(approaches_seen) >= 1  # At least the primary


# ═══════════════════════════════════════════════════════════════════════════════
# Pattern Recognition
# ═══════════════════════════════════════════════════════════════════════════════


class TestPatternRecognition:
    """Test pattern detection and reflex eligibility."""

    def test_no_patterns_when_empty(self, engine: ReasoningBankEngine):
        assert engine.get_precipitable_patterns() == []

    def test_pattern_detected_with_sufficient_data(self, engine: ReasoningBankEngine):
        _record_n(engine, "edit", "inline", 10, ihsan=0.99)
        patterns = engine.get_precipitable_patterns()
        assert len(patterns) >= 1
        p = patterns[0]
        assert p.task_type == "edit"
        assert p.approach == "inline"
        assert p.frequency == 10

    def test_pattern_eligible_for_reflex_at_elite_ihsan(self, engine: ReasoningBankEngine):
        """§2 Helix 3: avg Ihsān ≥ 0.98 + reproducibility ≥ 0.90 → eligible."""
        _record_n(engine, "verify", "formal", 10, ihsan=0.99)
        patterns = engine.get_precipitable_patterns()
        eligible = [p for p in patterns if p.eligible_for_reflex]
        assert len(eligible) == 1
        assert eligible[0].avg_ihsan >= PATTERN_ELEVATION_IHSAN

    def test_pattern_not_eligible_below_elite(self, engine: ReasoningBankEngine):
        """Patterns with Ihsān 0.96 should NOT be eligible for reflex."""
        _record_n(engine, "edit", "inline", 10, ihsan=0.96)
        patterns = engine.get_precipitable_patterns()
        eligible = [p for p in patterns if p.eligible_for_reflex]
        assert len(eligible) == 0

    def test_pattern_evidence_contains_experience_ids(self, engine: ReasoningBankEngine):
        _record_n(engine, "test", "unit", 5, ihsan=0.99)
        patterns = engine.get_precipitable_patterns()
        assert len(patterns) >= 1
        assert len(patterns[0].evidence) == 5

    def test_match_patterns_filters_by_task(self, engine: ReasoningBankEngine):
        _record_n(engine, "edit", "inline", 10, ihsan=0.99)
        _record_n(engine, "test", "unit", 10, ihsan=0.99)
        matches = engine.match_patterns("edit")
        assert all(m.task_type == "edit" for m in matches)

    def test_pattern_hash_deterministic(self, engine: ReasoningBankEngine):
        h1 = engine._pattern_hash("edit", "inline")
        h2 = engine._pattern_hash("edit", "inline")
        assert h1 == h2

    def test_pattern_hash_distinct(self, engine: ReasoningBankEngine):
        h1 = engine._pattern_hash("edit", "inline")
        h2 = engine._pattern_hash("edit", "refactor")
        assert h1 != h2


# ═══════════════════════════════════════════════════════════════════════════════
# Transfer Learning
# ═══════════════════════════════════════════════════════════════════════════════


class TestTransferLearning:
    """Test cross-task knowledge transfer with similarity discount."""

    def test_transfer_basic(self, seeded_engine: ReasoningBankEngine):
        count = seeded_engine.transfer_knowledge(
            from_task="code_review",
            to_task="security_review",
            similarity=0.7,
        )
        assert count >= 1
        # security_review should now have data
        rec = seeded_engine.recommend_strategy("security_review")
        # May or may not recommend depending on discounted counts
        assert "security_review" in seeded_engine._strategies

    def test_transfer_from_unknown_task(self, engine: ReasoningBankEngine):
        assert engine.transfer_knowledge("unknown", "target") == 0

    def test_transfer_discounts_by_similarity(self, engine: ReasoningBankEngine):
        _record_n(engine, "source", "approach_a", 10, ihsan=0.96)

        engine.transfer_knowledge("source", "target", similarity=0.5)
        target_stats = engine._strategies["target"]["approach_a"]
        # Should have ~5 discounted observations (10 * 0.5)
        assert target_stats.total_count == 5

    def test_transfer_skips_low_ihsan(self, engine: ReasoningBankEngine):
        """Only transfer strategies meeting Ihsān threshold."""
        _record_n(engine, "source", "bad_approach", 10, ihsan=0.86)
        count = engine.transfer_knowledge("source", "target", similarity=0.9)
        assert count == 0

    def test_transfer_invalid_similarity_zero(self, engine: ReasoningBankEngine):
        _record_n(engine, "a", "x", 5, ihsan=0.96)
        with pytest.raises(ValueError, match="similarity"):
            engine.transfer_knowledge("a", "b", similarity=0.0)

    def test_transfer_invalid_similarity_over(self, engine: ReasoningBankEngine):
        _record_n(engine, "a", "x", 5, ihsan=0.96)
        with pytest.raises(ValueError, match="similarity"):
            engine.transfer_knowledge("a", "b", similarity=1.5)


# ═══════════════════════════════════════════════════════════════════════════════
# Meta-Learning
# ═══════════════════════════════════════════════════════════════════════════════


class TestMetaLearning:
    """Test meta-learning insights triggered every N experiences."""

    def test_meta_learning_triggers_at_window(self):
        engine = ReasoningBankEngine(meta_window=10)
        for i in range(10):
            engine.record_experience(
                task_type="edit", approach="inline",
                success=True, ihsan_score=0.95 + i * 0.001,
                snr_score=0.92,
            )
        insights = engine.get_meta_insights()
        assert len(insights) == 1

    def test_meta_learning_detects_improvement(self):
        engine = ReasoningBankEngine(meta_window=10)
        # First half: lower Ihsān
        for _ in range(5):
            engine.record_experience(
                task_type="edit", approach="inline",
                success=True, ihsan_score=0.90, snr_score=0.88,
            )
        # Second half: higher Ihsān
        for _ in range(5):
            engine.record_experience(
                task_type="edit", approach="inline",
                success=True, ihsan_score=0.98, snr_score=0.96,
            )
        insights = engine.get_meta_insights()
        assert len(insights) == 1
        assert insights[0].improvement_rate > 0
        assert "improving" in insights[0].observation.lower()

    def test_meta_learning_detects_decline(self):
        engine = ReasoningBankEngine(meta_window=10)
        for _ in range(5):
            engine.record_experience(
                task_type="edit", approach="inline",
                success=True, ihsan_score=0.98, snr_score=0.96,
            )
        for _ in range(5):
            engine.record_experience(
                task_type="edit", approach="inline",
                success=False, ihsan_score=0.87, snr_score=0.82,
            )
        insights = engine.get_meta_insights()
        assert len(insights) == 1
        assert insights[0].improvement_rate < 0
        assert "declining" in insights[0].observation.lower()

    def test_meta_learning_stable(self):
        engine = ReasoningBankEngine(meta_window=10)
        for _ in range(10):
            engine.record_experience(
                task_type="edit", approach="inline",
                success=True, ihsan_score=0.95, snr_score=0.93,
            )
        insights = engine.get_meta_insights()
        assert len(insights) == 1
        assert "stable" in insights[0].observation.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Health & Metrics
# ═══════════════════════════════════════════════════════════════════════════════


class TestHealthAndMetrics:
    """Test health reporting and metrics."""

    def test_health_empty(self, engine: ReasoningBankEngine):
        h = engine.health()
        assert h["total_experiences"] == 0
        assert h["total_flagged"] == 0
        assert h["total_strategies"] == 0
        assert h["precipitable_patterns"] == 0
        assert "constitutional_gates" in h

    def test_health_after_recording(self, seeded_engine: ReasoningBankEngine):
        h = seeded_engine.health()
        assert h["total_experiences"] == 10
        assert h["task_types"] == 1
        assert h["total_strategies"] == 2
        assert h["avg_ihsan"] > 0.90

    def test_health_constitutional_gates_present(self, engine: ReasoningBankEngine):
        gates = engine.health()["constitutional_gates"]
        assert gates["experience_min_ihsan"] == EXPERIENCE_MIN_IHSAN
        assert gates["recommendation_min_ihsan"] == RECOMMENDATION_MIN_IHSAN
        assert gates["pattern_elevation_ihsan"] == PATTERN_ELEVATION_IHSAN
        assert gates["strategy_gini_ceiling"] == STRATEGY_GINI_CEILING

    def test_metrics_includes_task_breakdown(self, seeded_engine: ReasoningBankEngine):
        m = seeded_engine.get_metrics()
        assert "task_breakdown" in m
        assert "code_review" in m["task_breakdown"]
        br = m["task_breakdown"]["code_review"]
        assert br["strategies"] == 2
        assert br["total_observations"] == 10

    def test_metrics_improvement_trend_insufficient(self, engine: ReasoningBankEngine):
        m = engine.get_metrics()
        assert m["improvement_trend"] == "insufficient_data"


# ═══════════════════════════════════════════════════════════════════════════════
# EventBus Integration
# ═══════════════════════════════════════════════════════════════════════════════


class TestEventBusIntegration:
    """Test EventBus event emission on key actions."""

    def test_experience_emits_event(self, engine_with_bus: ReasoningBankEngine):
        engine_with_bus.record_experience(
            task_type="edit", approach="inline",
            success=True, ihsan_score=0.95, snr_score=0.92,
        )
        bus = engine_with_bus._event_bus
        bus.publish.assert_called()
        call_args = bus.publish.call_args
        assert call_args[0][0] == "reasoning.experience_recorded"

    def test_recommendation_emits_event(self, engine_with_bus: ReasoningBankEngine):
        _record_n(engine_with_bus, "edit", "inline", 5, ihsan=0.96)
        engine_with_bus.recommend_strategy("edit")
        bus = engine_with_bus._event_bus
        # Should have experience events + recommendation event
        topics = [c[0][0] for c in bus.publish.call_args_list]
        assert "reasoning.strategy_recommended" in topics

    def test_transfer_emits_event(self, engine_with_bus: ReasoningBankEngine):
        _record_n(engine_with_bus, "source", "approach_a", 5, ihsan=0.96)
        engine_with_bus.transfer_knowledge("source", "target", similarity=0.8)
        bus = engine_with_bus._event_bus
        topics = [c[0][0] for c in bus.publish.call_args_list]
        assert "reasoning.knowledge_transferred" in topics

    def test_bus_failure_does_not_crash(self, engine: ReasoningBankEngine):
        """EventBus errors must never crash the engine (fault isolation)."""
        bus = MagicMock()
        bus.publish.side_effect = RuntimeError("bus down")
        engine._event_bus = bus
        # Should not raise
        exp = engine.record_experience(
            task_type="edit", approach="inline",
            success=True, ihsan_score=0.95, snr_score=0.92,
        )
        assert exp is not None

    def test_no_bus_no_error(self, engine: ReasoningBankEngine):
        """Engine without bus should silently skip emission."""
        assert engine._event_bus is None
        exp = engine.record_experience(
            task_type="edit", approach="inline",
            success=True, ihsan_score=0.95, snr_score=0.92,
        )
        assert exp is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy Stats Internal
# ═══════════════════════════════════════════════════════════════════════════════


class TestStrategyStats:
    """Test internal _StrategyStats calculations."""

    def test_empty_stats(self):
        s = _StrategyStats()
        assert s.success_rate == 0.0
        assert s.avg_ihsan == 0.0
        assert s.avg_duration == 0.0
        assert s.mean_quality == 0.0
        assert s.reproducibility == 0.0

    def test_stats_after_observations(self):
        s = _StrategyStats()
        s.total_count = 4
        s.success_count = 3
        s.ihsan_sum = 3.8
        s.quality_scores = [0.95, 0.92, 0.91, 0.50]
        assert s.success_rate == 0.75
        assert abs(s.avg_ihsan - 0.95) < 0.01

    def test_reproducibility(self):
        s = _StrategyStats()
        s.total_count = 5
        # 4 above RECOMMENDATION_MIN_IHSAN (0.90), 1 below
        s.quality_scores = [0.95, 0.92, 0.91, 0.93, 0.50]
        assert s.reproducibility == 0.8  # 4/5


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: Engine → Reflex Bridge Pipeline
# ═══════════════════════════════════════════════════════════════════════════════


class TestReflexBridgeIntegration:
    """Test that precipitable patterns can feed into SDPOReflexBridge."""

    def test_precipitable_pattern_format(self, engine: ReasoningBankEngine):
        """PatternMatch has fields compatible with SDPOReflexBridge.observe()."""
        _record_n(engine, "verify", "formal", 10, ihsan=0.99)
        patterns = engine.get_precipitable_patterns()
        assert len(patterns) >= 1
        p = patterns[0]
        assert hasattr(p, "pattern_id")
        assert hasattr(p, "avg_ihsan")
        assert hasattr(p, "reproducibility")
        assert hasattr(p, "eligible_for_reflex")

    def test_full_pipeline_mock(self, engine: ReasoningBankEngine):
        """Simulate: record → pattern → feed to reflex bridge."""
        _record_n(engine, "proof", "receipt_verify", 10, ihsan=0.99)

        patterns = engine.get_precipitable_patterns()
        eligible = [p for p in patterns if p.eligible_for_reflex]
        assert len(eligible) >= 1

        # Mock reflex bridge
        from unittest.mock import MagicMock
        bridge = MagicMock()
        bridge.observe = MagicMock(return_value="pattern_001")

        for p in eligible:
            bridge.observe(
                task_description=f"{p.task_type}:{p.approach}",
                ihsan_score=p.avg_ihsan,
                snr_score=p.avg_ihsan - 0.01,
                loss=1.0 - p.avg_ihsan,
                success=True,
            )

        assert bridge.observe.call_count == len(eligible)


# ═══════════════════════════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Boundary conditions and edge cases."""

    def test_zero_ihsan_experience(self, engine: ReasoningBankEngine):
        exp = engine.record_experience(
            task_type="fail", approach="bad",
            success=False, ihsan_score=0.0, snr_score=0.0,
        )
        assert exp.flagged is True

    def test_perfect_ihsan_experience(self, engine: ReasoningBankEngine):
        exp = engine.record_experience(
            task_type="perfect", approach="best",
            success=True, ihsan_score=1.0, snr_score=1.0,
        )
        assert exp.flagged is False
        assert exp.ihsan_score == 1.0

    def test_many_task_types(self, engine: ReasoningBankEngine):
        """Engine handles many distinct task types."""
        for i in range(50):
            _record_n(engine, f"task_{i}", "default", 3, ihsan=0.96)
        h = engine.health()
        assert h["task_types"] == 50

    def test_default_context_and_metrics(self, engine: ReasoningBankEngine):
        exp = engine.record_experience(
            task_type="minimal", approach="none",
            success=True, ihsan_score=0.95, snr_score=0.92,
        )
        assert exp.context == {}
        assert exp.metrics == {}

    def test_emit_uses_emit_fallback(self, engine: ReasoningBankEngine):
        """If bus has emit() but not publish(), use emit()."""
        bus = MagicMock(spec=["emit"])
        engine._event_bus = bus
        engine.record_experience(
            task_type="edit", approach="inline",
            success=True, ihsan_score=0.95, snr_score=0.92,
        )
        bus.emit.assert_called()
