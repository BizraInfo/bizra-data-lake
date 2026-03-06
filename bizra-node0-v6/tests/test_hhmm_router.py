"""Tests for BIZRA HHMM Router — Complexity Classification & Action Bus."""

import os
import sys
import time
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hhmm_router import (
    HhmmRouter, ComplexityTier, ClassificationResult, TIER_CONFIGS,
    ActionBus, MissionTicket,
    _extract_length_score, _extract_question_complexity,
    _extract_domain_breadth, _extract_specificity,
)
from reflex_cache import ReflexCache


@pytest.fixture
def router():
    return HhmmRouter()


@pytest.fixture
def router_with_cache():
    cache = ReflexCache(max_entries=10)
    return HhmmRouter(reflex_cache=cache), cache


class TestFeatureExtraction:
    def test_length_score_empty(self):
        assert _extract_length_score("") == 0.0

    def test_length_score_short(self):
        score = _extract_length_score("hello")
        assert 0.0 < score < 0.2

    def test_length_score_long(self):
        score = _extract_length_score(" ".join(["word"] * 100))
        assert score > 0.8

    def test_question_complexity_simple(self):
        assert _extract_question_complexity("What is AI?") < 0.3

    def test_question_complexity_compound(self):
        score = _extract_question_complexity(
            "Create a plan and then analyze the results, also compare "
            "with alternatives, and finally write a report?"
        )
        assert score > 0.5

    def test_domain_breadth_single(self):
        score = _extract_domain_breadth("fix the Python bug")
        assert score <= 0.4

    def test_domain_breadth_multi(self):
        score = _extract_domain_breadth(
            "design the database, write the API code, deploy to production"
        )
        assert score > 0.5

    def test_specificity_vague(self):
        score = _extract_specificity("do something general about anything")
        assert score > 0.5

    def test_specificity_precise(self):
        score = _extract_specificity("fix exactly this function on line 42")
        assert score < 0.5


class TestClassification:
    def test_returns_classification_result(self, router):
        result = router.classify("hello")
        assert isinstance(result, ClassificationResult)

    def test_simple_query_is_simple(self, router):
        result = router.classify("What time is it?")
        assert result.tier in (ComplexityTier.TRIVIAL, ComplexityTier.SIMPLE)

    def test_complex_query_is_complex(self, router):
        result = router.classify(
            "Analyze the architectural differences between microservices "
            "and monoliths, then compare their tradeoffs for distributed "
            "AI platforms, create a decision matrix, and write test code"
        )
        assert result.tier in (ComplexityTier.COMPLEX, ComplexityTier.SOVEREIGN)

    def test_complexity_score_bounded(self, router):
        result = router.classify("test")
        assert 0.0 <= result.complexity_score <= 1.0

    def test_confidence_bounded(self, router):
        result = router.classify("test")
        assert 0.0 <= result.confidence <= 1.0

    def test_features_populated(self, router):
        result = router.classify("analyze this code")
        assert "length" in result.features or "reflex_hit" in result.features

    def test_handler_matches_tier(self, router):
        result = router.classify("hello")
        expected_handler = TIER_CONFIGS[result.tier].handler
        assert result.handler == expected_handler

    def test_latency_budget_positive(self, router):
        result = router.classify("test")
        assert result.latency_budget_ms > 0

    def test_classification_count_increments(self, router):
        router.classify("a")
        router.classify("b")
        assert router.classification_count == 2

    def test_classification_time_tracked(self, router):
        result = router.classify("test")
        assert result.classification_ms >= 0

    def test_as_evidence(self, router):
        result = router.classify("test")
        ev = result.as_evidence()
        assert "tier" in ev
        assert "handler" in ev
        assert "complexity_score" in ev


class TestReflexCacheIntegration:
    def test_cache_hit_returns_trivial(self, router_with_cache):
        router, cache = router_with_cache
        # Precipitate a pattern
        for _ in range(3):
            cache.record_observation("cached query", "response", 0.95,
                                     {"a": 0.95})
        result = router.classify("cached query")
        assert result.tier == ComplexityTier.TRIVIAL
        assert result.has_reflex is True
        assert result.confidence == 0.99

    def test_no_reflex_is_not_trivial(self, router_with_cache):
        router, cache = router_with_cache
        result = router.classify("not in cache")
        assert result.has_reflex is False


class TestTierConfigs:
    def test_all_tiers_have_configs(self):
        for tier in ComplexityTier:
            assert tier in TIER_CONFIGS

    def test_tier_ranges_are_contiguous(self):
        ranges = sorted(
            (c.score_range for c in TIER_CONFIGS.values()),
            key=lambda r: r[0],
        )
        for i in range(len(ranges) - 1):
            assert ranges[i][1] == ranges[i + 1][0]

    def test_tier_ranges_cover_unit(self):
        ranges = [c.score_range for c in TIER_CONFIGS.values()]
        assert min(r[0] for r in ranges) == 0.0
        assert max(r[1] for r in ranges) == 1.0


class TestActionBus:
    @pytest.fixture
    def bus(self):
        return ActionBus()

    def _ticket(self, mission_id="m1", priority=1.0, deadline_s=60):
        return MissionTicket(
            mission_id=mission_id,
            input_text="test",
            classification=ClassificationResult(
                tier=ComplexityTier.SIMPLE,
                config=TIER_CONFIGS[ComplexityTier.SIMPLE],
                complexity_score=0.2,
                confidence=0.8,
                features={},
                classification_ms=0.1,
                has_reflex=False,
            ),
            priority=priority,
            queued_at=time.time(),
            deadline=time.time() + deadline_s,
        )

    def test_submit_and_retrieve(self, bus):
        bus.submit(self._ticket("t1"))
        ticket = bus.next_ticket()
        assert ticket is not None
        assert ticket.mission_id == "t1"

    def test_priority_ordering(self, bus):
        bus.submit(self._ticket("low", priority=0.1))
        bus.submit(self._ticket("high", priority=0.9))
        ticket = bus.next_ticket()
        assert ticket.mission_id == "high"

    def test_expired_tickets_skipped(self, bus):
        bus.submit(self._ticket("expired", deadline_s=-1))
        bus.submit(self._ticket("valid", deadline_s=60))
        ticket = bus.next_ticket()
        assert ticket.mission_id == "valid"

    def test_complete_frees_slot(self, bus):
        bus.submit(self._ticket("done"))
        ticket = bus.next_ticket()
        bus.complete(ticket.mission_id)
        assert bus.active_count == 0

    def test_queue_depth(self, bus):
        bus.submit(self._ticket("a"))
        bus.submit(self._ticket("b"))
        assert bus.queue_depth == 2
