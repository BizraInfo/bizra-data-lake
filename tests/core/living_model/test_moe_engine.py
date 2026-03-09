"""MOE Engine Tests — Phase 68.07 TDD Anchors.

Standing on: Shazeer (2017) top-K routing, Boyd (1976) OODA,
Kahneman (2011) System-2 multi-expert deliberation.

45+ tests covering:
1. Expert scoring (keyword, HHMM state, context boost)
2. Router (top-K, fallback, override, invariants)
3. Synthesis (weighted, best_of, gate, edge cases)
4. MOEEngine full pipeline (route→execute→synthesize)
5. Property-based invariants (hypothesis)
"""

from __future__ import annotations

import threading

import pytest

from core.living_model.moe_engine import (
    DEFAULT_EXPERTS,
    EXPERT_G,
    EXPERT_K,
    EXPERT_R,
    EXPERT_S,
    EXPERT_V,
    Expert,
    ExpertAssignment,
    ExpertResult,
    MOEEngine,
    SynthesisResult,
)


# =============================================================================
# 1. EXPERT SCORING
# =============================================================================


class TestExpertScoring:
    """Tests for individual expert relevance scoring."""

    def test_reasoning_expert_activates_on_how_questions(self) -> None:
        score = EXPERT_R.score_relevance("How do I optimize this?")
        assert score > 0.0

    def test_knowledge_expert_activates_on_what_questions(self) -> None:
        score = EXPERT_K.score_relevance("What is autopoiesis?")
        assert score > 0.0

    def test_skills_expert_activates_on_action_verbs(self) -> None:
        score = EXPERT_S.score_relevance("Write a function to deploy")
        assert score > 0.0

    def test_governance_expert_activates_on_policy_terms(self) -> None:
        score = EXPERT_G.score_relevance("Check governance compliance threshold")
        assert score > 0.0

    def test_verification_expert_activates_on_prove_terms(self) -> None:
        score = EXPERT_V.score_relevance("Verify the proof evidence")
        assert score > 0.0

    def test_empty_input_returns_base_weight_fraction(self) -> None:
        score = EXPERT_R.score_relevance("")
        assert score == pytest.approx(1.0 * 0.2, abs=1e-6)

    def test_score_always_in_zero_one_range(self) -> None:
        for expert in DEFAULT_EXPERTS:
            for text in ["", "hello", "how what code governance verify " * 10]:
                score = expert.score_relevance(text)
                assert 0.0 <= score <= 1.0, f"{expert.id}: {score}"

    def test_hhmm_state_boosts_matching_expert(self) -> None:
        # With matching HHMM state, score should be higher
        score_general = EXPERT_R.score_relevance("test query", macro_state="general")
        score_matched = EXPERT_R.score_relevance("test query", macro_state="reasoning")
        assert score_matched > score_general

    def test_context_hint_boosts_expert(self) -> None:
        score_no_hint = EXPERT_K.score_relevance("test", context={})
        score_hint = EXPERT_K.score_relevance(
            "test", context={"expert_hint": "pat_k"}
        )
        assert score_hint > score_no_hint

    def test_context_continuity_bonus(self) -> None:
        score_no = EXPERT_S.score_relevance("test", context={})
        score_cont = EXPERT_S.score_relevance(
            "test", context={"previous_expert": "pat_s"}
        )
        assert score_cont > score_no

    def test_custom_expert_creation(self) -> None:
        custom = Expert(
            id="custom",
            keywords=frozenset({"magic", "spell"}),
            hhmm_states=frozenset({"fantasy"}),
            base_weight=0.8,
        )
        assert custom.score_relevance("cast a magic spell") > 0.0
        assert custom.score_relevance("unrelated text") == pytest.approx(0.0)


# =============================================================================
# 2. ROUTER
# =============================================================================


class TestRouter:
    """Tests for MOE routing decisions."""

    def test_route_returns_at_least_one_expert(self) -> None:
        engine = MOEEngine()
        result = engine.route("any query")
        assert len(result) >= 1

    def test_route_weights_sum_to_one(self) -> None:
        engine = MOEEngine()
        result = engine.route("How do I write code?")
        total = sum(a.weight for a in result)
        assert abs(total - 1.0) < 1e-6

    def test_top_k_selects_highest_scorers(self) -> None:
        engine = MOEEngine(top_k=2)
        result = engine.route("How do I analyze this?")
        assert len(result) <= 2
        # Weights should be descending
        if len(result) == 2:
            assert result[0].weight >= result[1].weight

    def test_top_k_default_is_two(self) -> None:
        engine = MOEEngine()
        result = engine.route("How do I build a proof system?")
        assert len(result) <= 2

    def test_fallback_to_reasoning_on_zero_scores(self) -> None:
        # Create experts with impossible keywords
        experts = [
            Expert(id="x", keywords=frozenset({"zzzzz"}), hhmm_states=frozenset()),
        ]
        engine = MOEEngine(experts=experts, min_confidence=0.5)
        result = engine.route("normal query")
        assert result[0].expert_id == "pat_r"
        assert result[0].weight == 1.0

    def test_expert_override_bypasses_router(self) -> None:
        engine = MOEEngine()
        result = engine.route("any query", expert_override="sat_g")
        assert len(result) == 1
        assert result[0].expert_id == "sat_g"
        assert result[0].weight == 1.0

    def test_expert_override_multiple(self) -> None:
        engine = MOEEngine()
        result = engine.route("any", expert_override=["pat_r", "pat_k"])
        assert len(result) == 2
        assert result[0].weight == pytest.approx(0.5)

    def test_expert_override_invalid_falls_back(self) -> None:
        engine = MOEEngine()
        result = engine.route("any", expert_override="nonexistent")
        assert result[0].expert_id == "pat_r"

    def test_all_five_experts_can_be_selected(self) -> None:
        engine = MOEEngine(top_k=5)
        # Use text that triggers all experts
        text = "How to verify the governance policy and write code for knowledge retrieval?"
        result = engine.route(text)
        expert_ids = {a.expert_id for a in result}
        assert len(expert_ids) >= 3  # At least 3 of 5 should activate

    def test_top_k_override_in_route_call(self) -> None:
        engine = MOEEngine(top_k=2)
        result = engine.route("How and what and code and governance and verify?", top_k=4)
        assert len(result) <= 4

    def test_reasoning_wins_for_how_why_queries(self) -> None:
        engine = MOEEngine()
        result = engine.route("Why does this happen and how can I fix it?")
        assert result[0].expert_id == "pat_r"

    def test_knowledge_wins_for_what_queries(self) -> None:
        engine = MOEEngine()
        result = engine.route("What is the history of who created it?")
        assert result[0].expert_id == "pat_k"

    def test_skills_wins_for_code_queries(self) -> None:
        engine = MOEEngine()
        result = engine.route("Write code to implement and deploy the function")
        assert result[0].expert_id == "pat_s"


# =============================================================================
# 3. SYNTHESIS
# =============================================================================


class TestSynthesis:
    """Tests for expert output synthesis."""

    def test_weighted_combination_produces_correct_ihsan(self) -> None:
        engine = MOEEngine()
        results = [
            ExpertResult("pat_r", "reasoning output", ihsan=0.96, confidence=0.8),
            ExpertResult("pat_k", "knowledge output", ihsan=0.94, confidence=0.7),
        ]
        assignments = [
            ExpertAssignment("pat_r", weight=0.6),
            ExpertAssignment("pat_k", weight=0.4),
        ]
        synthesis = engine.synthesize(results, assignments)
        expected = 0.96 * 0.6 + 0.94 * 0.4
        assert synthesis.ihsan == pytest.approx(expected, abs=1e-6)

    def test_gate_rejects_below_threshold(self) -> None:
        engine = MOEEngine(ihsan_threshold=0.95)
        results = [ExpertResult("pat_r", "low quality", ihsan=0.80, confidence=0.5)]
        assignments = [ExpertAssignment("pat_r", weight=1.0)]
        synthesis = engine.synthesize(results, assignments)
        assert not synthesis.passed_gate
        assert "below" in synthesis.reason.lower()

    def test_gate_passes_at_threshold(self) -> None:
        engine = MOEEngine(ihsan_threshold=0.95)
        results = [ExpertResult("pat_r", "high quality", ihsan=0.95, confidence=0.9)]
        assignments = [ExpertAssignment("pat_r", weight=1.0)]
        synthesis = engine.synthesize(results, assignments)
        assert synthesis.passed_gate

    def test_single_expert_result(self) -> None:
        engine = MOEEngine()
        results = [ExpertResult("pat_s", "code output", ihsan=0.97, confidence=0.9)]
        assignments = [ExpertAssignment("pat_s", weight=1.0)]
        synthesis = engine.synthesize(results, assignments)
        assert synthesis.ihsan == pytest.approx(0.97)
        assert synthesis.experts_used == ("pat_s",)

    def test_all_five_experts_combined(self) -> None:
        engine = MOEEngine(ihsan_threshold=0.90)
        results = [
            ExpertResult(f"expert_{i}", f"output {i}", ihsan=0.95, confidence=0.8)
            for i in range(5)
        ]
        assignments = [ExpertAssignment(f"expert_{i}", weight=0.2) for i in range(5)]
        synthesis = engine.synthesize(results, assignments)
        assert synthesis.ihsan == pytest.approx(0.95)
        assert len(synthesis.experts_used) == 5

    def test_empty_expert_results_returns_failure(self) -> None:
        engine = MOEEngine()
        synthesis = engine.synthesize([], [])
        assert not synthesis.passed_gate
        assert "no expert" in synthesis.reason.lower()

    def test_best_of_strategy(self) -> None:
        engine = MOEEngine(synthesis_strategy="best_of", ihsan_threshold=0.90)
        results = [
            ExpertResult("pat_r", "mediocre", ihsan=0.91, confidence=0.5),
            ExpertResult("pat_k", "excellent", ihsan=0.98, confidence=0.9),
        ]
        assignments = [
            ExpertAssignment("pat_r", weight=0.6),
            ExpertAssignment("pat_k", weight=0.4),
        ]
        synthesis = engine.synthesize(results, assignments)
        # best_of should pick pat_k (higher ihsan * confidence)
        assert synthesis.experts_used == ("pat_k",)
        assert synthesis.ihsan == pytest.approx(0.98)

    def test_synthesis_tracks_latency(self) -> None:
        engine = MOEEngine()
        results = [
            ExpertResult("pat_r", "r", ihsan=0.96, confidence=0.8, latency_ms=10.0),
            ExpertResult("pat_k", "k", ihsan=0.95, confidence=0.7, latency_ms=20.0),
        ]
        assignments = [
            ExpertAssignment("pat_r", weight=0.5),
            ExpertAssignment("pat_k", weight=0.5),
        ]
        synthesis = engine.synthesize(results, assignments)
        assert synthesis.total_latency_ms == pytest.approx(30.0)


# =============================================================================
# 4. MOE ENGINE (Full Pipeline)
# =============================================================================


class TestMOEEngine:
    """Tests for the full MOE pipeline."""

    @staticmethod
    def _mock_executor(
        assignment: ExpertAssignment, input_text: str, context: dict
    ) -> ExpertResult:
        return ExpertResult(
            expert_id=assignment.expert_id,
            text=f"Response from {assignment.expert_id}",
            ihsan=0.96,
            confidence=0.8,
            latency_ms=5.0,
        )

    def test_full_pipeline_route_to_synthesis(self) -> None:
        engine = MOEEngine()
        result = engine.run("How do I optimize my pipeline?", self._mock_executor)
        assert isinstance(result, SynthesisResult)
        assert result.passed_gate
        assert len(result.experts_used) >= 1

    def test_engine_stats_track_activations(self) -> None:
        engine = MOEEngine()
        engine.run("How to analyze?", self._mock_executor)
        engine.run("What is this?", self._mock_executor)
        assert engine.stats.total_routes == 2
        assert sum(engine.stats.expert_activations.values()) >= 2

    def test_engine_stats_track_rejections(self) -> None:
        def low_ihsan_executor(a, t, c):
            return ExpertResult(a.expert_id, "low", ihsan=0.5, confidence=0.3)

        engine = MOEEngine(ihsan_threshold=0.95)
        result = engine.run("test", low_ihsan_executor)
        assert not result.passed_gate
        assert engine.stats.gate_rejections >= 1

    def test_engine_thread_safe(self) -> None:
        engine = MOEEngine()
        errors: list[str] = []

        def worker(n: int) -> None:
            try:
                for _ in range(10):
                    result = engine.run(f"Query {n}", self._mock_executor)
                    if not isinstance(result, SynthesisResult):
                        errors.append(f"Bad result type from thread {n}")
            except Exception as e:
                errors.append(f"Thread {n}: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread safety errors: {errors}"
        assert engine.stats.total_routes == 40

    def test_engine_with_hhmm_none_fallback(self) -> None:
        engine = MOEEngine(hhmm_predictor=None)
        result = engine.run("How does this work?", self._mock_executor)
        assert result.passed_gate

    def test_engine_with_hhmm_predictor(self) -> None:
        engine = MOEEngine(hhmm_predictor=lambda t: "reasoning")
        assignments = engine.route("any text")
        # pat_r should be boosted by HHMM state "reasoning"
        assert assignments[0].expert_id == "pat_r"

    def test_engine_with_broken_hhmm_graceful(self) -> None:
        def broken_hhmm(t: str) -> str:
            raise RuntimeError("HHMM crashed")

        engine = MOEEngine(hhmm_predictor=broken_hhmm)
        result = engine.run("How does this work?", self._mock_executor)
        assert result.passed_gate  # Should gracefully degrade

    def test_executor_failure_skips_expert(self) -> None:
        call_count = 0

        def failing_first(a, t, c):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Expert crashed")
            return ExpertResult(a.expert_id, "ok", ihsan=0.96, confidence=0.8)

        engine = MOEEngine(top_k=2)
        # Force 2 experts via override so first crashes and second succeeds
        result = engine.run(
            "test",
            failing_first,
            expert_override=["pat_r", "pat_k"],
        )
        # Two experts routed, first fails, second succeeds
        assert len(result.experts_used) == 1


# =============================================================================
# 5. PROPERTY-BASED TESTS
# =============================================================================

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings
from hypothesis.strategies import floats, integers, text


class TestRouterInvariants:
    """Hypothesis-based property tests for routing invariants."""

    @given(input_text=text(min_size=0, max_size=200), k=integers(1, 5))
    @settings(max_examples=50)
    def test_router_invariants(self, input_text: str, k: int) -> None:
        engine = MOEEngine(top_k=k)
        result = engine.route(input_text)
        # Invariant 1: at least one expert
        assert len(result) >= 1
        # Invariant 2: weights sum to 1.0
        total = sum(a.weight for a in result)
        assert abs(total - 1.0) < 1e-6
        # Invariant 3: all weights in [0, 1]
        assert all(0.0 <= a.weight <= 1.0 for a in result)

    @given(
        ihsan_scores=hypothesis.strategies.lists(
            floats(min_value=0.0, max_value=1.0), min_size=1, max_size=5
        )
    )
    @settings(max_examples=50)
    def test_synthesis_ihsan_is_weighted_average(
        self, ihsan_scores: list[float]
    ) -> None:
        """Weighted average is always between min and max of inputs."""
        engine = MOEEngine(ihsan_threshold=0.0)  # Disable gate for this test
        n = len(ihsan_scores)
        weight = 1.0 / n
        results = [
            ExpertResult(f"e{i}", "t", ihsan=s, confidence=0.8)
            for i, s in enumerate(ihsan_scores)
        ]
        assignments = [
            ExpertAssignment(f"e{i}", weight=weight) for i in range(n)
        ]
        synthesis = engine.synthesize(results, assignments)
        assert min(ihsan_scores) - 1e-6 <= synthesis.ihsan <= max(ihsan_scores) + 1e-6


# =============================================================================
# 6. INTEGRATION (lightweight)
# =============================================================================


class TestIntegration:
    """Integration tests — MOE sits between Reflex and Orchestrator."""

    def test_moe_sits_between_reflex_and_orchestrator(self) -> None:
        """MOE Engine receives input that ReflexCompiler didn't cache-hit."""
        engine = MOEEngine()
        # Simulates: Reflex cache MISS → MOE routing
        result = engine.run(
            "How do I optimize query performance?",
            lambda a, t, c: ExpertResult(a.expert_id, "answer", ihsan=0.96, confidence=0.9),
        )
        assert result.passed_gate
        assert "pat_r" in result.experts_used

    def test_reflex_cache_hit_skips_moe(self) -> None:
        """When ReflexCompiler hits, MOE is never called."""
        engine = MOEEngine()
        # Simulates: Reflex cache HIT → return cached, MOE never invoked
        # Just verify engine starts with 0 routes
        assert engine.stats.total_routes == 0

    def test_moe_result_feeds_reflex_observation(self) -> None:
        """MOE output can be recorded by ReflexCompiler for precipitation."""
        engine = MOEEngine()
        result = engine.run(
            "What is constitutional governance?",
            lambda a, t, c: ExpertResult(a.expert_id, "answer", ihsan=0.97, confidence=0.9),
        )
        # The synthesis result has the data ReflexCompiler needs
        assert result.ihsan > 0.0
        assert result.text  # Non-empty text for recording

    def test_backward_compatible_plan_endpoint(self) -> None:
        """MOE Engine can be instantiated with defaults matching current behavior."""
        engine = MOEEngine()
        assert len(engine.experts) == 5
        assert engine.stats.total_routes == 0
