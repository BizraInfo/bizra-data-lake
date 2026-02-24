"""
Entropy Router Tests
=====================

Validates that the System 1/2 query router correctly classifies
queries by complexity and produces appropriate routing decisions.

Created: 2026-02-18 | BIZRA Node0 | Reasoning Module
"""

import pytest

from core.integration.constants import sat_frontier_quorum
from core.reasoning.entropy_router import (
    EntropyRouter,
    QueryComplexity,
    RoutingDecision,
)


@pytest.fixture()
def router() -> EntropyRouter:
    return EntropyRouter()


# ============================================================================
# Tier classification tests
# ============================================================================


class TestTierRouting:
    """Validate that queries route to the correct System 1/2 tier."""

    def test_trivial_query_routes_to_s1(self, router: EntropyRouter) -> None:
        decision = router.route("What is 2+2?")
        assert decision.query_complexity == QueryComplexity.TRIVIAL
        assert decision.system == "S1_REFLEXIVE"
        assert decision.quorum_size == 0
        assert decision.use_got is False
        assert decision.use_orchestrator is False

    def test_simple_query_routes_to_s1(self, router: EntropyRouter) -> None:
        decision = router.route(
            "What is the capital of France? "
            "What language do they primarily speak there? "
            "Is it a member of the European Union?"
        )
        assert decision.query_complexity == QueryComplexity.SIMPLE
        assert decision.system == "S1_REFLEXIVE"
        assert decision.quorum_size == 0
        assert decision.use_got is False

    def test_moderate_query_routes_to_s1_5(
        self, router: EntropyRouter
    ) -> None:
        query = (
            "Compare and contrast REST vs GraphQL APIs, "
            "considering the trade-offs for mobile applications "
            "and the pros and cons of each approach."
        )
        decision = router.route(query)
        assert decision.query_complexity == QueryComplexity.MODERATE
        assert decision.system == "S1_5_MODERATE"
        assert decision.use_got is True
        assert decision.use_orchestrator is False

    def test_complex_query_routes_to_s2(
        self, router: EntropyRouter
    ) -> None:
        query = (
            "Analyze the implications of quantum computing on modern "
            "cryptographic systems. How does Shor's algorithm relate to "
            "RSA security? Evaluate the trade-offs between lattice-based "
            "and code-based post-quantum cryptography from both a "
            "performance perspective and a security perspective. "
            "Additionally, what are the implications for blockchain "
            "consensus mechanisms? Furthermore, synthesize a step-by-step "
            "migration plan considering backward compatibility."
        )
        decision = router.route(query)
        assert decision.query_complexity in (
            QueryComplexity.COMPLEX,
            QueryComplexity.FRONTIER,
        )
        assert decision.system == "S2_DELIBERATIVE"
        assert decision.use_got is True
        assert decision.use_orchestrator is True

    def test_frontier_query_full_quorum(
        self, router: EntropyRouter
    ) -> None:
        # Build a clearly frontier-level query with many complexity signals.
        # Must trigger enough sub-question patterns, multi-domain markers,
        # question marks, length, and entropy to exceed the 0.85 threshold.
        query = (
            "Analyze and evaluate the intersection of quantum field theory "
            "and algebraic topology from a mathematical perspective. "
            "Compare and contrast the implications of string theory "
            "and loop quantum gravity. How does the holographic principle "
            "relate to information theory? What are the implications for "
            "condensed matter physics? Additionally, synthesize these "
            "considerations with recent advances in topological quantum "
            "computing from a computational perspective. Furthermore, "
            "what are the trade-offs between different approaches? "
            "Moreover, considering thermodynamics and statistical mechanics, "
            "how do these paradigms interact? Consider the pros and cons "
            "of each, and provide a step-by-step framework for evaluating "
            "competing theories. What are the implications for practical "
            "quantum error correction and fault-tolerant computation?"
        )
        # Frontier-level queries typically arrive with upstream context
        # signaling high complexity (e.g., from a research agent).
        # Default federation size in constants is 10 nodes -> quorum 33.
        decision = router.route(
            query,
            context={"complexity_hint": 0.25, "federation_node_count": 10},
        )
        assert decision.query_complexity == QueryComplexity.FRONTIER
        assert decision.quorum_size == sat_frontier_quorum(10)
        assert decision.snr_requirement == 0.98

    def test_frontier_quorum_scales_with_federation_size(
        self, router: EntropyRouter
    ) -> None:
        query = (
            "Analyze and evaluate the intersection of quantum field theory "
            "and algebraic topology from a mathematical perspective. "
            "Compare and contrast the implications of string theory "
            "and loop quantum gravity. How does the holographic principle "
            "relate to information theory? What are the implications for "
            "condensed matter physics? Additionally, synthesize these "
            "considerations with recent advances in topological quantum "
            "computing from a computational perspective."
        )
        decision = router.route(
            query,
            context={"complexity_hint": 1.0, "federation_node_count": 100},
        )
        assert decision.query_complexity == QueryComplexity.FRONTIER
        assert decision.quorum_size == sat_frontier_quorum(100)
        assert decision.quorum_size == 333

    def test_empty_query_is_trivial(self, router: EntropyRouter) -> None:
        decision = router.route("")
        assert decision.query_complexity == QueryComplexity.TRIVIAL
        assert decision.system == "S1_REFLEXIVE"


# ============================================================================
# Complexity hint from context
# ============================================================================


class TestContextHints:
    """Validate that external context influences routing."""

    def test_complexity_hint_from_context(
        self, router: EntropyRouter
    ) -> None:
        base_score = router.estimate_complexity("Hello world")
        boosted_score = router.estimate_complexity(
            "Hello world", context={"complexity_hint": 1.0}
        )
        assert boosted_score > base_score

    def test_zero_hint_has_no_effect(self, router: EntropyRouter) -> None:
        base_score = router.estimate_complexity("Hello world")
        same_score = router.estimate_complexity(
            "Hello world", context={"complexity_hint": 0.0}
        )
        assert abs(same_score - base_score) < 1e-9


# ============================================================================
# Entropy calculation tests
# ============================================================================


class TestTextEntropy:
    """Validate the Shannon entropy calculation."""

    def test_text_entropy_normalized_0_to_1(
        self, router: EntropyRouter
    ) -> None:
        for text in [
            "hello",
            "a" * 100,
            "abcdefghijklmnopqrstuvwxyz",
            "the quick brown fox jumps over the lazy dog",
        ]:
            entropy = router._text_entropy(text)
            assert 0.0 <= entropy <= 1.0, (
                f"Entropy {entropy} out of range for '{text[:30]}'"
            )

    def test_text_entropy_uniform_is_maximal(
        self, router: EntropyRouter
    ) -> None:
        # All unique characters -> maximum entropy (1.0)
        text = "abcdefghijklmnopqrstuvwxyz"
        entropy = router._text_entropy(text)
        assert entropy > 0.99, f"Expected ~1.0 for uniform distribution, got {entropy}"

    def test_text_entropy_single_char_is_zero(
        self, router: EntropyRouter
    ) -> None:
        entropy = router._text_entropy("aaaa")
        assert entropy == 0.0

    def test_text_entropy_empty_is_zero(
        self, router: EntropyRouter
    ) -> None:
        entropy = router._text_entropy("")
        assert entropy == 0.0


# ============================================================================
# RoutingDecision immutability
# ============================================================================


class TestRoutingDecision:
    """Validate the frozen dataclass contract."""

    def test_routing_decision_is_frozen(self, router: EntropyRouter) -> None:
        decision = router.route("test query")
        assert isinstance(decision, RoutingDecision)
        with pytest.raises(AttributeError):
            decision.system = "HACKED"  # type: ignore[misc]

    def test_routing_decision_has_reasoning(
        self, router: EntropyRouter
    ) -> None:
        decision = router.route("explain gravity")
        assert "score=" in decision.reasoning
        assert decision.query_complexity.name in decision.reasoning


# ============================================================================
# SNR requirement monotonicity
# ============================================================================


class TestSNRMonotonicity:
    """SNR requirements must not decrease as complexity increases."""

    def test_snr_requirements_increase_with_complexity(
        self, router: EntropyRouter
    ) -> None:
        tiers = [
            QueryComplexity.TRIVIAL,
            QueryComplexity.SIMPLE,
            QueryComplexity.MODERATE,
            QueryComplexity.COMPLEX,
            QueryComplexity.FRONTIER,
        ]
        decisions = [
            router._build_decision(tier, 0.5, "test") for tier in tiers
        ]
        snr_values = [d.snr_requirement for d in decisions]

        for i in range(len(snr_values) - 1):
            assert snr_values[i] <= snr_values[i + 1], (
                f"SNR decreased from {tiers[i].name} ({snr_values[i]}) "
                f"to {tiers[i + 1].name} ({snr_values[i + 1]})"
            )
