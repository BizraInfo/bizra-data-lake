"""Tests for core.command.sovereign_command -- Sovereign Command Center.

Covers:
- ComplexityTier and QueryIntent enums
- QueryAnalysis, ProvenanceRecord, CommandResult data classes
- QueryAnalyzer: intent detection, complexity classification
- SNRCalculator: signal/noise decomposition
- SimulatedBackend: health and generation
- InferenceGateway: backend selection and stats
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.command.sovereign_command import (
    CommandResult,
    ComplexityTier,
    InferenceGateway,
    LLMBackend,
    ProvenanceRecord,
    QueryAnalysis,
    QueryAnalyzer,
    QueryIntent,
    SimulatedBackend,
    SNRCalculator,
)


# ---------------------------------------------------------------------------
# ENUM TESTS
# ---------------------------------------------------------------------------


class TestEnums:

    def test_complexity_tiers(self):
        expected = {"TRIVIAL", "SIMPLE", "MODERATE", "COMPLEX", "FRONTIER"}
        actual = {t.name for t in ComplexityTier}
        assert actual == expected

    def test_complexity_tier_ordering(self):
        """Verify tiers increase in value."""
        assert ComplexityTier.TRIVIAL.value < ComplexityTier.SIMPLE.value
        assert ComplexityTier.SIMPLE.value < ComplexityTier.MODERATE.value
        assert ComplexityTier.MODERATE.value < ComplexityTier.COMPLEX.value
        assert ComplexityTier.COMPLEX.value < ComplexityTier.FRONTIER.value

    def test_query_intents(self):
        expected = {
            "FACTUAL", "ANALYTICAL", "CREATIVE", "TECHNICAL",
            "CRITICAL", "SYNTHESIS", "UNKNOWN",
        }
        actual = {i.name for i in QueryIntent}
        assert actual == expected


# ---------------------------------------------------------------------------
# DATA CLASS TESTS
# ---------------------------------------------------------------------------


class TestQueryAnalysis:

    def test_instantiation(self):
        qa = QueryAnalysis(
            query="test query",
            intent=QueryIntent.FACTUAL,
            complexity=ComplexityTier.TRIVIAL,
            estimated_tokens=10,
            domains=["general"],
            requires_reasoning=False,
            requires_tools=False,
            confidence=0.8,
        )
        assert qa.query == "test query"
        assert qa.intent == QueryIntent.FACTUAL
        assert qa.complexity == ComplexityTier.TRIVIAL
        assert qa.estimated_tokens == 10


class TestProvenanceRecord:

    def test_to_dict(self):
        prov = ProvenanceRecord(
            proof_id="abc123",
            query_hash="q_hash",
            response_hash="r_hash",
            snr_score=0.92,
            ihsan_compliant=True,
            backend_used="Simulated",
            complexity="MODERATE",
            intent="ANALYTICAL",
            timestamp="2026-02-12T00:00:00+00:00",
            giants_cited=["Shannon", "Boyd"],
        )
        d = prov.to_dict()
        assert d["proof_id"] == "abc123"
        assert d["snr_score"] == 0.92
        assert d["ihsan_compliant"] is True
        assert len(d["giants_cited"]) == 2


class TestCommandResult:

    def test_instantiation(self):
        prov = ProvenanceRecord(
            proof_id="test",
            query_hash="qh",
            response_hash="rh",
            snr_score=0.9,
            ihsan_compliant=True,
            backend_used="Simulated",
            complexity="SIMPLE",
            intent="FACTUAL",
            timestamp="2026-01-01T00:00:00Z",
            giants_cited=[],
        )
        result = CommandResult(
            query="test query",
            response="test response",
            snr_score=0.9,
            ihsan_compliant=True,
            backend_used="Simulated",
            complexity="SIMPLE",
            intent="FACTUAL",
            latency_ms=50.0,
            provenance=prov,
            metrics={"snr": 0.9},
        )
        assert result.query == "test query"
        assert result.latency_ms == 50.0
        assert result.provenance.proof_id == "test"


# ---------------------------------------------------------------------------
# QueryAnalyzer TESTS
# ---------------------------------------------------------------------------


class TestQueryAnalyzer:

    @pytest.fixture
    def analyzer(self):
        return QueryAnalyzer()

    @pytest.mark.parametrize("query,expected_intent", [
        ("what is entropy", QueryIntent.FACTUAL),
        ("analyze the system architecture", QueryIntent.ANALYTICAL),
        ("create a new module", QueryIntent.CREATIVE),
        ("implement a sorting algorithm", QueryIntent.TECHNICAL),
        ("evaluate the performance", QueryIntent.CRITICAL),
        ("synthesize the findings", QueryIntent.SYNTHESIS),
        ("some random query", QueryIntent.UNKNOWN),
    ])
    def test_intent_detection(self, analyzer, query, expected_intent):
        analysis = analyzer.analyze(query)
        assert analysis.intent == expected_intent

    @pytest.mark.parametrize("word_count,expected_complexity", [
        (5, ComplexityTier.TRIVIAL),      # < 10 words
        (20, ComplexityTier.SIMPLE),      # 10-29 words
        (50, ComplexityTier.MODERATE),    # 30-99 words
        (150, ComplexityTier.COMPLEX),    # 100+ words
    ])
    def test_complexity_by_length(self, analyzer, word_count, expected_complexity):
        query = " ".join(["word"] * word_count)
        analysis = analyzer.analyze(query)
        assert analysis.complexity == expected_complexity

    def test_technical_query_minimum_moderate(self, analyzer):
        """Technical queries should be at least MODERATE complexity."""
        analysis = analyzer.analyze("implement a function")  # short, but technical
        assert analysis.complexity.value >= ComplexityTier.MODERATE.value

    def test_synthesis_query_minimum_moderate(self, analyzer):
        """Synthesis queries should be at least MODERATE complexity."""
        analysis = analyzer.analyze("synthesize the data")  # short, but synthesis
        assert analysis.complexity.value >= ComplexityTier.MODERATE.value

    def test_estimated_tokens(self, analyzer):
        query = "What is the meaning of sovereignty?"
        analysis = analyzer.analyze(query)
        # estimated_tokens = word_count * 5
        assert analysis.estimated_tokens == len(query.split()) * 5

    def test_domains_default(self, analyzer):
        analysis = analyzer.analyze("test query")
        assert analysis.domains == ["general"]

    def test_requires_reasoning_default(self, analyzer):
        analysis = analyzer.analyze("test query")
        assert analysis.requires_reasoning is True


# ---------------------------------------------------------------------------
# SNRCalculator TESTS
# ---------------------------------------------------------------------------


class TestSNRCalculator:

    @pytest.fixture
    def calc(self):
        return SNRCalculator()

    def test_calculate_returns_all_components(self, calc):
        result = calc.calculate("This is a test response", "test query")
        expected_keys = {
            "snr", "signal", "noise", "relevance", "novelty",
            "groundedness", "coherence", "actionability", "ihsan_compliant",
        }
        assert set(result.keys()) == expected_keys

    def test_all_scores_bounded(self, calc):
        result = calc.calculate(
            "A comprehensive analysis because the evidence shows",
            "analyze the evidence",
        )
        for key in ["snr", "signal", "relevance", "novelty", "groundedness", "coherence", "actionability"]:
            assert 0.0 <= result[key] <= 1.0, f"{key}={result[key]} out of bounds"

    def test_noise_always_positive(self, calc):
        result = calc.calculate("unique words only", "query")
        assert result["noise"] > 0.0

    def test_groundedness_increases_with_markers(self, calc):
        """Responses with grounding markers should score higher on groundedness."""
        plain = calc.calculate("The sky is blue", "color query")
        grounded = calc.calculate(
            "The sky is blue because of Rayleigh scattering, therefore it shows wavelength",
            "color query",
        )
        assert grounded["groundedness"] >= plain["groundedness"]

    def test_ihsan_compliant_flag(self, calc):
        """ihsan_compliant is True when snr >= IHSAN_THRESHOLD."""
        result = calc.calculate("test", "test")
        assert result["ihsan_compliant"] == (result["snr"] >= 0.95)

    def test_empty_response(self, calc):
        """Should not crash on empty response."""
        result = calc.calculate("", "query")
        assert isinstance(result["snr"], float)


# ---------------------------------------------------------------------------
# SimulatedBackend TESTS
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
class TestSimulatedBackend:

    @pytest.mark.asyncio
    async def test_health_check(self):
        backend = SimulatedBackend()
        assert await backend.health_check() is True

    @pytest.mark.asyncio
    async def test_generate(self):
        backend = SimulatedBackend()
        result = await backend.generate("test prompt")
        assert len(result) > 0
        assert "test prompt" in result[:100]  # Prompt is referenced in output

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self):
        backend = SimulatedBackend()
        result = await backend.generate("test", system="You are a test bot")
        assert len(result) > 0

    def test_name(self):
        backend = SimulatedBackend()
        assert backend.name == "Simulated"


# ---------------------------------------------------------------------------
# InferenceGateway TESTS
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
class TestInferenceGateway:

    @pytest.mark.asyncio
    async def test_select_backend_falls_to_simulated(self):
        """When real backends are offline, gateway selects SimulatedBackend."""
        gateway = InferenceGateway()
        # Override real backends with failing health checks
        mock_lm = AsyncMock()
        mock_lm.health_check = AsyncMock(return_value=False)
        mock_lm.name = "LM Studio"
        mock_ollama = AsyncMock()
        mock_ollama.health_check = AsyncMock(return_value=False)
        mock_ollama.name = "Ollama"
        gateway.backends = [mock_lm, mock_ollama, SimulatedBackend()]

        backend = await gateway.select_backend()
        assert backend.name == "Simulated"

    @pytest.mark.asyncio
    async def test_generate_uses_simulated(self):
        gateway = InferenceGateway()
        # Force simulated backend
        gateway._active = SimulatedBackend()
        result, name = await gateway.generate("test prompt")
        assert name == "Simulated"
        assert len(result) > 0

    def test_stats_before_selection(self):
        gateway = InferenceGateway()
        stats = gateway.stats()
        assert stats["active_backend"] is None
        assert stats["backend_stats"] == {}

    @pytest.mark.asyncio
    async def test_stats_after_generation(self):
        gateway = InferenceGateway()
        gateway._active = SimulatedBackend()
        await gateway.generate("test")
        stats = gateway.stats()
        assert stats["active_backend"] == "Simulated"
        assert "Simulated" in stats["backend_stats"]
        assert stats["backend_stats"]["Simulated"]["calls"] == 1

    @pytest.mark.asyncio
    async def test_trivial_complexity_limits_tokens(self):
        """TRIVIAL complexity should cap max_tokens to 100."""
        mock_backend = AsyncMock()
        mock_backend.name = "Mock"
        mock_backend.generate = AsyncMock(return_value="response")

        gateway = InferenceGateway()
        gateway._active = mock_backend

        await gateway.generate("test", complexity=ComplexityTier.TRIVIAL)
        call_args = mock_backend.generate.call_args
        # max_tokens arg should be min(2048, 100) = 100
        assert call_args[0][2] <= 100 or call_args.kwargs.get("max_tokens", 2048) <= 100


# ---------------------------------------------------------------------------
# LLMBackend ABC TESTS
# ---------------------------------------------------------------------------


class TestLLMBackendABC:

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            LLMBackend()  # type: ignore[abstract]
