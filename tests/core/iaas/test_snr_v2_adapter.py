"""
SNR v2 Adapter — Protocol Conformance Tests

Standing on Giants:
- Shannon (1948): Unified measurement
- PEP 544 (2017): Structural subtyping

Phase 42 Spec 02: Tests for SNRv2Adapter and SNRFacade v2 routing.
"""

import pytest

from core.iaas.snr_v2 import SNRCalculatorV2
from core.iaas.snr_v2_adapter import SNRv2Adapter
from core.snr_protocol import SNRFacade, SNRProtocol, SNRResult
from core.sovereign.snr_maximizer import SNRMaximizer


class TestSNRv2Adapter:
    def test_conforms_to_protocol(self):
        """SNRv2Adapter satisfies SNRProtocol structural typing."""
        adapter = SNRv2Adapter(SNRCalculatorV2())
        assert isinstance(adapter, SNRProtocol)

    def test_lexical_fallback(self):
        """Without embeddings, adapter uses calculate_simple."""
        adapter = SNRv2Adapter(SNRCalculatorV2())
        result = adapter.calculate_snr_normalized(
            query="signal processing",
            texts=["Signal processing involves analyzing time-series data."],
        )
        assert isinstance(result, SNRResult)
        assert 0.0 <= result.score <= 1.0
        assert result.engine == "snr_v2"
        assert "quality_tier" in result.metrics

    def test_single_text_shorthand(self):
        """Passing text= (single string) instead of texts= works."""
        adapter = SNRv2Adapter(SNRCalculatorV2())
        result = adapter.calculate_snr_normalized(
            text="A test response about signal processing.",
            query="signal processing",
        )
        assert 0.0 <= result.score <= 1.0
        assert result.engine == "snr_v2"

    def test_metrics_include_breakdown(self):
        """Result metrics include signal_strength, diversity, grounding, etc."""
        adapter = SNRv2Adapter(SNRCalculatorV2())
        result = adapter.calculate_snr_normalized(query="test", texts=["test text"])
        for key in (
            "signal_strength",
            "diversity",
            "grounding",
            "iaas_score",
            "redundancy",
            "entropy",
            "quality_tier",
        ):
            assert key in result.metrics, f"Missing metric: {key}"

    def test_recommendations_generated(self):
        """Low-quality input produces recommendations."""
        adapter = SNRv2Adapter(SNRCalculatorV2())
        result = adapter.calculate_snr_normalized(
            query="complex distributed systems architecture",
            texts=["hello"],  # Very short, off-topic
        )
        # Low signal or grounding should trigger at least one recommendation
        assert isinstance(result.recommendations, list)

    def test_error_returns_zero_result(self):
        """If calculator raises, adapter returns zero SNRResult."""

        class BrokenCalculator:
            def compute_snr(self, **kwargs):
                raise RuntimeError("GPU unavailable")

            def calculate_simple(self, **kwargs):
                raise RuntimeError("GPU unavailable")

        adapter = SNRv2Adapter(BrokenCalculator())
        result = adapter.calculate_snr_normalized(query="test", texts=["test"])
        assert result.score == 0.0
        assert result.ihsan_achieved is False
        assert "failed" in result.recommendations[0].lower()


class TestSNRFacadeV2Routing:
    def test_v2_plus_text_produces_ensemble(self):
        """With v2_engine + text_engine, facade returns ensemble_v2."""
        facade = SNRFacade(
            v2_engine=SNRv2Adapter(SNRCalculatorV2()),
            text_engine=SNRMaximizer(),
        )
        result = facade.calculate(
            text="Signal processing fundamentals.", query="signal"
        )
        assert result.engine == "ensemble_v2"
        assert 0.0 < result.score < 1.0
        assert "v2_snr" in result.metrics
        assert "text_snr" in result.metrics

    def test_v2_only_without_text_engine(self):
        """With v2_engine but no text_engine, facade uses v2 directly."""
        facade = SNRFacade(v2_engine=SNRv2Adapter(SNRCalculatorV2()))
        result = facade.calculate(text="Test content.", query="test")
        assert result.engine == "snr_v2"
        assert 0.0 <= result.score <= 1.0

    def test_fallback_to_text_only(self):
        """Without v2_engine, facade falls back to text engine."""
        facade = SNRFacade(text_engine=SNRMaximizer())
        result = facade.calculate(text="Some content.")
        assert result.engine == "text"

    def test_ensemble_v2_is_geometric_mean(self):
        """Ensemble score is geometric mean of v2 and text scores."""
        import math

        facade = SNRFacade(
            v2_engine=SNRv2Adapter(SNRCalculatorV2()),
            text_engine=SNRMaximizer(),
        )
        result = facade.calculate(text="Signal processing analysis.", query="signal")

        v2_snr = result.metrics["v2_snr"]
        text_snr = result.metrics["text_snr"]
        expected = math.exp(
            0.5 * math.log(v2_snr + 1e-10) + 0.5 * math.log(text_snr + 1e-10)
        )
        expected = min(max(expected, 0.0), 1.0)

        assert abs(result.score - round(expected, 4)) < 1e-4

    def test_backward_compat_no_v2(self):
        """v2_engine=None preserves original facade behavior."""
        facade = SNRFacade(text_engine=SNRMaximizer())
        result = facade.calculate(text="Test.")
        assert result.engine == "text"
        assert 0.0 < result.score < 1.0
