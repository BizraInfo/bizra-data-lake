"""
SNR Rust Adapter — Protocol Conformance and Facade Integration Tests

Standing on Giants:
- Shannon (1948): Cross-language SNR measurement
- PEP 544 (2017): Structural subtyping via Protocol

Tests the Rust→Python SNR bridge (Gap G-2) using mocked Rust binding.
No maturin build required — all Rust interactions are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.snr_protocol import SNRFacade, SNRProtocol, SNRResult


# ── Fixtures ────────────────────────────────────────────────────────────


def _make_mock_metrics(
    snr: float = 0.92,
    signal_strength: float = 0.88,
    noise_level: float = 0.05,
    diversity: float = 0.75,
    grounding: float = 0.85,
    balance: float = 0.80,
    word_count: int = 50,
    unique_words: int = 40,
    analysis_duration_us: int = 120,
) -> dict:
    """Create a mock SignalMetrics dict matching Rust PySNREngine output."""
    return {
        "snr": snr,
        "signal_strength": signal_strength,
        "noise_level": noise_level,
        "diversity": diversity,
        "grounding": grounding,
        "balance": balance,
        "input_size": word_count * 5,  # approximate
        "word_count": word_count,
        "unique_words": unique_words,
        "analysis_duration_us": analysis_duration_us,
    }


def _make_mock_rust_engine(metrics: dict | None = None) -> MagicMock:
    """Create a mock PySNREngine that returns given metrics."""
    engine = MagicMock()
    engine.analyze_text.return_value = metrics or _make_mock_metrics()
    engine.average_snr.return_value = 0.91
    engine.stats.return_value = {
        "total_measurements": 42,
        "history_size": 42,
        "average_snr": 0.91,
        "snr_floor": 0.85,
        "ihsan_target": 0.95,
    }
    return engine


# ── SNRRustAdapter Tests ──────────────────────────────────────────────


class TestSNRRustAdapter:
    """Tests for core/iaas/snr_rust_adapter.py with mocked Rust binding."""

    def test_conforms_to_protocol(self):
        """SNRRustAdapter satisfies SNRProtocol structural typing."""
        # Patch the import to make Rust binding appear available
        mock_engine = _make_mock_rust_engine()
        with patch("core.iaas.snr_rust_adapter._RUST_SNR_AVAILABLE", True), \
             patch("core.iaas.snr_rust_adapter._RustSNREngine", lambda **kw: mock_engine):
            from core.iaas.snr_rust_adapter import SNRRustAdapter
            adapter = SNRRustAdapter.__new__(SNRRustAdapter)
            adapter._engine = mock_engine
            adapter._ihsan_threshold = 0.95
            assert isinstance(adapter, SNRProtocol)

    def test_calculate_returns_snr_result(self):
        """calculate_snr_normalized returns canonical SNRResult."""
        mock_engine = _make_mock_rust_engine()
        with patch("core.iaas.snr_rust_adapter._RUST_SNR_AVAILABLE", True), \
             patch("core.iaas.snr_rust_adapter._RustSNREngine", lambda **kw: mock_engine):
            from core.iaas.snr_rust_adapter import SNRRustAdapter
            adapter = SNRRustAdapter.__new__(SNRRustAdapter)
            adapter._engine = mock_engine
            adapter._ihsan_threshold = 0.95

            result = adapter.calculate_snr_normalized(text="test content")
            assert isinstance(result, SNRResult)
            assert result.score == 0.92
            assert result.engine == "rust"
            mock_engine.analyze_text.assert_called_once_with("test content")

    def test_ihsan_achieved_above_threshold(self):
        """Score above ihsan_target sets ihsan_achieved=True."""
        metrics = _make_mock_metrics(snr=0.97)
        mock_engine = _make_mock_rust_engine(metrics)
        with patch("core.iaas.snr_rust_adapter._RUST_SNR_AVAILABLE", True), \
             patch("core.iaas.snr_rust_adapter._RustSNREngine", lambda **kw: mock_engine):
            from core.iaas.snr_rust_adapter import SNRRustAdapter
            adapter = SNRRustAdapter.__new__(SNRRustAdapter)
            adapter._engine = mock_engine
            adapter._ihsan_threshold = 0.95

            result = adapter.calculate_snr_normalized(text="high quality content")
            assert result.ihsan_achieved is True

    def test_ihsan_not_achieved_below_threshold(self):
        """Score below ihsan_target sets ihsan_achieved=False."""
        metrics = _make_mock_metrics(snr=0.80)
        mock_engine = _make_mock_rust_engine(metrics)
        with patch("core.iaas.snr_rust_adapter._RUST_SNR_AVAILABLE", True), \
             patch("core.iaas.snr_rust_adapter._RustSNREngine", lambda **kw: mock_engine):
            from core.iaas.snr_rust_adapter import SNRRustAdapter
            adapter = SNRRustAdapter.__new__(SNRRustAdapter)
            adapter._engine = mock_engine
            adapter._ihsan_threshold = 0.95

            result = adapter.calculate_snr_normalized(text="low quality")
            assert result.ihsan_achieved is False

    def test_metrics_include_rust_fields(self):
        """Result metrics contain all Rust SignalMetrics fields."""
        mock_engine = _make_mock_rust_engine()
        with patch("core.iaas.snr_rust_adapter._RUST_SNR_AVAILABLE", True), \
             patch("core.iaas.snr_rust_adapter._RustSNREngine", lambda **kw: mock_engine):
            from core.iaas.snr_rust_adapter import SNRRustAdapter
            adapter = SNRRustAdapter.__new__(SNRRustAdapter)
            adapter._engine = mock_engine
            adapter._ihsan_threshold = 0.95

            result = adapter.calculate_snr_normalized(text="test")
            for key in ("signal_strength", "diversity", "grounding", "balance",
                        "noise_level", "word_count", "unique_words",
                        "analysis_duration_us", "aggregation"):
                assert key in result.metrics, f"Missing metric: {key}"
            assert result.metrics["aggregation"] == "weighted_geometric_mean"

    def test_empty_text_returns_zero(self):
        """Empty text input returns zero-score SNRResult."""
        mock_engine = _make_mock_rust_engine()
        with patch("core.iaas.snr_rust_adapter._RUST_SNR_AVAILABLE", True), \
             patch("core.iaas.snr_rust_adapter._RustSNREngine", lambda **kw: mock_engine):
            from core.iaas.snr_rust_adapter import SNRRustAdapter
            adapter = SNRRustAdapter.__new__(SNRRustAdapter)
            adapter._engine = mock_engine
            adapter._ihsan_threshold = 0.95

            result = adapter.calculate_snr_normalized(text="")
            assert result.score == 0.0
            assert result.ihsan_achieved is False
            mock_engine.analyze_text.assert_not_called()

    def test_rust_error_returns_zero(self):
        """If Rust engine raises, adapter returns zero SNRResult gracefully."""
        mock_engine = _make_mock_rust_engine()
        mock_engine.analyze_text.side_effect = ValueError("Input too large")
        with patch("core.iaas.snr_rust_adapter._RUST_SNR_AVAILABLE", True), \
             patch("core.iaas.snr_rust_adapter._RustSNREngine", lambda **kw: mock_engine):
            from core.iaas.snr_rust_adapter import SNRRustAdapter
            adapter = SNRRustAdapter.__new__(SNRRustAdapter)
            adapter._engine = mock_engine
            adapter._ihsan_threshold = 0.95

            result = adapter.calculate_snr_normalized(text="x" * 2_000_000)
            assert result.score == 0.0
            assert "failed" in result.recommendations[0].lower()

    def test_recommendations_for_low_signal(self):
        """Low signal_strength triggers improvement recommendation."""
        metrics = _make_mock_metrics(signal_strength=0.3)
        mock_engine = _make_mock_rust_engine(metrics)
        with patch("core.iaas.snr_rust_adapter._RUST_SNR_AVAILABLE", True), \
             patch("core.iaas.snr_rust_adapter._RustSNREngine", lambda **kw: mock_engine):
            from core.iaas.snr_rust_adapter import SNRRustAdapter
            adapter = SNRRustAdapter.__new__(SNRRustAdapter)
            adapter._engine = mock_engine
            adapter._ihsan_threshold = 0.95

            result = adapter.calculate_snr_normalized(text="sparse content")
            assert any("density" in r.lower() for r in result.recommendations)

    def test_stats_delegates(self):
        """stats() delegates to Rust engine."""
        mock_engine = _make_mock_rust_engine()
        with patch("core.iaas.snr_rust_adapter._RUST_SNR_AVAILABLE", True), \
             patch("core.iaas.snr_rust_adapter._RustSNREngine", lambda **kw: mock_engine):
            from core.iaas.snr_rust_adapter import SNRRustAdapter
            adapter = SNRRustAdapter.__new__(SNRRustAdapter)
            adapter._engine = mock_engine
            adapter._ihsan_threshold = 0.95

            stats = adapter.stats()
            assert stats["total_measurements"] == 42
            assert stats["average_snr"] == 0.91


class TestCreateRustSNRAdapter:
    """Tests for the factory function."""

    def test_returns_none_when_unavailable(self):
        """Factory returns None when Rust binding not available."""
        with patch("core.iaas.snr_rust_adapter._RUST_SNR_AVAILABLE", False):
            from core.iaas.snr_rust_adapter import create_rust_snr_adapter
            result = create_rust_snr_adapter()
            assert result is None

    def test_is_rust_snr_available_false(self):
        """is_rust_snr_available returns False when binding missing."""
        with patch("core.iaas.snr_rust_adapter._RUST_SNR_AVAILABLE", False):
            from core.iaas.snr_rust_adapter import is_rust_snr_available
            assert is_rust_snr_available() is False


# ── SNRFacade Rust Engine Routing Tests ──────────────────────────────


class TestSNRFacadeRustRouting:
    """Tests for SNRFacade with rust_engine parameter."""

    def _make_rust_adapter_mock(self, snr: float = 0.92) -> MagicMock:
        """Create a mock SNRRustAdapter."""
        adapter = MagicMock()
        adapter.calculate_snr_normalized.return_value = SNRResult(
            score=snr,
            ihsan_achieved=snr >= 0.95,
            engine="rust",
            metrics={"aggregation": "weighted_geometric_mean"},
        )
        return adapter

    def test_rust_engine_highest_priority(self):
        """Rust engine is used when available, even if v2 and text also available."""
        rust = self._make_rust_adapter_mock(snr=0.93)
        v2 = MagicMock()
        text = MagicMock()

        facade = SNRFacade(rust_engine=rust, v2_engine=v2, text_engine=text)
        result = facade.calculate(text="test content", query="test")

        assert result.engine == "rust"
        assert result.score == 0.93
        rust.calculate_snr_normalized.assert_called_once()
        v2.calculate_snr_normalized.assert_not_called()

    def test_fallback_to_v2_when_rust_none(self):
        """Without rust_engine, facade falls back to v2+text ensemble."""
        from core.iaas.snr_v2 import SNRCalculatorV2
        from core.iaas.snr_v2_adapter import SNRv2Adapter
        from core.sovereign.snr_maximizer import SNRMaximizer

        facade = SNRFacade(
            v2_engine=SNRv2Adapter(SNRCalculatorV2()),
            text_engine=SNRMaximizer(),
        )
        result = facade.calculate(text="Signal processing analysis.", query="signal")
        assert result.engine == "ensemble_v2"  # Not "rust"

    def test_rust_failure_falls_back_to_v2(self):
        """If rust_engine raises, facade falls back to v2_engine."""
        rust = self._make_rust_adapter_mock()
        rust.calculate_snr_normalized.side_effect = RuntimeError("Rust crashed")

        from core.iaas.snr_v2 import SNRCalculatorV2
        from core.iaas.snr_v2_adapter import SNRv2Adapter

        v2 = SNRv2Adapter(SNRCalculatorV2())

        facade = SNRFacade(rust_engine=rust, v2_engine=v2)
        result = facade.calculate(text="Fallback test content.", query="test")
        # Should fall back to v2
        assert result.engine == "snr_v2"
        assert result.score > 0.0

    def test_rust_failure_falls_back_to_text(self):
        """If rust_engine fails and no v2, falls back to text engine."""
        rust = self._make_rust_adapter_mock()
        rust.calculate_snr_normalized.side_effect = RuntimeError("Rust crashed")

        from core.sovereign.snr_maximizer import SNRMaximizer

        facade = SNRFacade(rust_engine=rust, text_engine=SNRMaximizer())
        result = facade.calculate(text="Fallback to text engine.")
        assert result.engine == "text"

    def test_backward_compat_no_rust_no_v2(self):
        """rust_engine=None, v2_engine=None preserves text-only behavior."""
        from core.sovereign.snr_maximizer import SNRMaximizer

        facade = SNRFacade(text_engine=SNRMaximizer())
        result = facade.calculate(text="Test backward compatibility.")
        assert result.engine == "text"
        assert 0.0 < result.score < 1.0

    def test_rust_with_no_text_skips_rust(self):
        """Rust engine is skipped when no text is provided."""
        rust = self._make_rust_adapter_mock()
        facade = SNRFacade(rust_engine=rust)
        result = facade.calculate()  # No text
        assert result.engine == "none"
        rust.calculate_snr_normalized.assert_not_called()
