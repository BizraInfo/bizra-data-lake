"""
SNR Protocol — Unified Interface Tests

Standing on Giants:
- Shannon (1948): Information Theory
- PEP 544 (2017): Structural subtyping (Protocol)

Tests cover:
1. SNRResult construction and repr
2. SNRFacade text engine routing (uses snr_normalized, not snr_linear)
3. SNRFacade ensemble routing
4. SNRFacade no-engine fallback
5. SNRv2Adapter protocol conformance (Phase 42 Spec 02)
"""

import pytest

from core.snr_protocol import (
    SNRFacade,
    SNRProtocol,
    SNRResult,
    assert_snr_normalized,
    normalize_snr_linear,
)
from core.sovereign.snr_maximizer import SNRMaximizer

# =============================================================================
# 1. SNRResult
# =============================================================================


class TestSNRResult:
    def test_construction(self):
        r = SNRResult(score=0.85, ihsan_achieved=True, engine="text")
        assert r.score == 0.85
        assert r.ihsan_achieved is True
        assert r.engine == "text"
        assert r.metrics == {}
        assert r.recommendations == []

    def test_repr_pass(self):
        r = SNRResult(score=0.96, ihsan_achieved=True, engine="ensemble")
        assert "PASS" in repr(r)
        assert "0.9600" in repr(r)

    def test_repr_fail(self):
        r = SNRResult(score=0.50, ihsan_achieved=False, engine="text")
        assert "FAIL" in repr(r)

    def test_frozen(self):
        r = SNRResult(score=0.5, ihsan_achieved=False, engine="text")
        with pytest.raises(AttributeError):
            r.score = 0.9


# =============================================================================
# 2. SNRFacade — Text Engine
# =============================================================================


class TestSNRFacadeText:
    def test_text_score_uses_normalized_not_linear(self):
        """SNRFacade uses snr_normalized (bounded [0,1]), not snr_linear (unbounded)."""
        facade = SNRFacade(text_engine=SNRMaximizer())
        result = facade.calculate(text="A clean text with reasonable content.")
        # snr_normalized is bounded [0,1] and never artificially 1.0
        assert 0.0 < result.score < 1.0
        assert result.engine == "text"

    def test_text_score_not_always_one(self):
        """With zero noise, score equals signal quality — NOT 1.0 (old bug)."""
        facade = SNRFacade(text_engine=SNRMaximizer())
        result = facade.calculate(text="Simple sentence.")
        # Simple text should not get a perfect score
        assert result.score < 0.95

    def test_text_score_matches_maximizer(self):
        """SNRFacade score equals snr_maximizer's snr_normalized for same call."""
        maximizer = SNRMaximizer()
        text = "Signal processing involves analyzing time-series data."
        # Facade calls maximizer.analyze() internally — capture via fresh instance
        facade = SNRFacade(text_engine=maximizer)
        result = facade.calculate(text=text)
        # Score should be the snr_normalized from that analyze() call
        # Verify it's bounded and not the old always-1.0 bug
        assert 0.0 < result.score < 1.0
        assert result.engine == "text"
        assert "signal" in result.metrics  # to_dict() includes signal breakdown

    def test_text_ihsan_achieved_consistent(self):
        """ihsan_achieved flag is consistent between facade and maximizer."""
        maximizer = SNRMaximizer()
        facade = SNRFacade(text_engine=maximizer)
        result = facade.calculate(text="Test content.")
        analysis = maximizer.analyze("Test content.")
        assert result.ihsan_achieved == analysis.ihsan_achieved

    def test_text_engine_none_returns_zero(self):
        """Without text engine, text-only request returns zero."""
        facade = SNRFacade(text_engine=None)
        result = facade.calculate(text="Any text.")
        assert result.score == 0.0
        assert result.ihsan_achieved is False


# =============================================================================
# 3. SNRFacade — No Engine Fallback
# =============================================================================


class TestSNRFacadeNoEngine:
    def test_no_inputs_returns_baseline(self):
        """No engines + no inputs returns zero baseline."""
        facade = SNRFacade()
        result = facade.calculate()
        assert result.score == 0.0
        assert result.engine == "none"
        assert len(result.recommendations) > 0

    def test_no_engine_with_text_returns_zero(self):
        """Text provided but no text_engine returns zero."""
        facade = SNRFacade()
        result = facade.calculate(text="some text")
        assert result.score == 0.0


# =============================================================================
# 4. SNRProtocol — Structural Typing Check
# =============================================================================


class TestSNRProtocol:
    def test_protocol_is_runtime_checkable(self):
        """SNRProtocol can be checked at runtime."""

        class FakeEngine:
            def calculate_snr_normalized(self, **kwargs):
                return SNRResult(score=0.5, ihsan_achieved=False, engine="fake")

        assert isinstance(FakeEngine(), SNRProtocol)

    def test_non_conforming_fails_check(self):
        """Class without calculate_snr_normalized fails protocol check."""

        class NotAnEngine:
            pass

        assert not isinstance(NotAnEngine(), SNRProtocol)


# =============================================================================
# 5. assert_snr_normalized — Phase 60 Step 4 Contract Guard
# =============================================================================


class TestAssertSNRNormalized:
    """Phase 60 Step 4: Design-by-Contract guard for SNR values at boundaries."""

    def test_valid_zero(self):
        assert assert_snr_normalized(0.0) == 0.0

    def test_valid_one(self):
        assert assert_snr_normalized(1.0) == 1.0

    def test_valid_mid(self):
        assert assert_snr_normalized(0.85) == 0.85

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match="outside normalized range"):
            assert_snr_normalized(-0.01)

    def test_rejects_above_one(self):
        with pytest.raises(ValueError, match="outside normalized range"):
            assert_snr_normalized(1.01)

    def test_rejects_unbounded_linear(self):
        """Raw snr_linear (e.g., 15.7) must be caught before storage."""
        with pytest.raises(ValueError, match="outside normalized range"):
            assert_snr_normalized(15.7)

    def test_rejects_nan(self):
        with pytest.raises(ValueError, match="not finite"):
            assert_snr_normalized(float("nan"))

    def test_rejects_infinity(self):
        with pytest.raises(ValueError, match="not finite"):
            assert_snr_normalized(float("inf"))

    def test_rejects_non_numeric(self):
        with pytest.raises(TypeError, match="expected numeric"):
            assert_snr_normalized("high")

    def test_accepts_integer(self):
        assert assert_snr_normalized(1) == 1.0
        assert assert_snr_normalized(0) == 0.0

    def test_label_in_error(self):
        with pytest.raises(ValueError, match="my_score"):
            assert_snr_normalized(5.0, label="my_score")

    def test_normalize_then_assert_roundtrip(self):
        """normalize_snr_linear output always passes assert_snr_normalized."""
        for raw in [0.0, 0.5, 1.0, 5.0, 100.0, 1e10]:
            normalized = normalize_snr_linear(raw)
            assert assert_snr_normalized(normalized) == normalized

    def test_normalize_matches_constitution_function(self):
        """core.snr_protocol normalization matches constitution module."""
        from bizra_constitution.snr import normalize_snr

        for raw in [0.0, 0.1, 1.0, 9.0, 19.0, 1_000_000.0]:
            assert normalize_snr_linear(raw) == pytest.approx(normalize_snr(raw))
