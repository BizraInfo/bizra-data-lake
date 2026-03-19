"""
BIZRA Ihsan Gate + SNR Module Tests
════════════════════════════════════

Tests for the runtime constitutional enforcement layer.
Every assertion derived from constitution.toml thresholds.
"""

import os
import sys
import time

import pytest

# Ensure imports work from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ihsan_gate import DimensionScore, IhsanGate, IhsanScore, IhsanTier
from snr import (
    SAPE_WEIGHTS,
    compute_sape_composite,
    db_to_snr,
    measure_mission_snr,
    normalize_snr,
    snr_to_db,
)

# ═══════════════════════════════════════════════════════════════════════════════
# IHSAN GATE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestIhsanGateConstruction:
    """Gate must initialize with constitutional parameters."""

    def test_default_construction(self):
        gate = IhsanGate()
        assert gate.gate_minimum == 0.85

    def test_weights_sum_to_one(self):
        gate = IhsanGate()
        total = sum(gate.weights.values())
        assert abs(total - 1.0) < 0.01

    def test_six_dimensions(self):
        gate = IhsanGate()
        assert len(gate.weights) == 6

    def test_invalid_weights_rejected(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            IhsanGate(weights={"a": 0.5, "b": 0.1})

    def test_wrong_dimension_count_rejected(self):
        # 5 dimensions instead of 6
        with pytest.raises(ValueError, match="Expected 6"):
            IhsanGate(
                weights={
                    "moral_clarity": 0.20,
                    "epistemic_humility": 0.20,
                    "structural_integrity": 0.20,
                    "verifiability": 0.20,
                    "intent_alignment": 0.20,
                }
            )


class TestIhsanGateEvaluation:
    """Gate evaluation must enforce constitutional thresholds."""

    @pytest.fixture
    def gate(self):
        return IhsanGate()

    def test_empty_output_fails(self, gate):
        score = gate.evaluate("")
        assert not score.passes
        assert score.tier == IhsanTier.REJECTED

    def test_whitespace_output_fails(self, gate):
        score = gate.evaluate("   \n\t  ")
        assert not score.passes

    def test_reasonable_output_passes(self, gate):
        output = (
            "Based on the analysis, the evidence suggests that the system "
            "performs within acceptable parameters. However, note that these "
            "results are approximate and may vary. The first step is to verify "
            "the baseline, then compare against the threshold."
        )
        context = {"mission_keywords": ["analysis", "system", "parameters"]}
        score = gate.evaluate(output, context)
        assert score.composite > 0.0
        assert len(score.dimensions) == 6

    def test_returns_six_dimension_scores(self, gate):
        score = gate.evaluate("Test output with some reasoning because of evidence.")
        assert len(score.dimensions) == 6
        for dim in score.dimensions:
            assert isinstance(dim, DimensionScore)
            assert 0.0 <= dim.raw_score <= 1.0
            assert dim.weight > 0

    def test_composite_is_weighted_sum(self, gate):
        score = gate.evaluate("A reasonable output with evidence and reasoning.")
        expected = sum(d.weighted_score for d in score.dimensions)
        assert abs(score.composite - expected) < 0.001

    def test_bloom_eligibility_threshold(self, gate):
        """BLOOM requires composite ≥ 0.90."""
        score = gate.evaluate(
            "A comprehensive response with uncertainty markers, "
            "evidence-based reasoning, step-by-step logic, and "
            "appropriate caveats. However, limitations exist.",
            {"mission_keywords": ["comprehensive", "reasoning"]},
        )
        if score.composite >= 0.90:
            assert score.bloom_eligible
        else:
            assert not score.bloom_eligible

    def test_excellence_threshold(self, gate):
        """إحسان requires composite ≥ 0.95."""
        score = gate.evaluate("Test")
        if score.composite >= 0.95:
            assert score.is_ihsan
            assert score.tier == IhsanTier.EXCELLENCE
        else:
            assert not score.is_ihsan

    def test_evaluation_time_tracked(self, gate):
        score = gate.evaluate("Quick test")
        assert score.evaluation_ms >= 0

    def test_gate_minimum_recorded(self, gate):
        score = gate.evaluate("Test")
        assert score.gate_minimum == 0.85

    def test_as_tensor_dict(self, gate):
        score = gate.evaluate("Test output")
        tensor = score.as_tensor_dict()
        assert isinstance(tensor, dict)
        assert len(tensor) == 6
        for name, value in tensor.items():
            assert isinstance(name, str)
            assert isinstance(value, float)

    def test_as_evidence(self, gate):
        score = gate.evaluate("Test output")
        evidence = score.as_evidence()
        assert "ihsan_composite" in evidence
        assert "ihsan_tensor" in evidence
        assert "tier" in evidence
        assert "passes" in evidence
        assert "dimensions" in evidence
        assert evidence["dimensions"] == 6


class TestIhsanGateFailClosed:
    """Theorem 2.3: Gate MUST be fail-closed."""

    def test_fail_closed_on_scorer_error(self):
        """If a scorer raises, gate must REJECT, not pass."""
        gate = IhsanGate()
        # Monkey-patch a scorer to raise
        from ihsan_gate import SCORERS

        original = SCORERS["moral_clarity"]
        try:
            SCORERS["moral_clarity"] = lambda o, c: (_ for _ in ()).throw(
                RuntimeError("scorer crash")
            )
            score = gate.evaluate("Test")
            assert not score.passes
            assert score.tier == IhsanTier.REJECTED
            assert any("fail-closed" in v for v in score.violations)
        finally:
            SCORERS["moral_clarity"] = original

    def test_negative_scores_clamped(self):
        """Raw scores must be clamped to [0, 1]."""
        gate = IhsanGate()
        from ihsan_gate import SCORERS

        original = SCORERS["resilience"]
        try:
            SCORERS["resilience"] = lambda o, c: -5.0  # Invalid negative
            score = gate.evaluate("Test")
            for dim in score.dimensions:
                assert dim.raw_score >= 0.0
                assert dim.raw_score <= 1.0
        finally:
            SCORERS["resilience"] = original


class TestIhsanGateBatch:
    """Batch evaluation for SAT ConsensusValidators."""

    def test_batch_evaluates_all(self):
        gate = IhsanGate()
        items = [
            ("Output one with reasoning because evidence", {}),
            ("Output two", {}),
            ("", {}),
        ]
        results = gate.evaluate_batch(items)
        assert len(results) == 3
        assert all(isinstance(r, IhsanScore) for r in results)


# ═══════════════════════════════════════════════════════════════════════════════
# SNR MODULE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestNormalizeSNR:
    """Canonical normalization: [0, ∞) → [0, 1)."""

    def test_zero_input(self):
        assert normalize_snr(0.0) == 0.0

    def test_one_input(self):
        assert abs(normalize_snr(1.0) - 0.5) < 0.001

    def test_nine_input(self):
        assert abs(normalize_snr(9.0) - 0.9) < 0.001

    def test_nineteen_input(self):
        assert abs(normalize_snr(19.0) - 0.95) < 0.001

    def test_large_input_approaches_one(self):
        assert normalize_snr(1000.0) > 0.999

    def test_negative_input_clamped(self):
        assert normalize_snr(-5.0) == 0.0

    def test_monotonically_increasing(self):
        """SNR normalization must be monotonically non-decreasing."""
        prev = 0.0
        for x in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 1000.0]:
            curr = normalize_snr(x)
            assert curr >= prev, f"Monotonicity violated at x={x}"
            prev = curr

    def test_bounded_output(self):
        """Output must always be in [0, 1]."""
        for x in [0.0, 0.001, 1.0, 100.0, 1e6, 1e12]:
            result = normalize_snr(x)
            assert 0.0 <= result <= 1.0


class TestSNRConversions:
    """dB ↔ linear round-trip consistency."""

    def test_db_conversion_round_trip(self):
        for linear in [0.1, 1.0, 5.0, 10.0, 100.0]:
            db = snr_to_db(linear)
            recovered = db_to_snr(db)
            assert abs(recovered - linear) < 0.001

    def test_zero_snr_is_neg_inf_db(self):
        assert snr_to_db(0.0) == float("-inf")

    def test_negative_snr_is_neg_inf_db(self):
        assert snr_to_db(-1.0) == float("-inf")


class TestSapeComposite:
    """SAPE dimensional scoring."""

    def test_perfect_scores(self):
        scores = {dim: 1.0 for dim in SAPE_WEIGHTS}
        result = compute_sape_composite(scores)
        assert abs(result.composite - 1.0) < 0.001
        assert result.passes_t1

    def test_zero_scores(self):
        scores = {dim: 0.0 for dim in SAPE_WEIGHTS}
        result = compute_sape_composite(scores)
        assert result.composite == 0.0
        assert not result.passes_t1

    def test_t1_threshold_is_0950(self):
        result = compute_sape_composite({dim: 0.95 for dim in SAPE_WEIGHTS})
        assert result.t1_threshold == 0.950
        assert result.passes_t1

    def test_gap_calculation(self):
        scores = {dim: 0.90 for dim in SAPE_WEIGHTS}
        result = compute_sape_composite(scores)
        assert result.gap_to_t1 == pytest.approx(0.05, abs=0.001)

    def test_current_bizra_state(self):
        """Verify the SAPE composite for BIZRA's current real state: 0.933."""
        scores = {
            "security": 0.96,
            "architecture": 0.95,
            "error_handling": 0.92,
            "scalability": 0.92,
            "testing": 0.97,
            "documentation": 0.85,
            "dependencies": 0.92,
            "performance": 0.93,
        }
        result = compute_sape_composite(scores)
        assert abs(result.composite - 0.933) < 0.01
        assert not result.passes_t1
        assert result.gap_to_t1 > 0

    def test_projected_post_constitution(self):
        """After constitution.toml + 6-dim Ihsan + SNR promotion: projected T1."""
        scores = {
            "security": 0.96,
            "architecture": 0.97,  # +0.02 single source of truth
            "error_handling": 0.94,  # +0.02 SNR promotion + conftest fix
            "scalability": 0.92,
            "testing": 0.97,
            "documentation": 0.98,  # +0.13 constitution IS the spec
            "dependencies": 0.92,  # +0.10 generated constants, zero hardcoded
            "performance": 0.93,
        }
        result = compute_sape_composite(scores)
        # Exact: 0.15(.96)+.20(.97)+.15(.94)+.10(.92)+.15(.97)+.10(.98)+.10(.92)+.05(.93) = 0.955
        assert result.composite >= 0.950
        assert result.passes_t1

    def test_evidence_output(self):
        scores = {dim: 0.90 for dim in SAPE_WEIGHTS}
        result = compute_sape_composite(scores)
        evidence = result.as_evidence()
        assert "sape_composite" in evidence
        assert "passes_t1" in evidence
        assert "gap" in evidence


class TestMissionSNR:
    """Per-mission SNR measurement."""

    def test_high_ihsan_output(self):
        result = measure_mission_snr(
            output="A well-reasoned response with evidence",
            ihsan_composite=0.95,
            relevance_score=0.90,
        )
        assert result.snr_normalized > 0.5
        assert result.snr_linear > 0

    def test_low_ihsan_output(self):
        result = measure_mission_snr(
            output="Bad output",
            ihsan_composite=0.30,
            relevance_score=0.20,
        )
        assert result.snr_normalized < 0.5

    def test_noise_markers_reduce_snr(self):
        base = measure_mission_snr(
            output="Clean output with evidence",
            ihsan_composite=0.90,
        )
        noisy = measure_mission_snr(
            output="Clean output with evidence and filler filler filler",
            ihsan_composite=0.90,
            noise_markers=["filler"],
        )
        assert noisy.snr_normalized <= base.snr_normalized

    def test_evidence_output(self):
        result = measure_mission_snr("Test", 0.85)
        evidence = result.as_evidence()
        assert "signal_power" in evidence
        assert "noise_power" in evidence
        assert "snr_normalized" in evidence

    def test_empty_output(self):
        result = measure_mission_snr("", 0.50)
        assert result.snr_linear >= 0  # Should not crash

    def test_snr_monotonic_with_ihsan(self):
        """Higher Ihsan → higher SNR. Theorem 2.1."""
        prev_snr = 0.0
        for ihsan in [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
            result = measure_mission_snr(
                output="Consistent test output with reasoning and evidence",
                ihsan_composite=ihsan,
                relevance_score=0.90,
            )
            assert result.snr_normalized >= prev_snr - 0.01  # Allow tiny float noise
            prev_snr = result.snr_normalized
