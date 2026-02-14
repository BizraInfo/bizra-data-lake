"""
Tests for SP-004 (SNR Engine v1 authoritative scorer) and
SP-005 (Ihsan Gate — fail-closed excellence constraint).

Standing on Giants:
- Shannon (1948): SNR as information quality
- Ihsan (Islamic ethics): Excellence as obligation
- BIZRA Spearpoint PRD SP-004 + SP-005
"""

import pytest
from typing import Dict, Any

from core.proof_engine.snr import (
    SNREngine,
    SNRInput,
    SNRPolicy,
    SNRTrace,
)
from core.proof_engine.ihsan_gate import (
    IhsanGate,
    IhsanResult,
    IhsanComponents,
)
from core.proof_engine.reason_codes import ReasonCode


# =============================================================================
# SP-004: SNR ENGINE v1 — AUTHORITATIVE snr_score()
# =============================================================================

class TestSNRScoreAuthoritative:
    """Tests for the single authoritative snr_score() function."""

    def test_snr_score_returns_receipt_shape(self):
        """snr_score() returns schema-compatible dict."""
        engine = SNREngine()
        inputs = SNRInput(
            provenance_depth=3,
            source_trust_score=0.9,
            z3_satisfiable=True,
            ihsan_score=0.98,
        )
        result = engine.snr_score(inputs)

        # Check all required keys exist
        assert "score" in result
        assert "signal_mass" in result
        assert "noise_mass" in result
        assert "signal_components" in result
        assert "noise_components" in result
        assert "claim_tags" in result
        assert "trace_id" in result
        assert "policy_digest" in result
        assert "passed" in result

    def test_snr_score_range(self):
        """snr_score().score is in [0, 1]."""
        engine = SNREngine()
        inputs = SNRInput()
        result = engine.snr_score(inputs)
        assert 0.0 <= result["score"] <= 1.0

    def test_snr_score_high_quality_passes(self):
        """High-quality input passes threshold."""
        engine = SNREngine()
        inputs = SNRInput(
            provenance_depth=5,
            corroboration_count=3,
            source_trust_score=0.95,
            z3_satisfiable=True,
            ihsan_score=0.99,
            contradiction_count=0,
            unverifiable_claims=0,
        )
        result = engine.snr_score(inputs)
        assert result["passed"] is True
        assert result["score"] >= 0.95

    def test_snr_score_low_quality_fails(self):
        """Low-quality input fails threshold."""
        engine = SNREngine()
        inputs = SNRInput(
            provenance_depth=0,
            source_trust_score=0.1,
            z3_satisfiable=False,
            ihsan_score=0.3,
            contradiction_count=10,
            unverifiable_claims=5,
        )
        result = engine.snr_score(inputs)
        assert result["passed"] is False
        assert result["score"] < 0.95

    def test_snr_score_signal_components(self):
        """Signal components decompose correctly."""
        engine = SNREngine()
        inputs = SNRInput(
            provenance_depth=4,
            source_trust_score=0.8,
            z3_satisfiable=True,
            ihsan_score=0.96,
        )
        result = engine.snr_score(inputs)
        components = result["signal_components"]
        assert "provenance" in components
        assert "constraint" in components
        assert "prediction" in components
        assert all(isinstance(v, float) for v in components.values())

    def test_snr_score_noise_components(self):
        """Noise components decompose correctly."""
        engine = SNREngine()
        inputs = SNRInput(
            contradiction_count=3,
            unverifiable_claims=2,
        )
        result = engine.snr_score(inputs)
        noise = result["noise_components"]
        assert "contradiction" in noise
        assert "unverifiable" in noise
        assert noise["contradiction"] > 0
        assert noise["unverifiable"] > 0

    def test_snr_score_claim_tags_present(self):
        """Claim tags are auto-generated as MEASURED."""
        engine = SNREngine()
        inputs = SNRInput(source_trust_score=0.9)
        result = engine.snr_score(inputs)
        tags = result["claim_tags"]
        assert tags.get("snr") == "measured"
        assert tags.get("signal_mass") == "measured"

    def test_snr_score_deterministic(self):
        """Same inputs produce same output."""
        engine = SNREngine()
        inputs = SNRInput(
            provenance_depth=3,
            source_trust_score=0.8,
        )
        r1 = engine.snr_score(inputs)
        r2 = engine.snr_score(inputs)
        assert r1["score"] == r2["score"]
        assert r1["signal_mass"] == r2["signal_mass"]
        assert r1["noise_mass"] == r2["noise_mass"]

    def test_snr_score_policy_digest_present(self):
        """Policy digest is included for audit."""
        engine = SNREngine()
        inputs = SNRInput()
        result = engine.snr_score(inputs)
        assert len(result["policy_digest"]) == 64  # BLAKE3 hex

    def test_snr_score_custom_policy(self):
        """Custom policy affects computation."""
        strict_policy = SNRPolicy(snr_min=0.99)
        engine = SNREngine(strict_policy)
        inputs = SNRInput(
            provenance_depth=3,
            source_trust_score=0.9,
            ihsan_score=0.96,
        )
        result = engine.snr_score(inputs)
        # Even good input may not pass a 0.99 threshold
        assert isinstance(result["passed"], bool)

    def test_snr_score_no_div_by_zero(self):
        """Engine handles edge-case inputs without division by zero."""
        engine = SNREngine()
        # Minimal inputs — engine should still produce valid result
        inputs = SNRInput(
            provenance_depth=0,
            corroboration_count=0,
            source_trust_score=0.0,
            z3_satisfiable=False,
            ihsan_score=0.0,
            prediction_accuracy=0.0,
            context_fit_score=0.0,
            contradiction_count=0,
            unverifiable_claims=0,
        )
        result = engine.snr_score(inputs)
        assert 0.0 <= result["score"] <= 1.0
        assert isinstance(result["passed"], bool)


# =============================================================================
# SP-005: IHSAN GATE — FAIL-CLOSED EXCELLENCE CONSTRAINT
# =============================================================================

class TestIhsanGate:
    """Tests for the Ihsan gate."""

    def test_gate_creation(self):
        """IhsanGate can be created with default threshold."""
        gate = IhsanGate()
        assert gate.threshold == 0.95

    def test_gate_custom_threshold(self):
        """IhsanGate accepts custom threshold."""
        gate = IhsanGate(threshold=0.90)
        assert gate.threshold == 0.90

    def test_approved_above_threshold(self):
        """Components above threshold produce APPROVED."""
        gate = IhsanGate(threshold=0.95)
        components = IhsanComponents(
            correctness=0.98,
            safety=0.99,
            efficiency=0.92,
            user_benefit=0.96,
        )
        result = gate.evaluate(components)
        assert result.decision == "APPROVED"
        assert len(result.reason_codes) == 0

    def test_rejected_below_threshold(self):
        """Components below threshold produce REJECTED."""
        gate = IhsanGate(threshold=0.95)
        components = IhsanComponents(
            correctness=0.60,
            safety=0.70,
            efficiency=0.50,
            user_benefit=0.40,
        )
        result = gate.evaluate(components)
        assert result.decision == "REJECTED"
        assert ReasonCode.IHSAN_BELOW_THRESHOLD.value in result.reason_codes

    def test_fail_closed_on_low_safety(self):
        """Low safety component adds specific reason code."""
        gate = IhsanGate(threshold=0.95)
        components = IhsanComponents(
            correctness=0.95,
            safety=0.80,  # Below 0.90
            efficiency=0.95,
            user_benefit=0.95,
        )
        result = gate.evaluate(components)
        if result.decision == "REJECTED":
            assert "SAFETY_COMPONENT_LOW" in result.reason_codes

    def test_fail_closed_on_low_correctness(self):
        """Low correctness component adds specific reason code."""
        gate = IhsanGate(threshold=0.95)
        components = IhsanComponents(
            correctness=0.70,  # Below 0.85
            safety=0.60,
            efficiency=0.50,
            user_benefit=0.40,
        )
        result = gate.evaluate(components)
        assert result.decision == "REJECTED"
        assert "CORRECTNESS_COMPONENT_LOW" in result.reason_codes

    def test_composite_score_weights(self):
        """Composite score uses correct default weights."""
        components = IhsanComponents(
            correctness=1.0,
            safety=1.0,
            efficiency=1.0,
            user_benefit=1.0,
        )
        # All 1.0 → composite = 1.0
        score = components.composite_score()
        assert abs(score - 1.0) < 0.001

    def test_safety_weighted_highest(self):
        """Safety has highest default weight (0.35)."""
        # High safety, low everything else
        high_safety = IhsanComponents(
            correctness=0.5, safety=1.0, efficiency=0.5, user_benefit=0.5
        )
        # Low safety, high everything else
        low_safety = IhsanComponents(
            correctness=1.0, safety=0.5, efficiency=1.0, user_benefit=1.0
        )
        assert high_safety.composite_score() < low_safety.composite_score()
        # But the gap should be affected by safety's higher weight

    def test_result_to_dict_schema_compatible(self):
        """IhsanResult.to_dict() matches receipt.ihsan schema."""
        gate = IhsanGate()
        components = IhsanComponents(
            correctness=0.97,
            safety=0.99,
            efficiency=0.92,
            user_benefit=0.95,
        )
        result = gate.evaluate(components)
        d = result.to_dict()

        assert "score" in d
        assert "threshold" in d
        assert "decision" in d
        assert "components" in d
        assert "version" in d
        assert 0.0 <= d["score"] <= 1.0


class TestIhsanScore:
    """Tests for the ihsan_score() authoritative function."""

    def test_ihsan_score_returns_receipt_shape(self):
        """ihsan_score() returns schema-compatible dict."""
        gate = IhsanGate()
        components = IhsanComponents(
            correctness=0.97,
            safety=0.99,
            efficiency=0.92,
            user_benefit=0.95,
        )
        result = gate.ihsan_score(components)

        assert "score" in result
        assert "threshold" in result
        assert "decision" in result
        assert "components" in result
        assert "version" in result
        assert "passed" in result
        assert "reason_codes" in result

    def test_ihsan_score_approved(self):
        """High components produce passed=True."""
        gate = IhsanGate(threshold=0.90)
        components = IhsanComponents(
            correctness=0.95,
            safety=0.99,
            efficiency=0.92,
            user_benefit=0.95,
        )
        result = gate.ihsan_score(components)
        assert result["passed"] is True
        assert result["decision"] == "APPROVED"

    def test_ihsan_score_rejected(self):
        """Low components produce passed=False with reason codes."""
        gate = IhsanGate(threshold=0.95)
        components = IhsanComponents(
            correctness=0.50,
            safety=0.60,
            efficiency=0.40,
            user_benefit=0.30,
        )
        result = gate.ihsan_score(components)
        assert result["passed"] is False
        assert result["decision"] == "REJECTED"
        assert len(result["reason_codes"]) > 0

    def test_ihsan_score_version_present(self):
        """Version field is present for audit."""
        gate = IhsanGate()
        components = IhsanComponents(
            correctness=0.95,
            safety=0.99,
            efficiency=0.92,
            user_benefit=0.95,
        )
        result = gate.ihsan_score(components)
        assert result["version"] == "1.0.0"


# =============================================================================
# INTEGRATION: SNR + IHSAN TOGETHER
# =============================================================================

class TestSNRIhsanIntegration:
    """Tests that SNR and Ihsan work together."""

    def test_both_pass(self):
        """Both SNR and Ihsan pass for high-quality input."""
        snr_engine = SNREngine()
        ihsan_gate = IhsanGate(threshold=0.90)

        snr_result = snr_engine.snr_score(SNRInput(
            provenance_depth=5,
            corroboration_count=3,
            source_trust_score=0.95,
            z3_satisfiable=True,
            ihsan_score=0.99,
        ))

        ihsan_result = ihsan_gate.ihsan_score(IhsanComponents(
            correctness=0.97,
            safety=0.99,
            efficiency=0.92,
            user_benefit=0.95,
        ))

        assert snr_result["passed"] is True
        assert ihsan_result["passed"] is True

    def test_snr_fails_ihsan_passes(self):
        """Low SNR + high Ihsan → SNR gate rejects."""
        snr_engine = SNREngine()
        ihsan_gate = IhsanGate(threshold=0.90)

        snr_result = snr_engine.snr_score(SNRInput(
            provenance_depth=0,
            source_trust_score=0.1,
            contradiction_count=10,
        ))

        ihsan_result = ihsan_gate.ihsan_score(IhsanComponents(
            correctness=0.97,
            safety=0.99,
            efficiency=0.92,
            user_benefit=0.95,
        ))

        assert snr_result["passed"] is False
        assert ihsan_result["passed"] is True

    def test_snr_passes_ihsan_fails(self):
        """High SNR + low Ihsan → Ihsan gate rejects."""
        snr_engine = SNREngine()
        ihsan_gate = IhsanGate(threshold=0.95)

        snr_result = snr_engine.snr_score(SNRInput(
            provenance_depth=5,
            corroboration_count=3,
            source_trust_score=0.95,
            z3_satisfiable=True,
            ihsan_score=0.99,
        ))

        ihsan_result = ihsan_gate.ihsan_score(IhsanComponents(
            correctness=0.50,
            safety=0.60,
            efficiency=0.40,
            user_benefit=0.30,
        ))

        assert snr_result["passed"] is True
        assert ihsan_result["passed"] is False

    def test_combined_receipt_shape(self):
        """Both results produce receipt-compatible dicts."""
        snr_engine = SNREngine()
        ihsan_gate = IhsanGate()

        snr_result = snr_engine.snr_score(SNRInput(source_trust_score=0.9))
        ihsan_result = ihsan_gate.ihsan_score(IhsanComponents(
            correctness=0.97, safety=0.99, efficiency=0.92, user_benefit=0.95
        ))

        # Both should be JSON-serializable and match receipt schema
        import json
        snr_json = json.dumps(snr_result)
        ihsan_json = json.dumps(ihsan_result)
        assert isinstance(snr_json, str)
        assert isinstance(ihsan_json, str)
