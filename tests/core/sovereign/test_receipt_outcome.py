"""
Tests for Receipt Outcome — Canonical decision/status computation
=================================================================
Covers all decision paths: APPROVED, REJECTED, QUARANTINED, and combined.

Standing on Giants: Shannon (1948) — SNR gating determines decision.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.sovereign.receipt_outcome import (
    _SNR_QUARANTINE_THRESHOLD,
    receipt_outcome,
)


class TestReceiptOutcomeApproved:
    """APPROVED path: validation_passed=True AND snr_score >= threshold."""

    def test_clean_approval(self):
        result = SimpleNamespace(validation_passed=True, snr_score=0.95)
        decision, status, codes = receipt_outcome(result)
        assert decision == "APPROVED"
        assert status == "accepted"
        assert codes == []

    def test_boundary_snr_exactly_at_threshold(self):
        result = SimpleNamespace(
            validation_passed=True, snr_score=_SNR_QUARANTINE_THRESHOLD
        )
        decision, status, codes = receipt_outcome(result)
        assert decision == "APPROVED"
        assert status == "accepted"
        assert codes == []

    def test_high_snr_approval(self):
        result = SimpleNamespace(validation_passed=True, snr_score=1.0)
        decision, status, codes = receipt_outcome(result)
        assert decision == "APPROVED"
        assert status == "accepted"


class TestReceiptOutcomeRejected:
    """REJECTED path: validation_passed=False, SNR above threshold."""

    def test_ihsan_rejection(self):
        result = SimpleNamespace(validation_passed=False, snr_score=0.92)
        decision, status, codes = receipt_outcome(result)
        assert decision == "REJECTED"
        assert status == "rejected"
        assert "IHSAN_BELOW_THRESHOLD" in codes

    def test_ihsan_rejection_no_snr_code(self):
        """When SNR is fine, only IHSAN code appears."""
        result = SimpleNamespace(validation_passed=False, snr_score=0.90)
        decision, status, codes = receipt_outcome(result)
        assert "SNR_BELOW_THRESHOLD" not in codes


class TestReceiptOutcomeQuarantined:
    """QUARANTINED path: validation_passed=True BUT snr_score < threshold."""

    def test_low_snr_quarantine(self):
        result = SimpleNamespace(validation_passed=True, snr_score=0.50)
        decision, status, codes = receipt_outcome(result)
        assert decision == "QUARANTINED"
        assert status == "quarantined"
        assert "SNR_BELOW_THRESHOLD" in codes

    def test_snr_just_below_threshold(self):
        result = SimpleNamespace(
            validation_passed=True,
            snr_score=_SNR_QUARANTINE_THRESHOLD - 0.001,
        )
        decision, status, codes = receipt_outcome(result)
        assert decision == "QUARANTINED"
        assert status == "quarantined"

    def test_zero_snr_quarantine(self):
        result = SimpleNamespace(validation_passed=True, snr_score=0.0)
        decision, status, codes = receipt_outcome(result)
        assert decision == "QUARANTINED"
        assert "SNR_BELOW_THRESHOLD" in codes


class TestReceiptOutcomeCombined:
    """Combined REJECTED + SNR_BELOW_THRESHOLD: both gates fail."""

    def test_rejected_plus_low_snr(self):
        result = SimpleNamespace(validation_passed=False, snr_score=0.40)
        decision, status, codes = receipt_outcome(result)
        assert decision == "REJECTED"
        assert status == "rejected"
        assert "IHSAN_BELOW_THRESHOLD" in codes
        assert "SNR_BELOW_THRESHOLD" in codes
        assert len(codes) == 2

    def test_rejected_zero_snr(self):
        result = SimpleNamespace(validation_passed=False, snr_score=0.0)
        decision, status, codes = receipt_outcome(result)
        assert decision == "REJECTED"
        assert len(codes) == 2


class TestReceiptOutcomeDefensiveAccess:
    """Defensive getattr() fallback for missing attributes."""

    def test_missing_validation_passed_defaults_rejected(self):
        result = SimpleNamespace(snr_score=0.95)
        decision, status, codes = receipt_outcome(result)
        assert decision == "REJECTED"
        assert "IHSAN_BELOW_THRESHOLD" in codes

    def test_missing_snr_score_defaults_zero(self):
        result = SimpleNamespace(validation_passed=True)
        decision, status, codes = receipt_outcome(result)
        assert decision == "QUARANTINED"
        assert "SNR_BELOW_THRESHOLD" in codes

    def test_empty_object(self):
        result = SimpleNamespace()
        decision, status, codes = receipt_outcome(result)
        assert decision == "REJECTED"
        assert "IHSAN_BELOW_THRESHOLD" in codes
        assert "SNR_BELOW_THRESHOLD" in codes
