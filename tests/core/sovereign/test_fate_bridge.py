"""Tests for FATE Bridge — SovereignRuntime integration adapter."""

from unittest.mock import patch

from core.sovereign.fate_bridge import run_fate_bridge


class TestFateBridgeSkip:
    """FATE bridge should skip when no evidence refs provided."""

    def test_no_refs_skipped(self):
        result = run_fate_bridge(answer="Hello world", evidence_refs=None)
        assert not result.enabled
        assert result.passed

    def test_empty_refs_skipped(self):
        result = run_fate_bridge(answer="Hello world", evidence_refs=[])
        assert not result.enabled
        assert result.passed


class TestFateBridgeWithEvidence:
    """FATE bridge with real evidence refs."""

    def test_valid_evidence_passes(self):
        result = run_fate_bridge(
            answer="The Spearpoint seal is commit b08f2208.",
            evidence_refs=["git-show:b08f2208"],
        )
        assert result.enabled
        assert result.evidence_valid
        # SAT verdict depends on live model — may be PASS or DEGRADED
        # but evidence_valid should be True regardless
        assert result.evidence_valid

    def test_invalid_evidence_blocked(self):
        result = run_fate_bridge(
            answer="According to the fake document...",
            evidence_refs=["file:DOES_NOT_EXIST.pdf"],
        )
        assert result.enabled
        assert not result.passed
        assert result.verdict == "BLOCKED_BY_EVIDENCE"
        assert not result.evidence_valid
        assert result.short_circuited

    def test_mixed_evidence_blocked(self):
        result = run_fate_bridge(
            answer="Mix of real and fake.",
            evidence_refs=["git-show:b08f2208", "file:NONEXISTENT.pdf"],
        )
        assert result.enabled
        assert not result.passed
        assert result.verdict == "BLOCKED_BY_EVIDENCE"


class TestFateBridgeGracefulDegradation:
    """FATE bridge must fail-open if components are unavailable."""

    def test_execution_error_fails_open(self):
        """If FATE gate raises during execution, bridge should fail-open."""
        with patch(
            "core.proof_engine.fate_gate.validate_with_evidence",
            side_effect=RuntimeError("boom"),
        ):
            result = run_fate_bridge(
                answer="test",
                evidence_refs=["git-show:b08f2208"],
            )
        assert result.enabled
        assert result.passed  # fail-open


class TestFateBridgeResultShape:
    """Verify result structure."""

    def test_all_fields_present(self):
        result = run_fate_bridge(answer="test", evidence_refs=[])
        assert hasattr(result, "enabled")
        assert hasattr(result, "passed")
        assert hasattr(result, "verdict")
        assert hasattr(result, "reason")
        assert hasattr(result, "ihsan_score")
        assert hasattr(result, "evidence_valid")
        assert hasattr(result, "short_circuited")
