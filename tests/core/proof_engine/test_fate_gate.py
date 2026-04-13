"""Tests for promoted FATE Gate in core.proof_engine."""

import json
import urllib.error
from unittest.mock import patch, MagicMock

import pytest

from core.proof_engine.fate_gate import (
    FateResult,
    validate_with_evidence,
)
from core.proof_engine.sat_validator import SimplePatOutput


def _mock_sat_response(verdict_json: dict):
    response_body = json.dumps({
        "message": {"content": json.dumps(verdict_json)}
    }).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = response_body
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestEvidenceShortCircuit:
    """FATE gate must block before reaching SAT when evidence is invalid."""

    def test_no_evidence_blocked(self):
        pat = SimplePatOutput(answer="Some claim.", evidence_refs=[])
        result = validate_with_evidence(pat)
        assert result.verdict.verdict == "BLOCKED_BY_EVIDENCE"
        assert result.short_circuited
        assert not result.passed

    def test_invalid_ref_blocked(self):
        pat = SimplePatOutput(
            answer="Citing fake document.",
            evidence_refs=["file:DOES_NOT_EXIST.pdf"],
        )
        result = validate_with_evidence(pat)
        assert result.verdict.verdict == "BLOCKED_BY_EVIDENCE"
        assert result.short_circuited
        assert result.evidence_audit.invalid_count == 1

    def test_mixed_refs_blocked(self):
        pat = SimplePatOutput(
            answer="Mix of real and fake.",
            evidence_refs=["git-show:b08f2208", "file:FAKE.pdf"],
        )
        result = validate_with_evidence(pat)
        assert result.verdict.verdict == "BLOCKED_BY_EVIDENCE"
        assert result.short_circuited
        assert result.evidence_audit.valid_count == 1
        assert result.evidence_audit.invalid_count == 1


class TestEvidencePassThroughToSat:
    """When evidence is valid, FATE must reach SAT Validator."""

    @patch("core.proof_engine.sat_validator.urllib.request.urlopen")
    def test_valid_evidence_sat_pass(self, mock_urlopen):
        mock_urlopen.return_value = _mock_sat_response({
            "verdict": "PASS",
            "reason": "excellent",
            "ihsan_score": 0.99,
            "evidence_sufficient": True,
        })
        pat = SimplePatOutput(
            answer="The Spearpoint seal is commit b08f2208.",
            evidence_refs=["git-show:b08f2208"],
        )
        result = validate_with_evidence(pat)
        assert result.verdict.verdict == "PASS"
        assert not result.short_circuited
        assert result.passed
        assert result.evidence_audit.all_refs_valid

    @patch("core.proof_engine.sat_validator.urllib.request.urlopen")
    def test_valid_evidence_sat_blocked(self, mock_urlopen):
        mock_urlopen.return_value = _mock_sat_response({
            "verdict": "BLOCKED_BY_IHSAN",
            "reason": "vague answer",
            "ihsan_score": 0.3,
            "evidence_sufficient": False,
        })
        pat = SimplePatOutput(
            answer="It's a thing.",
            evidence_refs=["git-show:b08f2208"],
        )
        result = validate_with_evidence(pat)
        assert result.verdict.verdict == "BLOCKED_BY_IHSAN"
        assert not result.short_circuited
        assert not result.passed
        assert result.evidence_audit.all_refs_valid

    @patch("core.proof_engine.sat_validator.urllib.request.urlopen")
    def test_valid_evidence_sat_degraded(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("timeout")
        pat = SimplePatOutput(
            answer="Good answer but model down.",
            evidence_refs=["git-show:b08f2208"],
        )
        result = validate_with_evidence(pat)
        assert result.verdict.verdict == "DEGRADED"
        assert not result.short_circuited


class TestFateResultShape:
    """Verify FateResult serialization and properties."""

    def test_to_dict(self):
        pat = SimplePatOutput(answer="test", evidence_refs=[])
        result = validate_with_evidence(pat)
        d = result.to_dict()
        assert "verdict" in d
        assert "evidence_audit" in d
        assert "short_circuited" in d
        assert "passed" in d

    def test_no_mvda_imports(self):
        """Verify no MVDA import dependency exists in the promoted module."""
        import core.proof_engine.fate_gate as module
        source = open(module.__file__).read()
        # Check import lines only, not docstrings/comments
        import_lines = [l for l in source.split("\n") if l.strip().startswith(("import ", "from "))]
        for line in import_lines:
            assert "mvda" not in line.lower(), f"MVDA import found: {line}"
