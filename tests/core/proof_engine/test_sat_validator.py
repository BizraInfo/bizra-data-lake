"""Tests for promoted SAT Validator in core.proof_engine."""

import json
import pytest
from unittest.mock import patch, MagicMock

from core.proof_engine.sat_validator import (
    SatVerdict,
    SimplePatOutput,
    validate,
    _extract_json_from_llm_response,
    VALID_VERDICTS,
)


class TestJsonExtraction:
    """Test LLM response JSON extraction (handles thinking models, markdown, etc.)."""

    def test_clean_json(self):
        raw = '{"verdict": "PASS", "reason": "good", "ihsan_score": 0.97, "evidence_sufficient": true}'
        result = _extract_json_from_llm_response(raw)
        assert result["verdict"] == "PASS"

    def test_json_in_markdown(self):
        raw = 'Some thinking\n```json\n{"verdict": "PASS", "reason": "ok", "ihsan_score": 0.95, "evidence_sufficient": true}\n```'
        result = _extract_json_from_llm_response(raw)
        assert result["verdict"] == "PASS"

    def test_json_embedded_in_text(self):
        raw = 'Let me evaluate... {"verdict": "BLOCKED_BY_IHSAN", "reason": "vague", "ihsan_score": 0.3, "evidence_sufficient": false} done.'
        result = _extract_json_from_llm_response(raw)
        assert result["verdict"] == "BLOCKED_BY_IHSAN"

    def test_empty_string(self):
        assert _extract_json_from_llm_response("") == {}

    def test_no_json(self):
        assert _extract_json_from_llm_response("just some text without json") == {}


class TestCodeGates:
    """Test pre-LLM code-level gates."""

    def test_empty_answer_degraded(self):
        pat = SimplePatOutput(answer="", evidence_refs=["ref1"])
        verdict = validate(pat)
        assert verdict.verdict == "DEGRADED"
        assert verdict.model == "code-gate"

    def test_error_answer_degraded(self):
        pat = SimplePatOutput(answer="ERROR: model crashed", evidence_refs=["ref1"])
        verdict = validate(pat)
        assert verdict.verdict == "DEGRADED"

    def test_no_evidence_blocked(self):
        pat = SimplePatOutput(answer="Some answer.", evidence_refs=[])
        verdict = validate(pat)
        assert verdict.verdict == "BLOCKED_BY_EVIDENCE"
        assert "CLAIM_MUST_BIND" in verdict.reason

    def test_verdict_has_all_fields(self):
        pat = SimplePatOutput(answer="test", evidence_refs=[])
        verdict = validate(pat)
        d = verdict.to_dict()
        assert "verdict" in d
        assert "reason" in d
        assert "ihsan_score" in d
        assert "evidence_sufficient" in d
        assert "model" in d


class TestLlmGates:
    """Test LLM-powered governance with mocked Ollama responses."""

    def _mock_ollama_response(self, verdict_json: dict):
        """Create a mock urllib response returning the given verdict."""
        response_body = json.dumps({
            "message": {"content": json.dumps(verdict_json)}
        }).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = response_body
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    @patch("core.proof_engine.sat_validator.urllib.request.urlopen")
    def test_llm_pass(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_ollama_response({
            "verdict": "PASS",
            "reason": "high quality answer",
            "ihsan_score": 0.98,
            "evidence_sufficient": True,
        })
        pat = SimplePatOutput(answer="Good answer.", evidence_refs=["ref1"])
        verdict = validate(pat)
        assert verdict.verdict == "PASS"
        assert verdict.ihsan_score == 0.98

    @patch("core.proof_engine.sat_validator.urllib.request.urlopen")
    def test_llm_blocked_ihsan(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_ollama_response({
            "verdict": "BLOCKED_BY_IHSAN",
            "reason": "vague and imprecise",
            "ihsan_score": 0.3,
            "evidence_sufficient": False,
        })
        pat = SimplePatOutput(answer="Vague answer.", evidence_refs=["ref1"])
        verdict = validate(pat)
        assert verdict.verdict == "BLOCKED_BY_IHSAN"
        assert verdict.ihsan_score == 0.3

    @patch("core.proof_engine.sat_validator.urllib.request.urlopen")
    def test_ihsan_threshold_enforced(self, mock_urlopen):
        """LLM says PASS but ihsan < threshold → override to BLOCKED_BY_IHSAN."""
        mock_urlopen.return_value = self._mock_ollama_response({
            "verdict": "PASS",
            "reason": "looks ok",
            "ihsan_score": 0.7,
            "evidence_sufficient": True,
        })
        pat = SimplePatOutput(answer="Mediocre answer.", evidence_refs=["ref1"])
        verdict = validate(pat)
        assert verdict.verdict == "BLOCKED_BY_IHSAN"

    @patch("core.proof_engine.sat_validator.urllib.request.urlopen")
    def test_invalid_verdict_becomes_degraded(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_ollama_response({
            "verdict": "MAYBE",
            "reason": "unsure",
            "ihsan_score": 0.5,
            "evidence_sufficient": True,
        })
        pat = SimplePatOutput(answer="Some answer.", evidence_refs=["ref1"])
        verdict = validate(pat)
        assert verdict.verdict == "DEGRADED"

    @patch("core.proof_engine.sat_validator.urllib.request.urlopen")
    def test_llm_unreachable_degraded(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("connection refused")
        pat = SimplePatOutput(answer="Good answer.", evidence_refs=["ref1"])
        verdict = validate(pat)
        assert verdict.verdict == "DEGRADED"
        assert "unreachable" in verdict.reason


class TestVerdictSchema:
    """Verify verdict type constraints."""

    def test_valid_verdicts_are_frozen(self):
        assert isinstance(VALID_VERDICTS, frozenset)
        assert len(VALID_VERDICTS) == 4

    def test_all_verdicts_present(self):
        assert "PASS" in VALID_VERDICTS
        assert "BLOCKED_BY_IHSAN" in VALID_VERDICTS
        assert "BLOCKED_BY_EVIDENCE" in VALID_VERDICTS
        assert "DEGRADED" in VALID_VERDICTS


import urllib.error
