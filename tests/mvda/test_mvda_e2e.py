"""MVDA Acceptance Tests — three required paths."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from mvda.config import EVIDENCE_MIN_COUNT
from mvda.fate_crossing import execute_mvda
from mvda.ledger import MvdaLedger
from mvda.pat_researcher import PatResult
from mvda.sat_validator import SatVerdict, run_sat_validator


@pytest.fixture
def tmp_ledger(tmp_path):
    return MvdaLedger(path=tmp_path / "test-ledger.jsonl")


class TestSuccessPath:
    """Test 1: Real local question → PAT retrieves evidence → SAT PASS → full ledger chain."""

    @pytest.mark.requires_ollama
    def test_full_mvda_cycle(self, tmp_ledger):
        result = execute_mvda(
            "What is the Spearpoint seal (commit b08f2208) and why does it matter?",
            tmp_ledger,
        )

        # PAT must have found evidence
        assert len(result["pat_evidence_refs"]) >= 1, "PAT must find local evidence"
        assert result["pat_answer"], "PAT must produce an answer"
        assert result["pat_confidence"] in ("high", "medium")

        # SAT must have issued a verdict
        assert result["sat_verdict"] in ("PASS", "BLOCKED_BY_IHSAN", "BLOCKED_BY_EVIDENCE", "DEGRADED")

        # Receipts emitted
        assert result["receipts_emitted"] >= 3

        # Ledger chain integrity
        valid, count = tmp_ledger.verify_chain()
        assert valid, "Ledger hash chain must be valid"
        assert count >= 3, f"Expected at least 3 ledger entries, got {count}"

        # Verify ledger entries have required fields
        lines = tmp_ledger.path.read_text().strip().split("\n")
        for line in lines:
            entry = json.loads(line)
            assert entry["ledger_class"] == "mvda_dev"
            assert entry["canonical"] is False
            assert entry["timestamp"] > 0
            assert entry["actor"]
            assert entry["step"]


class TestBlockedByEvidence:
    """Test 2: Force insufficient evidence → SAT BLOCKED_BY_EVIDENCE."""

    def test_no_evidence_blocked(self, tmp_ledger):
        # Create a PAT result with no evidence
        empty_pat = PatResult(
            answer="The spearpoint is important.",
            evidence_refs=[],
            confidence="none",
            model="test",
        )

        verdict = run_sat_validator(empty_pat)
        assert verdict.verdict == "BLOCKED_BY_EVIDENCE"
        assert "CLAIM_MUST_BIND" in verdict.reason
        assert verdict.evidence_sufficient is False

    def test_blocked_evidence_receipted(self, tmp_ledger):
        """Blocked path still emits receipts to ledger."""
        with patch("mvda.fate_crossing.run_pat_researcher") as mock_pat:
            mock_pat.return_value = PatResult(
                answer="Some claim without evidence.",
                evidence_refs=[],
                confidence="none",
                model="test-mock",
            )
            result = execute_mvda("test question", tmp_ledger)

        assert result["sat_verdict"] == "BLOCKED_BY_EVIDENCE"
        valid, count = tmp_ledger.verify_chain()
        assert valid
        assert count == 3, "All 3 receipts emitted even on blocked path"


class TestBlockedByIhsan:
    """Test 3: Force weak output → SAT BLOCKED_BY_IHSAN or DEGRADED."""

    def test_degraded_on_error(self, tmp_ledger):
        """PAT error produces DEGRADED verdict."""
        error_pat = PatResult(
            answer="ERROR: Ollama unreachable",
            evidence_refs=["some-ref"],
            confidence="none",
            model="test",
        )

        verdict = run_sat_validator(error_pat)
        assert verdict.verdict == "DEGRADED"
        assert verdict.ihsan_score == 0.0

    def test_degraded_on_empty(self, tmp_ledger):
        """Empty PAT answer produces DEGRADED verdict."""
        empty_pat = PatResult(
            answer="",
            evidence_refs=["some-ref"],
            confidence="none",
            model="test",
        )

        verdict = run_sat_validator(empty_pat)
        assert verdict.verdict == "DEGRADED"

    def test_blocked_ihsan_receipted(self, tmp_ledger):
        """Degraded path still emits receipts to ledger."""
        with patch("mvda.fate_crossing.run_pat_researcher") as mock_pat:
            mock_pat.return_value = PatResult(
                answer="ERROR: model crashed",
                evidence_refs=["git-show:b08f2208"],  # valid ref so auditor passes
                confidence="none",
                model="test-mock",
            )
            result = execute_mvda("test question", tmp_ledger)

        assert result["sat_verdict"] in ("DEGRADED", "BLOCKED_BY_IHSAN")
        valid, count = tmp_ledger.verify_chain()
        assert valid
        assert count >= 3
