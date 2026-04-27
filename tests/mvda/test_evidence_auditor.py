"""MVDA v0.3 — Evidence Auditor tests."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from mvda.evidence_auditor import audit_evidence, EvidenceAuditResult
from mvda.fate_crossing import execute_mvda
from mvda.ledger import MvdaLedger
from mvda.pat_researcher import PatResult


class TestAuditorDirect:
    """Direct evidence auditor unit tests."""

    def test_valid_git_refs(self):
        result = audit_evidence([
            "git-show:b08f2208",
            "git-merge-base:ancestry-check",
        ])
        assert result.valid_count == 2
        assert result.invalid_count == 0
        assert result.all_refs_valid

    def test_valid_file_ref(self):
        result = audit_evidence(["file:core/zpk/kernel.py"])
        assert result.valid_count == 1
        assert result.all_refs_valid

    def test_invalid_file_ref(self):
        result = audit_evidence(["file:BIZRA_Constitutional_White_Paper_v7.3.pdf"])
        assert result.invalid_count == 1
        assert not result.all_refs_valid
        assert "BIZRA_Constitutional_White_Paper_v7.3.pdf" in result.invalid_refs[0]

    def test_invalid_git_ref(self):
        result = audit_evidence(["git-show:deadbeef99999999"])
        assert result.invalid_count == 1
        assert not result.all_refs_valid

    def test_mixed_refs(self):
        result = audit_evidence([
            "git-show:b08f2208",        # valid
            "file:NONEXISTENT.pdf",     # invalid
            "file:core/zpk/kernel.py",  # valid
        ])
        assert result.valid_count == 2
        assert result.invalid_count == 1
        assert not result.all_refs_valid

    def test_unknown_ref_type(self):
        result = audit_evidence(["fabricated-ref"])
        assert result.invalid_count == 1
        assert not result.all_refs_valid

    def test_empty_refs(self):
        result = audit_evidence([])
        assert result.total_count == 0
        assert not result.all_refs_valid

    @pytest.mark.requires_real_data
    def test_gold_chunk_valid(self):
        # Use a known chunk_id from the real GOLD corpus.
        # Skipped in CI: the parquet file is hundreds of MB and lives on the
        # operator's data lake, not the repo. Local devs run with the full
        # corpus mounted at $BIZRA_DATA_LAKE_ROOT/04_GOLD/.
        import os

        import pandas as pd

        gold_root = os.environ.get(
            "BIZRA_GOLD_DIR",
            os.path.join(
                os.environ.get("BIZRA_DATA_LAKE_ROOT", "/data/bizra"),
                "04_GOLD",
            ),
        )
        chunks_path = os.path.join(gold_root, "chunks.parquet")
        if not os.path.exists(chunks_path):
            pytest.skip(f"GOLD chunks.parquet not present at {chunks_path}")

        df = pd.read_parquet(chunks_path, columns=["chunk_id"])
        real_id = df.iloc[0]["chunk_id"]
        result = audit_evidence([f"04_GOLD:chunk:{real_id}"])
        assert result.valid_count == 1
        assert result.all_refs_valid

    def test_gold_chunk_invalid(self):
        result = audit_evidence(["04_GOLD:chunk:nonexistent_chunk_id_12345"])
        assert result.invalid_count == 1
        assert not result.all_refs_valid


class TestAuditorInFate:
    """Evidence auditor integrated into FATE crossing."""

    @pytest.fixture
    def tmp_ledger(self, tmp_path):
        return MvdaLedger(path=tmp_path / "test-ledger.jsonl")

    def test_fabricated_evidence_now_blocked(self, tmp_ledger):
        """The v0.2 failure case: fabricated evidence should now be blocked."""
        with patch("mvda.fate_crossing.run_pat_researcher") as mock_pat:
            mock_pat.return_value = PatResult(
                answer=(
                    "According to the BIZRA Constitutional White Paper v7.3 (Section 14.2.1), "
                    "the Spearpoint seal establishes a cryptographic anchor."
                ),
                evidence_refs=["file:BIZRA_Constitutional_White_Paper_v7.3.pdf"],
                confidence="high",
                model="test-mock",
            )
            result = execute_mvda("test fabricated evidence", tmp_ledger)

        assert result["sat_verdict"] == "BLOCKED_BY_EVIDENCE"
        assert not result["evidence_audit_valid"]
        assert "BIZRA_Constitutional_White_Paper_v7.3.pdf" in str(result["evidence_audit_invalid_refs"])

        # Verify ledger has evidence_auditor step
        lines = tmp_ledger.path.read_text().strip().split("\n")
        actors = [json.loads(l)["actor"] for l in lines]
        assert "evidence_auditor" in actors

    @pytest.mark.requires_ollama
    def test_valid_refs_pass_through(self, tmp_ledger):
        """Real evidence refs should pass auditor and reach SAT."""
        result = execute_mvda(
            "What is the Spearpoint seal (commit b08f2208)?",
            tmp_ledger,
        )
        assert result["evidence_audit_valid"] is True
        # SAT should have been reached (verdict is from SAT, not auditor)
        assert result["sat_model"] != "evidence-auditor-gate"

    def test_mixed_refs_blocked(self, tmp_ledger):
        """Mix of valid and invalid refs should block at auditor."""
        with patch("mvda.fate_crossing.run_pat_researcher") as mock_pat:
            mock_pat.return_value = PatResult(
                answer="Some answer with mixed refs.",
                evidence_refs=[
                    "git-show:b08f2208",             # valid
                    "file:DOES_NOT_EXIST.pdf",        # invalid
                ],
                confidence="high",
                model="test-mock",
            )
            result = execute_mvda("test mixed refs", tmp_ledger)

        assert result["sat_verdict"] == "BLOCKED_BY_EVIDENCE"
        assert not result["evidence_audit_valid"]
