"""Tests for promoted Evidence Auditor in core.proof_engine."""

import pytest

from core.proof_engine.evidence_audit import (
    EvidenceAuditResult,
    RefAudit,
    audit_evidence,
    _classify_ref,
    _extract_value,
)


class TestRefClassification:
    def test_git_log(self):
        assert _classify_ref("git-log:spearpoint") == "git-log"

    def test_git_show(self):
        assert _classify_ref("git-show:b08f2208") == "git-show"

    def test_file(self):
        assert _classify_ref("file:core/zpk/kernel.py") == "file"

    def test_gold_chunk(self):
        assert _classify_ref("04_GOLD:chunk:abc123") == "gold-chunk"

    def test_gold_doc(self):
        assert _classify_ref("04_GOLD:doc:xyz789") == "gold-doc"

    def test_unknown(self):
        assert _classify_ref("fabricated-ref") == "unknown"


class TestValueExtraction:
    def test_simple(self):
        assert _extract_value("git-show:b08f2208", "git-show") == "b08f2208"

    def test_gold_chunk(self):
        assert _extract_value("04_GOLD:chunk:abc123", "gold-chunk") == "abc123"

    def test_gold_doc(self):
        assert _extract_value("04_GOLD:doc:xyz789", "gold-doc") == "xyz789"


class TestAuditEvidence:
    def test_valid_git_refs(self):
        result = audit_evidence(["git-show:b08f2208", "git-merge-base:ancestry-check"])
        assert result.valid_count == 2
        assert result.all_refs_valid

    def test_valid_file_ref(self):
        result = audit_evidence(["file:core/zpk/kernel.py"])
        assert result.valid_count == 1
        assert result.all_refs_valid

    def test_invalid_file_ref(self):
        result = audit_evidence(["file:NONEXISTENT_DOCUMENT.pdf"])
        assert result.invalid_count == 1
        assert not result.all_refs_valid

    def test_invalid_git_ref(self):
        result = audit_evidence(["git-show:deadbeef99999999"])
        assert result.invalid_count == 1
        assert not result.all_refs_valid

    def test_mixed_refs(self):
        result = audit_evidence([
            "git-show:b08f2208",
            "file:NONEXISTENT.pdf",
            "file:core/zpk/kernel.py",
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

    def test_gold_chunk_valid(self):
        import pandas as pd
        df = pd.read_parquet("/data/bizra/04_GOLD/chunks.parquet", columns=["chunk_id"])
        real_id = df.iloc[0]["chunk_id"]
        result = audit_evidence([f"04_GOLD:chunk:{real_id}"])
        assert result.valid_count == 1
        assert result.all_refs_valid

    def test_gold_chunk_invalid(self):
        result = audit_evidence(["04_GOLD:chunk:nonexistent_id_99999"])
        assert result.invalid_count == 1
        assert not result.all_refs_valid

    def test_result_contains_audit_details(self):
        result = audit_evidence(["git-show:b08f2208"])
        assert len(result.ref_audits) == 1
        assert result.ref_audits[0].ref_type == "git-show"
        assert result.ref_audits[0].valid
