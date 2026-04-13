"""Tests for Canonical Loop Proof."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from core.proof_engine.loop_proof import LoopProof, LoopStep, execute_loop_proof


def _mock_sat_pass():
    body = json.dumps({"message": {"content": json.dumps({
        "verdict": "PASS", "reason": "good", "ihsan_score": 0.98, "evidence_sufficient": True,
    })}}).encode()
    mock = MagicMock()
    mock.read.return_value = body
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=False)
    return mock


class TestLoopProofStructure:
    def test_empty_proof_has_genesis(self):
        proof = LoopProof(mission="test")
        assert proof.genesis_hash == "0" * 64
        assert proof.steps == []

    def test_add_step_chains(self):
        proof = LoopProof(mission="test")
        s1 = proof.add_step("a", "do_x", "ok")
        s2 = proof.add_step("b", "do_y", "ok")
        assert s1.hash != s2.hash
        assert s1.seq == 0
        assert s2.seq == 1

    def test_chain_verification(self):
        proof = LoopProof(mission="test")
        proof.add_step("a", "x", "ok")
        proof.add_step("b", "y", "ok")
        proof.add_step("c", "z", "ok")
        assert proof.verify_chain()

    def test_tampered_chain_fails(self):
        proof = LoopProof(mission="test")
        proof.add_step("a", "x", "ok")
        proof.add_step("b", "y", "ok")
        proof.steps[0].hash = "tampered"
        assert not proof.verify_chain()

    def test_manifest_hash_deterministic(self):
        proof = LoopProof(mission="test", node_id="n0")
        proof.add_step("a", "x", "ok")
        h1 = proof.compute_manifest_hash()
        h2 = proof.compute_manifest_hash()
        assert h1 == h2
        assert len(h1) == 64

    def test_to_dict_shape(self):
        proof = LoopProof(mission="test")
        proof.add_step("a", "x", "ok")
        proof.compute_manifest_hash()
        d = proof.to_dict()
        assert d["proof_class"] == "node0_loop_proof"
        assert d["canonical"] is False
        assert d["step_count"] == 1
        assert d["chain_valid"] is True
        assert "manifest_hash" in d

    def test_to_json_parseable(self):
        proof = LoopProof(mission="test")
        proof.add_step("a", "x", "ok")
        proof.compute_manifest_hash()
        parsed = json.loads(proof.to_json())
        assert parsed["mission"] == "test"


class TestExecuteLoopProof:
    @patch("core.proof_engine.sat_validator.urllib.request.urlopen")
    def test_full_execution_pass(self, mock_urlopen, tmp_path):
        mock_urlopen.return_value = _mock_sat_pass()
        output = tmp_path / "proof.json"

        proof = execute_loop_proof(
            mission="Test mission",
            pat_answer="The spearpoint seal is commit b08f2208.",
            evidence_refs=["git-show:b08f2208"],
            output_path=output,
        )

        assert proof.verify_chain()
        assert len(proof.steps) >= 5
        assert proof.manifest_hash
        assert output.exists()

        # Verify artifact is valid JSON
        data = json.loads(output.read_text())
        assert data["proof_class"] == "node0_loop_proof"
        assert data["chain_valid"] is True

    def test_blocked_execution(self, tmp_path):
        output = tmp_path / "proof.json"

        proof = execute_loop_proof(
            mission="Test blocked",
            pat_answer="Some claim.",
            evidence_refs=["file:NONEXISTENT.pdf"],
            output_path=output,
        )

        assert proof.verify_chain()
        assert not proof.fate_result.get("passed")
        assert output.exists()

    def test_no_evidence_blocked(self, tmp_path):
        proof = execute_loop_proof(
            mission="Test no evidence",
            pat_answer="Claim without evidence.",
            evidence_refs=[],
            output_path=tmp_path / "proof.json",
        )

        assert proof.verify_chain()
        assert not proof.fate_result.get("passed")

    def test_routing_table_included(self, tmp_path):
        proof = execute_loop_proof(
            mission="Test routing",
            pat_answer="answer",
            evidence_refs=["git-show:b08f2208"],
            output_path=tmp_path / "proof.json",
        )
        assert "pat.researcher" in proof.routing
        assert "sat.sentinel" in proof.routing
