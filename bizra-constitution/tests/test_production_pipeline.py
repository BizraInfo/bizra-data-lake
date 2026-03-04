"""Tests for BIZRA Production Pipeline — Signed Evidence & Real Identity."""

import os
import sys
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BIZRA_CONSTITUTION_PATH",
                       str(Path(__file__).parent.parent / "constitution.toml"))

from production_pipeline import ProductionPipeline, create_node0
from identity_genesis import create_identity, NodeIdentity
from ollama_provider import OllamaProvider, InferenceResult
from mission_pipeline import MissionStatus


@pytest.fixture
def identity():
    return create_identity()


@pytest.fixture
def mock_ollama():
    """OllamaProvider that doesn't need a real server."""
    provider = OllamaProvider(
        base_url="http://localhost:99999",  # Won't connect
        model_chain=["mock-model"],
    )
    return provider


@pytest.fixture
def pipeline(identity, tmp_path):
    """Production pipeline with template fallback (no real Ollama)."""
    return ProductionPipeline(
        identity=identity,
        ollama=OllamaProvider(base_url="http://localhost:99999"),
        evidence_path=tmp_path / "test_evidence.jsonl",
        cache_path=tmp_path / "test_cache.json",
    )


class TestProductionPipelineExecution:
    """End-to-end with real identity, template inference."""

    def test_mission_completes(self, pipeline):
        m = pipeline.execute("Hello world")
        assert m.status == MissionStatus.COMPLETE

    def test_mission_has_output(self, pipeline):
        m = pipeline.execute("What is AI?")
        assert len(m.output_text) > 0

    def test_evidence_receipt_is_signed(self, pipeline):
        m = pipeline.execute("Sign this evidence")
        assert m.evidence_receipt is not None
        meta = m.evidence_receipt.metadata
        assert "signature_hex" in meta
        assert "signer_public_key" in meta
        assert "signer_agent" in meta
        assert "node_id" in meta

    def test_signer_is_integrator(self, pipeline):
        m = pipeline.execute("test")
        meta = m.evidence_receipt.metadata
        assert meta["signer_agent"] == "Integrator"

    def test_node_id_in_receipt(self, pipeline):
        m = pipeline.execute("test")
        meta = m.evidence_receipt.metadata
        assert meta["node_id"] == pipeline.identity.node_id

    def test_signature_is_nonempty_hex(self, pipeline):
        m = pipeline.execute("test")
        sig_hex = m.evidence_receipt.metadata["signature_hex"]
        assert len(sig_hex) > 0
        int(sig_hex, 16)  # Valid hex

    def test_signer_public_key_matches_integrator(self, pipeline):
        m = pipeline.execute("test")
        meta = m.evidence_receipt.metadata
        integrator = pipeline.identity.get_agent("Integrator")
        assert meta["signer_public_key"] == integrator.public_key_hex


class TestSignedEvidenceChain:
    """Evidence chain with cryptographic signatures."""

    def test_chain_of_three_all_signed(self, pipeline):
        for text in ["first", "second", "third"]:
            m = pipeline.execute(text)
            assert "signature_hex" in m.evidence_receipt.metadata

    def test_chain_integrity_preserved(self, pipeline):
        pipeline.execute("a")
        pipeline.execute("b")
        pipeline.execute("c")
        valid, count, errors = pipeline.evidence_ledger.verify_chain()
        assert valid
        assert count == 3

    def test_each_receipt_has_unique_signature(self, pipeline):
        sigs = set()
        for text in ["x", "y", "z"]:
            m = pipeline.execute(text)
            sig = m.evidence_receipt.metadata["signature_hex"]
            sigs.add(sig)
        assert len(sigs) == 3


class TestHealthReport:
    """Extended health with identity and Ollama."""

    def test_health_has_node_id(self, pipeline):
        h = pipeline.health()
        assert "node_id" in h
        assert len(h["node_id"]) == 64

    def test_health_has_public_key(self, pipeline):
        h = pipeline.health()
        assert "public_key" in h

    def test_health_has_agent_count(self, pipeline):
        h = pipeline.health()
        assert h["total_agents"] == 12

    def test_health_has_ollama_section(self, pipeline):
        h = pipeline.health()
        assert "ollama" in h
        assert "model_chain" in h["ollama"]


class TestNodeFactory:
    """create_node0 factory function."""

    def test_creates_pipeline(self, tmp_path):
        node = create_node0(
            data_dir=tmp_path / "node0",
            ollama_url="http://localhost:99999",
        )
        assert isinstance(node, ProductionPipeline)

    def test_creates_identity_file(self, tmp_path):
        data_dir = tmp_path / "node0"
        create_node0(data_dir=data_dir, ollama_url="http://localhost:99999")
        assert (data_dir / "identity.json").exists()

    def test_identity_file_has_node_id(self, tmp_path):
        data_dir = tmp_path / "node0"
        node = create_node0(data_dir=data_dir, ollama_url="http://localhost:99999")
        identity_data = json.loads((data_dir / "identity.json").read_text())
        assert identity_data["node_id"] == node.identity.node_id

    def test_pipeline_executes_missions(self, tmp_path):
        node = create_node0(
            data_dir=tmp_path / "node0",
            ollama_url="http://localhost:99999",
        )
        m = node.execute("factory test")
        assert m.status == MissionStatus.COMPLETE

    def test_custom_model_chain(self, tmp_path):
        node = create_node0(
            data_dir=tmp_path / "node0",
            ollama_url="http://localhost:99999",
            model_chain=["custom-model-a", "custom-model-b"],
        )
        assert node.ollama.model_chain == ["custom-model-a", "custom-model-b"]


class TestOllamaIntegration:
    """Test with mocked Ollama responses."""

    @patch("ollama_provider.urllib.request.urlopen")
    def test_real_llm_output_flows_through(self, mock_urlopen, identity, tmp_path):
        body = json.dumps({
            "response": "LLM says: distributed AI is the future",
            "eval_count": 8,
            "eval_duration": 200_000_000,
        }).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        pipeline = ProductionPipeline(
            identity=identity,
            ollama=OllamaProvider(model_chain=["test-model"]),
            evidence_path=tmp_path / "llm_evidence.jsonl",
        )

        m = pipeline.execute("What is distributed AI?")
        assert m.status == MissionStatus.COMPLETE
        assert "distributed AI" in m.output_text
        assert m.evidence_receipt is not None
        assert "signature_hex" in m.evidence_receipt.metadata
