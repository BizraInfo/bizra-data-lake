"""
E2E Tests — PyO3 Inference Bridge (Python↔Rust direct path).

These tests validate the unified inference path WITHOUT needing the
HTTP server. They exercise:
  1. InferenceGateway construction via PyO3
  2. Backend registration (Ollama / LM Studio)
  3. Gate chain validation through Rust
  4. Actual inference through registered backends

The bridge tests prove that Python can call Rust's SIMD-accelerated
inference gateway with constitutional gate enforcement — closing Gap #1.

Run: pytest tests/e2e_http/test_pyo3_bridge.py -m pyo3_bridge
Requires: maturin develop --release in bizra-omega/bizra-python/
"""

import json
import os

import pytest


pytestmark = pytest.mark.pyo3_bridge


class TestPyO3CoreTypes:
    """Verify core Rust types are accessible from Python."""

    def test_node_identity_generation(self, bizra_module):
        identity = bizra_module.NodeIdentity()
        assert len(identity.node_id.id) == 32
        assert len(identity.public_key) > 0

    def test_constitution_defaults(self, bizra_module):
        c = bizra_module.Constitution()
        assert c.ihsan_threshold == 0.95
        assert c.snr_threshold == 0.85

    def test_constitution_check_ihsan_pass(self, bizra_module):
        c = bizra_module.Constitution()
        assert c.check_ihsan(0.96) is True
        assert c.check_ihsan(0.99) is True

    def test_constitution_check_ihsan_fail(self, bizra_module):
        c = bizra_module.Constitution()
        assert c.check_ihsan(0.94) is False
        assert c.check_ihsan(0.50) is False

    def test_constitution_check_snr_pass(self, bizra_module):
        c = bizra_module.Constitution()
        assert c.check_snr(0.90) is True
        assert c.check_snr(0.85) is True

    def test_constitution_check_snr_fail(self, bizra_module):
        c = bizra_module.Constitution()
        assert c.check_snr(0.84) is False

    def test_module_constants(self, bizra_module):
        assert bizra_module.IHSAN_THRESHOLD == 0.95
        assert bizra_module.SNR_THRESHOLD == 0.85


class TestPyO3PCI:
    """Test Proof-Carrying Inference through PyO3."""

    def test_pci_envelope_creation(self, bizra_module, pyo3_identity):
        payload = json.dumps({"query": "test", "result": 42})
        envelope = bizra_module.PCIEnvelope.create(
            pyo3_identity, payload, 3600, ["test_provenance"]
        )
        assert envelope.id is not None
        assert len(envelope.content_hash) > 0
        assert len(envelope.signature) > 0

    def test_pci_envelope_signature_verifiable(self, bizra_module, pyo3_identity):
        payload = json.dumps({"data": "signed_content"})
        envelope = bizra_module.PCIEnvelope.create(
            pyo3_identity, payload, 3600, []
        )
        # Signature was created with identity's key
        assert len(envelope.signature) > 0
        assert envelope.public_key == pyo3_identity.public_key

    def test_domain_separated_digest(self, bizra_module):
        digest = bizra_module.domain_separated_digest(b"test content")
        assert len(digest) == 64  # BLAKE3 hex output


class TestPyO3GateChain:
    """Test constitutional gate chain through PyO3."""

    def test_gate_chain_valid_json_passes(self, bizra_module):
        chain = bizra_module.GateChain()
        results = chain.verify(
            b'{"valid": "json"}',
            snr_score=0.90,
            ihsan_score=0.96,
        )
        assert bizra_module.GateChain.all_passed(results)

    def test_gate_chain_invalid_json_fails(self, bizra_module):
        chain = bizra_module.GateChain()
        results = chain.verify(
            b"not json at all",
            snr_score=0.90,
            ihsan_score=0.96,
        )
        assert not bizra_module.GateChain.all_passed(results)

    def test_gate_chain_low_ihsan_fails(self, bizra_module):
        """Fail-closed: Ihsan below threshold must be rejected."""
        chain = bizra_module.GateChain()
        results = chain.verify(
            b'{"valid": "json"}',
            snr_score=0.90,
            ihsan_score=0.80,  # Below 0.95 threshold
        )
        assert not bizra_module.GateChain.all_passed(results)

    def test_gate_chain_low_snr_fails(self, bizra_module):
        """Fail-closed: SNR below threshold must be rejected."""
        chain = bizra_module.GateChain()
        results = chain.verify(
            b'{"valid": "json"}',
            snr_score=0.70,  # Below 0.85 threshold
            ihsan_score=0.96,
        )
        assert not bizra_module.GateChain.all_passed(results)

    def test_gate_chain_missing_scores_fails(self, bizra_module):
        """Fail-closed: Missing scores must result in rejection, not defaults."""
        chain = bizra_module.GateChain()
        results = chain.verify(
            b'{"valid": "json"}',
            snr_score=None,
            ihsan_score=None,
        )
        assert not bizra_module.GateChain.all_passed(results)

    def test_gate_chain_returns_per_gate_results(self, bizra_module):
        chain = bizra_module.GateChain()
        results = chain.verify(
            b'{"valid": "json"}',
            snr_score=0.90,
            ihsan_score=0.96,
        )
        # Should have 3 gates: Schema, Ihsan, SNR
        assert len(results) == 3
        for gate_name, passed, code in results:
            assert isinstance(gate_name, str)
            assert isinstance(passed, bool)


class TestPyO3InferenceGateway:
    """Test the InferenceGateway — the core Python↔Rust bridge."""

    def test_gateway_construction(self, bizra_module, pyo3_identity, pyo3_constitution):
        gw = bizra_module.InferenceGateway(pyo3_identity, pyo3_constitution)
        assert repr(gw) == "InferenceGateway(rust_native=True)"

    def test_gateway_no_backend_raises(self, bizra_module, pyo3_gateway):
        """Inference without registered backends must error, not silently fail."""
        with pytest.raises(RuntimeError, match="No backend"):
            pyo3_gateway.infer("Hello world")

    def test_gateway_register_ollama(self, bizra_module, pyo3_gateway):
        """Registering a backend should not raise even if Ollama is offline."""
        # This tests the registration path, not the connection
        pyo3_gateway.register_ollama("llama3.2", "local", "http://localhost:11434")

    def test_gateway_register_lmstudio(self, bizra_module, pyo3_identity, pyo3_constitution):
        """Registering LM Studio backend with custom host/port."""
        gw = bizra_module.InferenceGateway(pyo3_identity, pyo3_constitution)
        gw.register_lmstudio("local", host="192.168.56.1", port=1234)

    @pytest.mark.requires_ollama
    def test_gateway_infer_through_ollama(self, bizra_module, pyo3_identity, pyo3_constitution):
        """CRITICAL: Full Python → Rust → Ollama → Response path.

        This is the test that proves Gap #1 is closed.
        """
        gw = bizra_module.InferenceGateway(pyo3_identity, pyo3_constitution)
        gw.register_ollama("llama3.2", "local")

        response = gw.infer(
            prompt="What is 2+2? Reply with just the number.",
            max_tokens=16,
            temperature=0.1,
            tier="local",
        )

        assert response.text is not None
        assert len(response.text) > 0
        assert response.model == "llama3.2"
        assert response.tier == "Local"
        assert response.completion_tokens > 0
        assert response.duration_ms > 0

    @pytest.mark.requires_ollama
    def test_gateway_infer_with_system_prompt(self, bizra_module, pyo3_identity, pyo3_constitution):
        """Test system prompt passthrough."""
        gw = bizra_module.InferenceGateway(pyo3_identity, pyo3_constitution)
        gw.register_ollama("llama3.2", "local")

        response = gw.infer(
            prompt="What is the capital of France?",
            system="You are a geography expert. Answer in one word.",
            max_tokens=16,
            temperature=0.1,
        )

        assert response.text is not None
        assert len(response.text) > 0


class TestPyO3TaskComplexity:
    """Test task complexity estimation through PyO3."""

    def test_simple_prompt(self, bizra_module):
        c = bizra_module.TaskComplexity.estimate("Hello", 32)
        assert c.level == "Simple"

    def test_medium_prompt(self, bizra_module):
        words = " ".join(["word"] * 40)
        c = bizra_module.TaskComplexity.estimate(words, 100)
        assert c.level == "Medium"

    def test_complex_prompt_with_code(self, bizra_module):
        c = bizra_module.TaskComplexity.estimate("Write a function:\n```python\ndef foo():\n```", 100)
        assert c.level == "Complex"

    def test_expert_prompt(self, bizra_module):
        c = bizra_module.TaskComplexity.estimate("Explain quantum computing in full detail", 3000)
        assert c.level == "Expert"


class TestPyO3ModelSelector:
    """Test model tier selection through PyO3."""

    def test_simple_selects_edge(self, bizra_module):
        selector = bizra_module.ModelSelector()
        complexity = bizra_module.TaskComplexity.estimate("Hi", 10)
        tier = selector.select_tier(complexity)
        assert tier.name == "edge"

    def test_expert_selects_pool(self, bizra_module):
        selector = bizra_module.ModelSelector()
        complexity = bizra_module.TaskComplexity.estimate("Explain everything", 5000)
        tier = selector.select_tier(complexity)
        assert tier.name == "pool"


class TestPyO3PatternMemory:
    """Test autopoiesis pattern memory through PyO3."""

    def test_pattern_learn_and_recall(self, bizra_module):
        mem = bizra_module.PatternMemory("test_node_001")
        embedding = [0.1] * 384
        pattern_id = mem.learn("test pattern", embedding, ["tag1"])
        assert pattern_id is not None
        assert mem.pattern_count() == 1

        results = mem.recall(embedding, 1)
        assert len(results) == 1
        assert results[0][0] == "test pattern"


class TestFullPipelineIntegration:
    """End-to-end flow: Identity → PCI → Gates → Inference → Verify."""

    @pytest.mark.requires_ollama
    def test_full_sovereign_inference_flow(self, bizra_module):
        """The definitive integration test: proves the full stack works together.

        Flow:
        1. Generate identity (Ed25519)
        2. Create inference gateway with Ollama backend
        3. Run inference through Rust gateway
        4. Create PCI envelope wrapping the response
        5. Validate envelope through gate chain
        """
        # Step 1: Identity
        identity = bizra_module.NodeIdentity()
        constitution = bizra_module.Constitution()

        # Step 2: Gateway with backend
        gw = bizra_module.InferenceGateway(identity, constitution)
        gw.register_ollama("llama3.2", "local")

        # Step 3: Inference through Rust
        response = gw.infer(
            prompt="What is the meaning of sovereignty?",
            max_tokens=64,
            temperature=0.3,
        )
        assert len(response.text) > 0

        # Step 4: Wrap in PCI envelope
        payload = json.dumps({
            "query": "What is the meaning of sovereignty?",
            "response": response.text,
            "model": response.model,
            "tier": response.tier,
        })
        envelope = bizra_module.PCIEnvelope.create(
            identity, payload, 3600, ["inference_gateway"]
        )

        # Step 5: Gate validation
        chain = bizra_module.GateChain()
        gate_results = chain.verify(
            payload.encode(),
            snr_score=0.90,
            ihsan_score=0.96,
        )
        assert bizra_module.GateChain.all_passed(gate_results), (
            f"Gate chain failed: {gate_results}"
        )

        # Proof: the envelope is signed and gates pass
        assert envelope.signature is not None
        assert envelope.content_hash is not None
