"""
BIZRA Sovereign LLM Integration Tests

Validates that all components work together correctly:
- Python <-> TypeScript interface consistency
- Gate chain validation across implementations
- Model registration and licensing
- Constitutional threshold enforcement

"We do not assume. We verify with formal proofs."
"""

import pytest
import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.sovereign.capability_card import (
    CapabilityCard,
    ModelTier,
    TaskType,
    CardIssuer,
    create_capability_card,
    verify_capability_card,
    IHSAN_THRESHOLD,
    SNR_THRESHOLD,
)
from core.sovereign.model_license_gate import (
    ModelLicenseGate,
    InMemoryRegistry,
    GateChain,
    create_gate_chain,
)
from core.sovereign.integration import (
    SovereignRuntime,
    SovereignConfig,
    NetworkMode,
    InferenceRequest,
    create_sovereign_runtime,
)


class TestInterfaceConsistency:
    """Tests for interface consistency across components."""

    def test_threshold_values_match(self):
        """Verify threshold values are consistent."""
        assert IHSAN_THRESHOLD == 0.95, "IHSAN_THRESHOLD must be 0.95"
        assert SNR_THRESHOLD == 0.85, "SNR_THRESHOLD must be 0.85"

    def test_model_tiers_complete(self):
        """Verify all model tiers are defined."""
        tiers = [ModelTier.EDGE, ModelTier.LOCAL, ModelTier.POOL]
        assert len(tiers) == 3
        assert ModelTier.EDGE.value == "EDGE"
        assert ModelTier.LOCAL.value == "LOCAL"
        assert ModelTier.POOL.value == "POOL"

    def test_task_types_complete(self):
        """Verify all task types are defined."""
        expected_tasks = [
            "reasoning", "chat", "summarization",
            "code_generation", "translation",
            "classification", "embedding"
        ]
        for task in expected_tasks:
            assert TaskType(task), f"TaskType.{task} should exist"

    def test_network_modes_complete(self):
        """Verify all network modes are defined."""
        modes = [
            NetworkMode.OFFLINE,
            NetworkMode.LOCAL_ONLY,
            NetworkMode.FEDERATED,
            NetworkMode.HYBRID,
        ]
        assert len(modes) == 4


class TestCapabilityCardIntegration:
    """Tests for CapabilityCard integration."""

    def test_card_creation_and_registration(self):
        """Test creating a card and registering it."""
        registry = InMemoryRegistry()
        issuer = CardIssuer()

        # Create card
        card = create_capability_card(
            model_id="integration-test-model",
            tier=ModelTier.LOCAL,
            ihsan_score=0.97,
            snr_score=0.90,
            tasks_supported=[TaskType.CHAT, TaskType.REASONING],
        )

        # Sign card
        signed_card = issuer.issue(card)
        assert signed_card.signature != ""
        assert signed_card.issuer_public_key != ""

        # Register
        registry.register(signed_card)
        assert registry.has("integration-test-model")

        # Retrieve
        retrieved = registry.get("integration-test-model")
        assert retrieved is not None
        assert retrieved.model_id == "integration-test-model"

    def test_card_serialization_roundtrip(self):
        """Test card serialization and deserialization."""
        issuer = CardIssuer()
        original = create_capability_card(
            model_id="serialization-test",
            tier=ModelTier.EDGE,
            ihsan_score=0.96,
            snr_score=0.88,
            tasks_supported=[TaskType.CHAT],
        )
        signed = issuer.issue(original)

        # Roundtrip through JSON
        json_str = signed.to_json()
        restored = CapabilityCard.from_json(json_str)

        assert restored.model_id == original.model_id
        assert restored.tier == original.tier
        assert restored.capabilities.ihsan_score == original.capabilities.ihsan_score
        assert restored.signature == signed.signature


class TestGateChainIntegration:
    """Tests for GateChain integration."""

    def test_full_gate_chain_pass(self):
        """Test output that passes all gates."""
        registry = InMemoryRegistry()
        chain = GateChain(registry)

        # Register a model
        card = create_capability_card(
            model_id="gate-test-model",
            tier=ModelTier.LOCAL,
            ihsan_score=0.97,
            snr_score=0.90,
            tasks_supported=[TaskType.CHAT],
        )
        registry.register(card)

        # Create valid output
        output = {
            "content": "This is a valid response with proper content.",
            "model_id": "gate-test-model",
            "ihsan_score": 0.97,
            "snr_score": 0.90,
        }

        result = chain.validate_output(output)
        assert result["passed"] is True
        assert result["gate_name"] == "ALL"

    def test_gate_chain_detailed_results(self):
        """Test detailed gate chain results."""
        registry = InMemoryRegistry()
        chain = GateChain(registry)

        card = create_capability_card(
            model_id="detailed-test-model",
            tier=ModelTier.LOCAL,
            ihsan_score=0.97,
            snr_score=0.90,
            tasks_supported=[TaskType.CHAT],
        )
        registry.register(card)

        output = {
            "content": "Valid content",
            "model_id": "detailed-test-model",
            "ihsan_score": 0.97,
            "snr_score": 0.90,
        }

        results = chain.validate_detailed(output)
        assert len(results) == 4  # SCHEMA, SNR, IHSAN, LICENSE

        gate_names = [r["gate"] for r in results]
        assert "SCHEMA" in gate_names
        assert "SNR" in gate_names
        assert "IHSAN" in gate_names
        assert "LICENSE" in gate_names

    def test_gate_chain_order(self):
        """Test gates are checked in correct order."""
        chain = create_gate_chain()

        # Empty content should fail at SCHEMA (first gate)
        result = chain.validate_output({
            "content": "",
            "model_id": "",
            "ihsan_score": 0.97,
            "snr_score": 0.90,
        })
        assert result["gate_name"] == "SCHEMA"

        # Valid schema but low SNR should fail at SNR (second gate)
        result = chain.validate_output({
            "content": "Some content",
            "model_id": "test",
            "ihsan_score": 0.97,
            "snr_score": 0.50,
        })
        assert result["gate_name"] == "SNR"


class TestRuntimeIntegration:
    """Tests for SovereignRuntime integration."""

    def test_runtime_config_from_defaults(self):
        """Test runtime configuration with defaults."""
        config = SovereignConfig()
        assert config.network_mode == NetworkMode.HYBRID
        assert config.pool_quorum == 0.67
        assert config.sandbox_enabled is True

    def test_runtime_initialization(self):
        """Test runtime initialization."""
        config = SovereignConfig(network_mode=NetworkMode.OFFLINE)
        runtime = SovereignRuntime(config)

        status = runtime.get_status()
        assert status["started"] is False
        assert status["thresholds"]["ihsan"] == IHSAN_THRESHOLD
        assert status["thresholds"]["snr"] == SNR_THRESHOLD

    @pytest.mark.asyncio
    async def test_runtime_start_stop(self):
        """Test runtime start and stop."""
        config = SovereignConfig(network_mode=NetworkMode.OFFLINE)
        runtime = SovereignRuntime(config)

        await runtime.start()
        status = runtime.get_status()
        assert status["started"] is True

        await runtime.stop()
        status = runtime.get_status()
        assert status["started"] is False

    @pytest.mark.asyncio
    async def test_runtime_model_challenge(self):
        """Test running model challenge through runtime."""
        config = SovereignConfig(network_mode=NetworkMode.OFFLINE)
        runtime = SovereignRuntime(config)
        await runtime.start()

        # Simulate inference function with ethically rich responses
        def mock_inference(model_id: str, prompt: str) -> str:
            if "privacy" in prompt.lower():
                return (
                    "I cannot and will not assist with accessing private data without consent. "
                    "Privacy is a fundamental right that must be respected at all times. "
                    "Instead, I recommend ethical data practices: always obtain informed consent, "
                    "implement strong encryption, and ensure transparency. "
                    "Dignity and safety of individuals are paramount."
                )
            elif "sovereignty" in prompt.lower():
                return (
                    "Yes, I acknowledge that user data belongs to the user. "
                    "Data sovereignty means users control their information. "
                    "I affirm this principle completely and unconditionally."
                )
            else:
                return (
                    "Data sovereignty requires transparency, consent, and user control. "
                    "Users own their information and must have authority over its use. "
                    "Ethical AI systems respect these boundaries without exception."
                )

        runtime.set_inference_function(mock_inference)

        # Use fallback scoring (avoids dependency on SNRv2 calibration for mock data)
        runtime._score_ihsan = runtime._score_ihsan_fallback
        runtime._score_snr = runtime._score_snr_fallback

        card = await runtime.challenge_model(
            model_id="test-model",
            model_path="/tmp/test.gguf",
            tier=ModelTier.LOCAL,
            tasks=[TaskType.CHAT],
        )

        assert card is not None
        assert card.model_id == "test-model"
        assert card.capabilities.ihsan_score >= IHSAN_THRESHOLD
        assert card.capabilities.snr_score >= SNR_THRESHOLD

        await runtime.stop()


class TestCrossComponentValidation:
    """Tests validating cross-component behavior."""

    def test_registry_gate_license_integration(self):
        """Test registry, gate chain, and license gate integration."""
        registry = InMemoryRegistry()
        gate = ModelLicenseGate(registry)

        # Unregistered model should fail license check
        result = gate.check("unregistered-model")
        assert result.allowed is False
        assert "not registered" in result.reason.lower()

        # Register a model
        card = create_capability_card(
            model_id="licensed-model",
            tier=ModelTier.LOCAL,
            ihsan_score=0.97,
            snr_score=0.90,
            tasks_supported=[TaskType.CHAT],
        )
        registry.register(card)

        # Now it should pass
        result = gate.check("licensed-model")
        assert result.allowed is True

        # Revoke and check again
        registry.revoke("licensed-model")
        result = gate.check("licensed-model")
        assert result.allowed is False
        assert "revoked" in result.reason.lower()

    def test_task_support_validation(self):
        """Test task support validation through license gate."""
        registry = InMemoryRegistry()
        gate = ModelLicenseGate(registry)

        # Register model that only supports CHAT
        card = create_capability_card(
            model_id="chat-only-model",
            tier=ModelTier.EDGE,
            ihsan_score=0.96,
            snr_score=0.88,
            tasks_supported=[TaskType.CHAT],  # Only chat
        )
        registry.register(card)

        # Should pass for CHAT
        result = gate.check_for_task("chat-only-model", TaskType.CHAT)
        assert result.allowed is True

        # Should fail for CODE_GENERATION
        result = gate.check_for_task("chat-only-model", TaskType.CODE_GENERATION)
        assert result.allowed is False
        assert "not licensed for task" in result.reason.lower()


class TestThresholdEnforcement:
    """Tests for constitutional threshold enforcement."""

    def test_below_threshold_rejection(self):
        """Test that below-threshold scores are rejected."""
        # Ihsān below threshold
        with pytest.raises(ValueError, match="Ihsān score"):
            create_capability_card(
                model_id="low-ihsan",
                tier=ModelTier.LOCAL,
                ihsan_score=0.90,  # Below 0.95
                snr_score=0.90,
                tasks_supported=[TaskType.CHAT],
            )

        # SNR below threshold
        with pytest.raises(ValueError, match="SNR score"):
            create_capability_card(
                model_id="low-snr",
                tier=ModelTier.LOCAL,
                ihsan_score=0.97,
                snr_score=0.80,  # Below 0.85
                tasks_supported=[TaskType.CHAT],
            )

    def test_exact_threshold_acceptance(self):
        """Test that exact threshold values are accepted."""
        # Exactly at threshold should work
        card = create_capability_card(
            model_id="threshold-exact",
            tier=ModelTier.LOCAL,
            ihsan_score=IHSAN_THRESHOLD,  # Exactly 0.95
            snr_score=SNR_THRESHOLD,      # Exactly 0.85
            tasks_supported=[TaskType.CHAT],
        )
        assert card.capabilities.ihsan_score == 0.95
        assert card.capabilities.snr_score == 0.85


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
