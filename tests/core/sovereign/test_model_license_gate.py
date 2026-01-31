"""
Tests for BIZRA Model License Gate

Tests the gate chain validation and license checks.
"""

import pytest

from core.sovereign.capability_card import (
    create_capability_card,
    ModelTier,
    TaskType,
    CardIssuer,
    IHSAN_THRESHOLD,
    SNR_THRESHOLD,
)
from core.sovereign.model_license_gate import (
    ModelLicenseGate,
    InMemoryRegistry,
    GateChain,
    create_gate_chain,
)


class TestInMemoryRegistry:
    """Tests for the in-memory model registry."""

    def test_register_and_get(self):
        """Test registering and retrieving a card."""
        registry = InMemoryRegistry()
        issuer = CardIssuer()

        card = create_capability_card(
            model_id="test-model",
            tier=ModelTier.LOCAL,
            ihsan_score=0.97,
            snr_score=0.90,
            tasks_supported=[TaskType.CHAT],
        )
        signed_card = issuer.issue(card)

        registry.register(signed_card)
        assert registry.has("test-model") is True

        retrieved = registry.get("test-model")
        assert retrieved is not None
        assert retrieved.model_id == "test-model"

    def test_get_unregistered_model(self):
        """Test getting an unregistered model returns None."""
        registry = InMemoryRegistry()
        assert registry.get("nonexistent") is None
        assert registry.has("nonexistent") is False

    def test_revoke_model(self):
        """Test revoking a model's registration."""
        registry = InMemoryRegistry()

        card = create_capability_card(
            model_id="test-model",
            tier=ModelTier.LOCAL,
            ihsan_score=0.97,
            snr_score=0.90,
            tasks_supported=[TaskType.CHAT],
        )
        registry.register(card)

        assert registry.revoke("test-model") is True
        retrieved = registry.get("test-model")
        assert retrieved.revoked is True

    def test_list_valid(self):
        """Test listing only valid cards."""
        registry = InMemoryRegistry()

        # Add a valid card
        valid_card = create_capability_card(
            model_id="valid-model",
            tier=ModelTier.LOCAL,
            ihsan_score=0.97,
            snr_score=0.90,
            tasks_supported=[TaskType.CHAT],
        )
        registry.register(valid_card)

        # Add and revoke a card
        revoked_card = create_capability_card(
            model_id="revoked-model",
            tier=ModelTier.LOCAL,
            ihsan_score=0.97,
            snr_score=0.90,
            tasks_supported=[TaskType.CHAT],
        )
        registry.register(revoked_card)
        registry.revoke("revoked-model")

        valid_cards = registry.list_valid()
        assert len(valid_cards) == 1
        assert valid_cards[0].model_id == "valid-model"


class TestModelLicenseGate:
    """Tests for the model license gate."""

    def test_check_registered_model(self):
        """Test checking a registered model."""
        registry = InMemoryRegistry()
        gate = ModelLicenseGate(registry)

        card = create_capability_card(
            model_id="test-model",
            tier=ModelTier.LOCAL,
            ihsan_score=0.97,
            snr_score=0.90,
            tasks_supported=[TaskType.CHAT],
        )
        registry.register(card)

        result = gate.check("test-model")
        assert result.allowed is True
        assert result.model_id == "test-model"
        assert result.tier == ModelTier.LOCAL

    def test_check_unregistered_model(self):
        """Test checking an unregistered model."""
        registry = InMemoryRegistry()
        gate = ModelLicenseGate(registry)

        result = gate.check("unregistered-model")
        assert result.allowed is False
        assert "not registered" in result.reason.lower()

    def test_check_revoked_model(self):
        """Test checking a revoked model."""
        registry = InMemoryRegistry()
        gate = ModelLicenseGate(registry)

        card = create_capability_card(
            model_id="test-model",
            tier=ModelTier.LOCAL,
            ihsan_score=0.97,
            snr_score=0.90,
            tasks_supported=[TaskType.CHAT],
        )
        registry.register(card)
        registry.revoke("test-model")

        result = gate.check("test-model")
        assert result.allowed is False
        assert "revoked" in result.reason.lower()

    def test_check_for_supported_task(self):
        """Test checking for a supported task."""
        registry = InMemoryRegistry()
        gate = ModelLicenseGate(registry)

        card = create_capability_card(
            model_id="test-model",
            tier=ModelTier.LOCAL,
            ihsan_score=0.97,
            snr_score=0.90,
            tasks_supported=[TaskType.CHAT, TaskType.REASONING],
        )
        registry.register(card)

        result = gate.check_for_task("test-model", TaskType.CHAT)
        assert result.allowed is True

    def test_check_for_unsupported_task(self):
        """Test checking for an unsupported task."""
        registry = InMemoryRegistry()
        gate = ModelLicenseGate(registry)

        card = create_capability_card(
            model_id="test-model",
            tier=ModelTier.LOCAL,
            ihsan_score=0.97,
            snr_score=0.90,
            tasks_supported=[TaskType.CHAT],  # Only chat
        )
        registry.register(card)

        result = gate.check_for_task("test-model", TaskType.CODE_GENERATION)
        assert result.allowed is False
        assert "not licensed for task" in result.reason.lower()


class TestGateChain:
    """Tests for the complete gate chain."""

    def test_validate_valid_output(self):
        """Test validating a valid output."""
        registry = InMemoryRegistry()
        chain = GateChain(registry)

        # Register a model
        card = create_capability_card(
            model_id="test-model",
            tier=ModelTier.LOCAL,
            ihsan_score=0.97,
            snr_score=0.90,
            tasks_supported=[TaskType.CHAT],
        )
        registry.register(card)

        output = {
            "content": "Hello, world!",
            "model_id": "test-model",
            "ihsan_score": 0.97,
            "snr_score": 0.90,
        }

        result = chain.validate_output(output)
        assert result["passed"] is True
        assert result["gate_name"] == "ALL"

    def test_schema_gate_fails_empty_content(self):
        """Test schema gate fails on empty content."""
        chain = create_gate_chain()

        output = {
            "content": "",
            "model_id": "test-model",
            "ihsan_score": 0.97,
            "snr_score": 0.90,
        }

        result = chain.validate_output(output)
        assert result["passed"] is False
        assert result["gate_name"] == "SCHEMA"

    def test_snr_gate_fails_low_score(self):
        """Test SNR gate fails on low score."""
        chain = create_gate_chain()

        output = {
            "content": "Hello, world!",
            "model_id": "test-model",
            "ihsan_score": 0.97,
            "snr_score": 0.50,  # Below threshold
        }

        result = chain.validate_output(output)
        assert result["passed"] is False
        assert result["gate_name"] == "SNR"

    def test_ihsan_gate_fails_low_score(self):
        """Test Ihsān gate fails on low score."""
        chain = create_gate_chain()

        output = {
            "content": "Hello, world!",
            "model_id": "test-model",
            "ihsan_score": 0.80,  # Below threshold
            "snr_score": 0.90,
        }

        result = chain.validate_output(output)
        assert result["passed"] is False
        assert result["gate_name"] == "IHSAN"

    def test_license_gate_fails_unregistered(self):
        """Test license gate fails on unregistered model."""
        chain = create_gate_chain()

        output = {
            "content": "Hello, world!",
            "model_id": "unregistered-model",
            "ihsan_score": 0.97,
            "snr_score": 0.90,
        }

        result = chain.validate_output(output)
        assert result["passed"] is False
        assert result["gate_name"] == "LICENSE"

    def test_validate_detailed(self):
        """Test detailed validation results."""
        registry = InMemoryRegistry()
        chain = GateChain(registry)

        card = create_capability_card(
            model_id="test-model",
            tier=ModelTier.LOCAL,
            ihsan_score=0.97,
            snr_score=0.90,
            tasks_supported=[TaskType.CHAT],
        )
        registry.register(card)

        output = {
            "content": "Hello, world!",
            "model_id": "test-model",
            "ihsan_score": 0.97,
            "snr_score": 0.90,
        }

        results = chain.validate_detailed(output)
        assert len(results) == 4  # SCHEMA, SNR, IHSAN, LICENSE
        assert all(r["passed"] for r in results)

    def test_gate_order(self):
        """Test gates are checked in correct order."""
        # Gates should be: SCHEMA → SNR → IHSAN → LICENSE
        chain = create_gate_chain()

        # Missing everything - should fail at SCHEMA
        output = {"content": "", "model_id": ""}
        result = chain.validate_output(output)
        assert result["gate_name"] == "SCHEMA"

        # Valid schema but no scores - should fail at SNR
        output = {"content": "Hello", "model_id": "test"}
        result = chain.validate_output(output)
        assert result["gate_name"] == "SNR"

        # Valid SNR but missing Ihsān - should fail at IHSAN
        output = {"content": "Hello", "model_id": "test", "snr_score": 0.90}
        result = chain.validate_output(output)
        assert result["gate_name"] == "IHSAN"


class TestThresholdConstants:
    """Tests to verify threshold constants are correct."""

    def test_ihsan_threshold(self):
        """Verify Ihsān threshold is 0.95."""
        assert IHSAN_THRESHOLD == 0.95

    def test_snr_threshold(self):
        """Verify SNR threshold is 0.85."""
        assert SNR_THRESHOLD == 0.85
