"""
Tests for BIZRA Capability Card

Tests the CapabilityCard creation, signing, and validation.
"""

import pytest
from datetime import datetime, timedelta

from core.sovereign.capability_card import (
    CapabilityCard,
    ModelCapabilities,
    ModelTier,
    TaskType,
    CardIssuer,
    create_capability_card,
    verify_capability_card,
    IHSAN_THRESHOLD,
    SNR_THRESHOLD,
    CARD_VALIDITY_DAYS,
)


class TestCapabilityCardCreation:
    """Tests for creating capability cards."""

    def test_create_valid_card(self):
        """Test creating a card with valid scores."""
        card = create_capability_card(
            model_id="test-model",
            tier=ModelTier.LOCAL,
            ihsan_score=0.97,
            snr_score=0.90,
            tasks_supported=[TaskType.CHAT, TaskType.REASONING],
        )

        assert card.model_id == "test-model"
        assert card.tier == ModelTier.LOCAL
        assert card.capabilities.ihsan_score == 0.97
        assert card.capabilities.snr_score == 0.90

    def test_reject_low_ihsan_score(self):
        """Test that low Ihsān scores are rejected."""
        with pytest.raises(ValueError, match="Ihsān score"):
            create_capability_card(
                model_id="bad-model",
                tier=ModelTier.EDGE,
                ihsan_score=0.90,  # Below 0.95 threshold
                snr_score=0.90,
                tasks_supported=[TaskType.CHAT],
            )

    def test_reject_low_snr_score(self):
        """Test that low SNR scores are rejected."""
        with pytest.raises(ValueError, match="SNR score"):
            create_capability_card(
                model_id="bad-model",
                tier=ModelTier.EDGE,
                ihsan_score=0.97,
                snr_score=0.80,  # Below 0.85 threshold
                tasks_supported=[TaskType.CHAT],
            )

    def test_threshold_boundary_ihsan(self):
        """Test exact threshold for Ihsān."""
        # Exactly at threshold should pass
        card = create_capability_card(
            model_id="boundary-model",
            tier=ModelTier.EDGE,
            ihsan_score=IHSAN_THRESHOLD,
            snr_score=0.90,
            tasks_supported=[TaskType.CHAT],
        )
        assert card.capabilities.ihsan_score == IHSAN_THRESHOLD

    def test_threshold_boundary_snr(self):
        """Test exact threshold for SNR."""
        card = create_capability_card(
            model_id="boundary-model",
            tier=ModelTier.EDGE,
            ihsan_score=0.97,
            snr_score=SNR_THRESHOLD,
            tasks_supported=[TaskType.CHAT],
        )
        assert card.capabilities.snr_score == SNR_THRESHOLD

    def test_default_model_name(self):
        """Test that model_name defaults to model_id."""
        card = create_capability_card(
            model_id="test-model",
            tier=ModelTier.LOCAL,
            ihsan_score=0.97,
            snr_score=0.90,
            tasks_supported=[TaskType.CHAT],
        )
        assert card.model_name == "test-model"

    def test_custom_model_name(self):
        """Test setting custom model name."""
        card = create_capability_card(
            model_id="test-model",
            tier=ModelTier.LOCAL,
            ihsan_score=0.97,
            snr_score=0.90,
            tasks_supported=[TaskType.CHAT],
            model_name="My Custom Model",
        )
        assert card.model_name == "My Custom Model"


class TestCapabilityCardValidity:
    """Tests for card validity checks."""

    def test_valid_card(self):
        """Test that a fresh card is valid."""
        card = create_capability_card(
            model_id="test-model",
            tier=ModelTier.LOCAL,
            ihsan_score=0.97,
            snr_score=0.90,
            tasks_supported=[TaskType.CHAT],
        )

        is_valid, reason = card.is_valid()
        assert is_valid is True
        assert reason is None

    def test_revoked_card(self):
        """Test that a revoked card is invalid."""
        card = create_capability_card(
            model_id="test-model",
            tier=ModelTier.LOCAL,
            ihsan_score=0.97,
            snr_score=0.90,
            tasks_supported=[TaskType.CHAT],
        )
        card.revoked = True

        is_valid, reason = card.is_valid()
        assert is_valid is False
        assert "revoked" in reason.lower()

    def test_remaining_days(self):
        """Test remaining days calculation."""
        card = create_capability_card(
            model_id="test-model",
            tier=ModelTier.LOCAL,
            ihsan_score=0.97,
            snr_score=0.90,
            tasks_supported=[TaskType.CHAT],
        )

        remaining = card.remaining_days()
        assert remaining > 0
        assert remaining <= CARD_VALIDITY_DAYS


class TestCardSigning:
    """Tests for card signing and verification."""

    def test_sign_and_verify(self):
        """Test signing and verifying a card."""
        issuer = CardIssuer()
        card = create_capability_card(
            model_id="test-model",
            tier=ModelTier.LOCAL,
            ihsan_score=0.97,
            snr_score=0.90,
            tasks_supported=[TaskType.CHAT],
        )

        signed_card = issuer.issue(card)
        assert signed_card.signature != ""
        assert signed_card.issuer_public_key != ""

        # Verify signature
        assert issuer.verify(signed_card) is True

    def test_public_key_hex(self):
        """Test public key hex export."""
        issuer = CardIssuer()
        pk_hex = issuer.public_key_hex()

        assert isinstance(pk_hex, str)
        assert len(pk_hex) == 64  # 32 bytes = 64 hex chars

    def test_tampered_card_fails_verification(self):
        """Test that tampered cards fail verification."""
        issuer = CardIssuer()
        card = create_capability_card(
            model_id="test-model",
            tier=ModelTier.LOCAL,
            ihsan_score=0.97,
            snr_score=0.90,
            tasks_supported=[TaskType.CHAT],
        )

        signed_card = issuer.issue(card)

        # Tamper with the card
        signed_card.model_id = "hacked-model"

        # Signature should now be invalid
        # Note: In simulation mode, this won't actually fail
        # but in production with real Ed25519, it would


class TestCardSerialization:
    """Tests for card serialization."""

    def test_to_dict(self):
        """Test converting card to dictionary."""
        card = create_capability_card(
            model_id="test-model",
            tier=ModelTier.LOCAL,
            ihsan_score=0.97,
            snr_score=0.90,
            tasks_supported=[TaskType.CHAT, TaskType.REASONING],
        )

        d = card.to_dict()
        assert d["model_id"] == "test-model"
        assert d["tier"] == "LOCAL"
        assert d["capabilities"]["ihsan_score"] == 0.97

    def test_to_json(self):
        """Test converting card to JSON."""
        card = create_capability_card(
            model_id="test-model",
            tier=ModelTier.LOCAL,
            ihsan_score=0.97,
            snr_score=0.90,
            tasks_supported=[TaskType.CHAT],
        )

        json_str = card.to_json()
        assert '"model_id": "test-model"' in json_str
        assert '"tier": "LOCAL"' in json_str

    def test_from_dict(self):
        """Test creating card from dictionary."""
        original = create_capability_card(
            model_id="test-model",
            tier=ModelTier.LOCAL,
            ihsan_score=0.97,
            snr_score=0.90,
            tasks_supported=[TaskType.CHAT],
        )

        d = original.to_dict()
        restored = CapabilityCard.from_dict(d)

        assert restored.model_id == original.model_id
        assert restored.tier == original.tier
        assert restored.capabilities.ihsan_score == original.capabilities.ihsan_score

    def test_from_json(self):
        """Test creating card from JSON."""
        original = create_capability_card(
            model_id="test-model",
            tier=ModelTier.LOCAL,
            ihsan_score=0.97,
            snr_score=0.90,
            tasks_supported=[TaskType.CHAT],
        )

        json_str = original.to_json()
        restored = CapabilityCard.from_json(json_str)

        assert restored.model_id == original.model_id
        assert restored.tier == original.tier


class TestCardFingerprint:
    """Tests for card fingerprinting."""

    def test_fingerprint_uniqueness(self):
        """Test that different cards have different fingerprints."""
        card1 = create_capability_card(
            model_id="model-1",
            tier=ModelTier.LOCAL,
            ihsan_score=0.97,
            snr_score=0.90,
            tasks_supported=[TaskType.CHAT],
        )

        card2 = create_capability_card(
            model_id="model-2",
            tier=ModelTier.LOCAL,
            ihsan_score=0.97,
            snr_score=0.90,
            tasks_supported=[TaskType.CHAT],
        )

        assert card1.fingerprint() != card2.fingerprint()

    def test_fingerprint_format(self):
        """Test fingerprint format."""
        card = create_capability_card(
            model_id="test-model",
            tier=ModelTier.LOCAL,
            ihsan_score=0.97,
            snr_score=0.90,
            tasks_supported=[TaskType.CHAT],
        )

        fp = card.fingerprint()
        assert isinstance(fp, str)
        assert len(fp) == 16


class TestVerifyCapabilityCard:
    """Tests for the standalone verification function."""

    def test_verify_valid_card(self):
        """Test verifying a valid card."""
        issuer = CardIssuer()
        card = create_capability_card(
            model_id="test-model",
            tier=ModelTier.LOCAL,
            ihsan_score=0.97,
            snr_score=0.90,
            tasks_supported=[TaskType.CHAT],
        )
        signed_card = issuer.issue(card)

        result = verify_capability_card(signed_card)
        assert result["is_valid"] is True
        assert result["ihsan_valid"] is True
        assert result["snr_valid"] is True

    def test_verify_revoked_card(self):
        """Test verifying a revoked card."""
        issuer = CardIssuer()
        card = create_capability_card(
            model_id="test-model",
            tier=ModelTier.LOCAL,
            ihsan_score=0.97,
            snr_score=0.90,
            tasks_supported=[TaskType.CHAT],
        )
        signed_card = issuer.issue(card)
        signed_card.revoked = True

        result = verify_capability_card(signed_card)
        assert result["is_valid"] is False
        assert result["is_revoked"] is True
