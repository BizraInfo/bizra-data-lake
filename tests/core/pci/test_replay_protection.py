"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   REPLAY PROTECTION TEST SUITE                                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Tests for Security Hardening S-1: Replay Attack Prevention                 ║
║                                                                              ║
║   Standing on Giants:                                                        ║
║   - Lamport (1982): "Time, Clocks, and the Ordering of Events"              ║
║   - Merkle (1988): Cryptographic hash integrity                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pytest
from datetime import datetime, timezone, timedelta

from core.pci.envelope import (
    PCIEnvelope,
    EnvelopeSender,
    EnvelopePayload,
    EnvelopeMetadata,
    AgentType,
    MAX_MESSAGE_AGE_SECONDS,
    MAX_FUTURE_TIMESTAMP_SECONDS,
    _seen_nonces,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def valid_envelope():
    """Create a valid, fresh envelope."""
    return PCIEnvelope(
        version="1.0.0",
        envelope_id="test-envelope-001",
        timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        nonce="unique_nonce_12345",
        sender=EnvelopeSender(
            agent_type=AgentType.PAT,
            agent_id="test-agent",
            public_key="a" * 64,
        ),
        payload=EnvelopePayload(
            action="test",
            data={"test": "data"},
            policy_hash="policy123",
            state_hash="state456",
        ),
        metadata=EnvelopeMetadata(
            ihsan_score=0.96,
            snr_score=0.92,
        ),
    )


@pytest.fixture(autouse=True)
def clear_nonce_cache():
    """Clear nonce cache before each test."""
    global _seen_nonces
    _seen_nonces.clear()
    yield
    _seen_nonces.clear()


# =============================================================================
# TIMESTAMP VALIDATION TESTS
# =============================================================================

class TestTimestampValidation:
    """Test timestamp-based replay protection."""

    def test_fresh_message_not_expired(self, valid_envelope):
        """Fresh message should not be expired."""
        assert valid_envelope.is_expired() is False

    def test_old_message_expired(self, valid_envelope):
        """Message older than MAX_MESSAGE_AGE_SECONDS should be expired."""
        old_time = datetime.now(timezone.utc) - timedelta(seconds=MAX_MESSAGE_AGE_SECONDS + 60)
        valid_envelope.timestamp = old_time.isoformat().replace('+00:00', 'Z')

        assert valid_envelope.is_expired() is True

    def test_future_message_expired(self, valid_envelope):
        """Message from too far in future should be rejected (time-travel attack)."""
        future_time = datetime.now(timezone.utc) + timedelta(seconds=MAX_FUTURE_TIMESTAMP_SECONDS + 60)
        valid_envelope.timestamp = future_time.isoformat().replace('+00:00', 'Z')

        assert valid_envelope.is_expired() is True

    def test_slightly_future_message_ok(self, valid_envelope):
        """Message slightly in future (within tolerance) should be accepted."""
        future_time = datetime.now(timezone.utc) + timedelta(seconds=10)
        valid_envelope.timestamp = future_time.isoformat().replace('+00:00', 'Z')

        assert valid_envelope.is_expired() is False

    def test_invalid_timestamp_rejected(self, valid_envelope):
        """Invalid timestamp format should be rejected."""
        valid_envelope.timestamp = "not-a-timestamp"

        assert valid_envelope.is_expired() is True

    def test_empty_timestamp_rejected(self, valid_envelope):
        """Empty timestamp should be rejected."""
        valid_envelope.timestamp = ""

        assert valid_envelope.is_expired() is True


# =============================================================================
# NONCE REPLAY TESTS
# =============================================================================

class TestNonceReplayProtection:
    """Test nonce-based replay protection."""

    def test_first_message_not_replay(self, valid_envelope):
        """First occurrence of nonce should not be flagged as replay."""
        assert valid_envelope.is_replay() is False

    def test_second_message_is_replay(self, valid_envelope):
        """Second occurrence of same nonce should be flagged as replay."""
        # First call - not a replay
        assert valid_envelope.is_replay() is False

        # Create another envelope with same nonce
        replay_envelope = PCIEnvelope(
            version="1.0.0",
            envelope_id="different-id",
            timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            nonce=valid_envelope.nonce,  # Same nonce!
            sender=valid_envelope.sender,
            payload=valid_envelope.payload,
            metadata=valid_envelope.metadata,
        )

        # Second call with same nonce - IS a replay
        assert replay_envelope.is_replay() is True

    def test_different_nonces_both_accepted(self, valid_envelope):
        """Different nonces should both be accepted."""
        # First envelope
        assert valid_envelope.is_replay() is False

        # Second envelope with different nonce
        valid_envelope.nonce = "different_nonce_67890"
        assert valid_envelope.is_replay() is False


# =============================================================================
# COMBINED FRESHNESS VALIDATION TESTS
# =============================================================================

class TestFreshnessValidation:
    """Test combined timestamp + nonce validation."""

    def test_fresh_unique_message_passes(self, valid_envelope):
        """Fresh message with unique nonce should pass validation."""
        is_valid, error = valid_envelope.validate_freshness()

        assert is_valid is True
        assert error == ""

    def test_expired_message_fails(self, valid_envelope):
        """Expired message should fail validation."""
        old_time = datetime.now(timezone.utc) - timedelta(seconds=MAX_MESSAGE_AGE_SECONDS + 60)
        valid_envelope.timestamp = old_time.isoformat().replace('+00:00', 'Z')

        is_valid, error = valid_envelope.validate_freshness()

        assert is_valid is False
        assert "expired" in error.lower() or "future" in error.lower()

    def test_replayed_message_fails(self, valid_envelope):
        """Replayed message should fail validation."""
        # First call passes
        is_valid1, _ = valid_envelope.validate_freshness()
        assert is_valid1 is True

        # Create replay
        replay_envelope = PCIEnvelope(
            version="1.0.0",
            envelope_id="different-id",
            timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            nonce=valid_envelope.nonce,  # Same nonce
            sender=valid_envelope.sender,
            payload=valid_envelope.payload,
            metadata=valid_envelope.metadata,
        )

        # Second call fails
        is_valid2, error = replay_envelope.validate_freshness()

        assert is_valid2 is False
        assert "replay" in error.lower()


# =============================================================================
# ATTACK SCENARIO TESTS
# =============================================================================

class TestAttackScenarios:
    """Test real-world attack scenarios."""

    def test_replay_attack_blocked(self, valid_envelope):
        """
        SCENARIO: Attacker captures legitimate message and replays it later.

        Attack flow:
        1. Alice sends legitimate signed message at T0
        2. Attacker captures the message
        3. Attacker replays message at T1 (minutes/hours later)

        Expected: Message should be rejected (expired timestamp OR replayed nonce)
        """
        # Step 1: Original message sent
        is_valid, _ = valid_envelope.validate_freshness()
        assert is_valid is True

        # Step 3: Attacker replays exact same message
        # Even if timestamp is updated, nonce will catch it
        is_valid, error = valid_envelope.validate_freshness()
        assert is_valid is False
        assert "replay" in error.lower()

    def test_time_travel_attack_blocked(self, valid_envelope):
        """
        SCENARIO: Attacker creates message with future timestamp to bypass
        replay detection (pre-positioning attack).

        Expected: Message should be rejected (timestamp too far in future)
        """
        # Attacker sets timestamp 1 hour in future
        future_time = datetime.now(timezone.utc) + timedelta(hours=1)
        valid_envelope.timestamp = future_time.isoformat().replace('+00:00', 'Z')

        is_valid, error = valid_envelope.validate_freshness()

        assert is_valid is False

    def test_clock_skew_tolerance(self, valid_envelope):
        """
        SCENARIO: Legitimate messages from nodes with slight clock skew.

        Expected: Small clock differences should be tolerated
        """
        # Node with clock 20 seconds ahead
        slightly_future = datetime.now(timezone.utc) + timedelta(seconds=20)
        valid_envelope.timestamp = slightly_future.isoformat().replace('+00:00', 'Z')

        is_valid, _ = valid_envelope.validate_freshness()
        assert is_valid is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
