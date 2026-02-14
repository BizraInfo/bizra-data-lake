"""
BIZRA PCI Gates Test Suite

Tests for the Proof-Carrying Inference gate chain.
Target: 70% coverage of core/pci/gates.py (139 lines)
"""

import pytest
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path (works across platforms)
_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from core.pci import (
    PCIGateKeeper,
    VerificationResult,
    RejectCode,
    EnvelopeBuilder,
    generate_keypair,
    IHSAN_MINIMUM_THRESHOLD,
    SNR_MINIMUM_THRESHOLD,
)
from core.pci.gates import (
    MAX_CLOCK_SKEW_SECONDS,
    NONCE_TTL_SECONDS,
    MAX_NONCE_CACHE_SIZE,
)


@pytest.fixture
def keypair():
    """Generate a test keypair."""
    return generate_keypair()


@pytest.fixture
def gatekeeper():
    """Create a fresh gatekeeper instance."""
    return PCIGateKeeper()


@pytest.fixture
def valid_envelope(keypair):
    """Create a valid signed envelope."""
    priv, pub = keypair
    return (
        EnvelopeBuilder()
        .with_sender("PAT", "test_agent", pub)
        .with_payload("test_action", {"key": "value"}, "", "")
        .with_metadata(0.96, 0.90)
        .build()
        .sign(priv)
    )


class TestSignatureGate:
    """Tests for Ed25519 signature validation."""

    def test_missing_signature_rejected(self, gatekeeper, keypair):
        """Envelope without signature should be rejected."""
        _, pub = keypair
        envelope = (
            EnvelopeBuilder()
            .with_sender("PAT", "test_agent", pub)
            .with_payload("test", {}, "", "")
            .with_metadata(0.96, 0.90)
            .build()
        )
        # Don't sign - signature is None

        result = gatekeeper.verify(envelope)

        assert not result.passed
        assert result.reject_code == RejectCode.REJECT_SIGNATURE
        assert "Missing signature" in result.details

    def test_invalid_signature_rejected(self, gatekeeper, keypair):
        """Envelope with tampered signature should be rejected."""
        priv, pub = keypair
        envelope = (
            EnvelopeBuilder()
            .with_sender("PAT", "test_agent", pub)
            .with_payload("test", {}, "", "")
            .with_metadata(0.96, 0.90)
            .build()
            .sign(priv)
        )
        # Tamper with signature
        envelope.signature.value = "00" * 64

        result = gatekeeper.verify(envelope)

        assert not result.passed
        assert result.reject_code == RejectCode.REJECT_SIGNATURE

    def test_wrong_key_signature_rejected(self, gatekeeper):
        """Envelope signed with wrong key should be rejected."""
        priv1, pub1 = generate_keypair()
        priv2, pub2 = generate_keypair()

        envelope = (
            EnvelopeBuilder()
            .with_sender("PAT", "test_agent", pub1)  # Claims pub1
            .with_payload("test", {}, "", "")
            .with_metadata(0.96, 0.90)
            .build()
            .sign(priv2)  # But signs with priv2
        )

        result = gatekeeper.verify(envelope)

        assert not result.passed
        assert result.reject_code == RejectCode.REJECT_SIGNATURE


class TestTimestampGate:
    """Tests for timestamp validation."""

    def test_stale_timestamp_rejected(self, gatekeeper, keypair):
        """Envelope with old timestamp should be rejected."""
        priv, pub = keypair
        envelope = (
            EnvelopeBuilder()
            .with_sender("PAT", "test_agent", pub)
            .with_payload("test", {}, "", "")
            .with_metadata(0.96, 0.90)
            .build()
        )
        # Set timestamp to 10 minutes ago
        old_time = datetime.now(timezone.utc) - timedelta(minutes=10)
        envelope.timestamp = old_time.isoformat().replace('+00:00', 'Z')
        envelope.sign(priv)

        result = gatekeeper.verify(envelope)

        assert not result.passed
        assert result.reject_code == RejectCode.REJECT_TIMESTAMP_STALE

    def test_future_timestamp_rejected(self, gatekeeper, keypair):
        """Envelope with future timestamp should be rejected."""
        priv, pub = keypair
        envelope = (
            EnvelopeBuilder()
            .with_sender("PAT", "test_agent", pub)
            .with_payload("test", {}, "", "")
            .with_metadata(0.96, 0.90)
            .build()
        )
        # Set timestamp to 10 minutes in future
        future_time = datetime.now(timezone.utc) + timedelta(minutes=10)
        envelope.timestamp = future_time.isoformat().replace('+00:00', 'Z')
        envelope.sign(priv)

        result = gatekeeper.verify(envelope)

        assert not result.passed
        assert result.reject_code == RejectCode.REJECT_TIMESTAMP_FUTURE

    def test_valid_timestamp_passes(self, gatekeeper, valid_envelope):
        """Envelope with current timestamp should pass."""
        result = gatekeeper.verify(valid_envelope)

        assert result.passed or result.reject_code != RejectCode.REJECT_TIMESTAMP_STALE


class TestReplayProtection:
    """Tests for nonce-based replay attack prevention."""

    def test_replay_attack_blocked(self, gatekeeper, valid_envelope):
        """Same envelope used twice should be rejected."""
        result1 = gatekeeper.verify(valid_envelope)
        assert result1.passed

        result2 = gatekeeper.verify(valid_envelope)
        assert not result2.passed
        assert result2.reject_code == RejectCode.REJECT_NONCE_REPLAY

    def test_nonce_cache_pruning(self, gatekeeper, keypair):
        """Old nonces should be pruned from cache."""
        priv, pub = keypair

        # Add a nonce to the cache with old timestamp
        old_nonce = "old_nonce_12345"
        gatekeeper.seen_nonces[old_nonce] = time.time() - NONCE_TTL_SECONDS - 10

        # Force pruning
        gatekeeper._prune_expired_nonces()

        assert old_nonce not in gatekeeper.seen_nonces

    def test_cache_size_limit_enforced(self, gatekeeper):
        """Cache should not exceed maximum size."""
        # Fill cache beyond limit
        now = time.time()
        for i in range(MAX_NONCE_CACHE_SIZE + 100):
            gatekeeper.seen_nonces[f"nonce_{i}"] = now

        gatekeeper._prune_expired_nonces()

        assert len(gatekeeper.seen_nonces) <= MAX_NONCE_CACHE_SIZE


class TestIhsanGate:
    """Tests for Ihsan (excellence) threshold."""

    def test_ihsan_below_threshold_rejected(self, gatekeeper, keypair):
        """Envelope with Ihsan < 0.95 should be rejected."""
        priv, pub = keypair
        envelope = (
            EnvelopeBuilder()
            .with_sender("PAT", "test_agent", pub)
            .with_payload("test", {}, "", "")
            .with_metadata(0.80, 0.90)  # Ihsan too low
            .build()
            .sign(priv)
        )

        result = gatekeeper.verify(envelope)

        assert not result.passed
        assert result.reject_code == RejectCode.REJECT_IHSAN_BELOW_MIN
        assert str(IHSAN_MINIMUM_THRESHOLD) in result.details

    def test_ihsan_at_threshold_passes(self, gatekeeper, keypair):
        """Envelope with Ihsan = 0.95 should pass."""
        priv, pub = keypair
        envelope = (
            EnvelopeBuilder()
            .with_sender("PAT", "test_agent", pub)
            .with_payload("test", {}, "", "")
            .with_metadata(0.95, 0.90)  # Exactly at threshold
            .build()
            .sign(priv)
        )

        result = gatekeeper.verify(envelope)

        # Should pass Ihsan gate (may fail on other gates like policy)
        assert result.reject_code != RejectCode.REJECT_IHSAN_BELOW_MIN


class TestSNRGate:
    """Tests for SNR (signal-to-noise ratio) threshold (SEC-020)."""

    def test_snr_below_threshold_rejected(self, gatekeeper, keypair):
        """Envelope with SNR < 0.85 should be rejected."""
        priv, pub = keypair
        envelope = (
            EnvelopeBuilder()
            .with_sender("PAT", "test_agent", pub)
            .with_payload("test", {}, "", "")
            .with_metadata(0.96, 0.70)  # SNR too low
            .build()
            .sign(priv)
        )

        result = gatekeeper.verify(envelope)

        assert not result.passed
        assert result.reject_code == RejectCode.REJECT_SNR_BELOW_MIN
        assert str(SNR_MINIMUM_THRESHOLD) in result.details

    def test_snr_at_threshold_passes(self, gatekeeper, keypair):
        """Envelope with SNR = 0.85 should pass."""
        priv, pub = keypair
        envelope = (
            EnvelopeBuilder()
            .with_sender("PAT", "test_agent", pub)
            .with_payload("test", {}, "", "")
            .with_metadata(0.96, 0.85)  # Exactly at threshold
            .build()
            .sign(priv)
        )

        result = gatekeeper.verify(envelope)

        # Should pass SNR gate (may fail on other gates like policy)
        assert result.reject_code != RejectCode.REJECT_SNR_BELOW_MIN


class TestPolicyGate:
    """Tests for policy hash verification."""

    def test_policy_mismatch_rejected(self, keypair):
        """Envelope with wrong policy hash should be rejected."""
        priv, pub = keypair
        gatekeeper = PCIGateKeeper(policy_enforcement=True)

        envelope = (
            EnvelopeBuilder()
            .with_sender("PAT", "test_agent", pub)
            .with_payload("test", {}, "wrong_policy_hash", "")
            .with_metadata(0.96, 0.90)
            .build()
            .sign(priv)
        )

        result = gatekeeper.verify(envelope)

        assert not result.passed
        assert result.reject_code == RejectCode.REJECT_POLICY_MISMATCH

    def test_policy_enforcement_disabled(self, keypair):
        """Policy mismatch should pass when enforcement disabled."""
        priv, pub = keypair
        gatekeeper = PCIGateKeeper(policy_enforcement=False)

        envelope = (
            EnvelopeBuilder()
            .with_sender("PAT", "test_agent", pub)
            .with_payload("test", {}, "wrong_policy_hash", "")
            .with_metadata(0.96, 0.90)
            .build()
            .sign(priv)
        )

        result = gatekeeper.verify(envelope)

        # Should pass policy gate when enforcement is off
        assert result.reject_code != RejectCode.REJECT_POLICY_MISMATCH


class TestFullGateChain:
    """Tests for complete gate chain verification."""

    def test_valid_envelope_passes_all_gates(self, gatekeeper, keypair):
        """Fully valid envelope should pass all gates."""
        priv, pub = keypair
        gatekeeper = PCIGateKeeper(policy_enforcement=False)

        envelope = (
            EnvelopeBuilder()
            .with_sender("PAT", "test_agent", pub)
            .with_payload("test", {"key": "value"}, "", "")
            .with_metadata(0.98, 0.92)
            .build()
            .sign(priv)
        )

        result = gatekeeper.verify(envelope)

        assert result.passed
        assert result.reject_code == RejectCode.SUCCESS
        assert "SCHEMA" in result.gate_passed
        assert "SIGNATURE" in result.gate_passed
        assert "TIMESTAMP" in result.gate_passed
        assert "REPLAY" in result.gate_passed
        assert "IHSAN" in result.gate_passed
        assert "SNR" in result.gate_passed
        assert "POLICY" in result.gate_passed

    def test_gate_order_preserved(self, gatekeeper, keypair):
        """Gates should be checked in correct order."""
        priv, pub = keypair
        gatekeeper = PCIGateKeeper(policy_enforcement=False)

        envelope = (
            EnvelopeBuilder()
            .with_sender("PAT", "test_agent", pub)
            .with_payload("test", {}, "", "")
            .with_metadata(0.96, 0.90)
            .build()
            .sign(priv)
        )

        result = gatekeeper.verify(envelope)

        # Verify expected order
        expected_order = ["SCHEMA", "SIGNATURE", "TIMESTAMP", "REPLAY", "IHSAN", "SNR", "POLICY"]
        assert result.gate_passed == expected_order


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_verification_result_fields(self):
        """VerificationResult should have expected fields."""
        result = VerificationResult(
            passed=True,
            reject_code=RejectCode.SUCCESS,
            details="All gates passed",
            gate_passed=["SCHEMA", "SIGNATURE"]
        )

        assert result.passed is True
        assert result.reject_code == RejectCode.SUCCESS
        assert result.details == "All gates passed"
        assert result.gate_passed == ["SCHEMA", "SIGNATURE"]

    def test_failed_result_fields(self):
        """Failed VerificationResult should contain rejection info."""
        result = VerificationResult(
            passed=False,
            reject_code=RejectCode.REJECT_SIGNATURE,
            details="Invalid signature"
        )

        assert result.passed is False
        assert result.reject_code == RejectCode.REJECT_SIGNATURE
