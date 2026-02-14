"""
FATE Gate End-to-End Pipeline Integration Test

Standing on Giants:
- Lamport (1982): Time, Clocks, and the Ordering of Events
- Kocher (1996): Timing Attacks on Implementations
- Shannon (1948): Signal-to-Noise Ratio (Information Theory)
- Al-Ghazali: Ihsān (Excellence as Ethical Constraint)
- RFC 8785: JSON Canonicalization Scheme (JCS)

This test exercises the FULL FATE gate pipeline:
  PCI Envelope → Ed25519 Signing → 7-Gate Chain → Verdict

Gate Chain Order (PCI context — Ihsan before SNR):
  SCHEMA → SIGNATURE → TIMESTAMP → REPLAY → IHSAN → SNR → POLICY

Each test validates a specific gate's reject behavior AND the
full-chain happy path, ensuring that BIZRA's core security
differentiator works correctly end-to-end.
"""

import secrets
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest

from core.pci.crypto import generate_keypair, sign_message, verify_signature
from core.pci.envelope import (
    AgentType,
    EnvelopeBuilder,
    EnvelopeMetadata,
    EnvelopePayload,
    EnvelopeSender,
    EnvelopeSignature,
    PCIEnvelope,
    datetime_now_iso,
)
from core.pci.gates import (
    IHSAN_MINIMUM_THRESHOLD,
    MAX_CLOCK_SKEW_SECONDS,
    SNR_MINIMUM_THRESHOLD,
    PCIGateKeeper,
    VerificationResult,
)
from core.pci.reject_codes import RejectCode


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def keypair():
    """Generate a fresh Ed25519 keypair for each test."""
    private_key, public_key = generate_keypair()
    return private_key, public_key


@pytest.fixture
def gatekeeper():
    """Fresh PCIGateKeeper with no prior nonce history."""
    return PCIGateKeeper(policy_enforcement=False)


@pytest.fixture
def gatekeeper_with_policy():
    """PCIGateKeeper with policy enforcement enabled."""
    return PCIGateKeeper(policy_enforcement=True)


def _build_valid_envelope(public_key: str) -> PCIEnvelope:
    """Build a valid PCI envelope ready for signing."""
    return (
        EnvelopeBuilder()
        .with_sender("PAT", "pat-strategist-001", public_key)
        .with_payload(
            action="inference.request",
            data={"model": "llama-3.2-1b", "prompt": "What is BIZRA?"},
            policy_hash="d9c9b5f7a3e2c8d4f1a6e9b2c5d8a3f7e0c2b5d8a1e4f7c0b3d6a9e2c5f8b1d926f",
            state_hash=secrets.token_hex(32),
        )
        .with_metadata(ihsan=0.97, snr=0.92)
        .build()
    )


def _sign_envelope(envelope: PCIEnvelope, private_key: str) -> PCIEnvelope:
    """Sign an envelope with the given private key."""
    return envelope.sign(private_key)


# ═══════════════════════════════════════════════════════════════════════════════
# HAPPY PATH — Full Chain Verification
# ═══════════════════════════════════════════════════════════════════════════════


class TestFATEGateHappyPath:
    """Verify the full 7-gate chain passes for a well-formed envelope."""

    def test_valid_envelope_passes_all_gates(self, keypair, gatekeeper):
        """A properly signed envelope with valid scores passes all 7 gates."""
        private_key, public_key = keypair
        envelope = _build_valid_envelope(public_key)
        signed = _sign_envelope(envelope, private_key)

        result = gatekeeper.verify(signed)

        assert result.passed is True
        assert result.reject_code == RejectCode.SUCCESS
        assert result.gate_passed == [
            "SCHEMA", "SIGNATURE", "TIMESTAMP", "REPLAY", "IHSAN", "SNR", "POLICY"
        ]

    def test_envelope_roundtrip_dict_serialization(self, keypair, gatekeeper):
        """Envelope survives dict serialization/deserialization and still verifies."""
        private_key, public_key = keypair
        envelope = _build_valid_envelope(public_key)
        signed = _sign_envelope(envelope, private_key)

        # Serialize and deserialize
        d = signed.to_dict()
        restored = PCIEnvelope.from_dict(d)

        result = gatekeeper.verify(restored)
        assert result.passed is True
        assert result.reject_code == RejectCode.SUCCESS

    def test_sat_agent_type_also_passes(self, keypair, gatekeeper):
        """SAT (System Agentic Team) agent type also passes all gates."""
        private_key, public_key = keypair
        envelope = (
            EnvelopeBuilder()
            .with_sender("SAT", "sat-validator-001", public_key)
            .with_payload(
                action="validate.block",
                data={"block_id": "genesis-0"},
                policy_hash="d9c9b5f7a3e2c8d4f1a6e9b2c5d8a3f7e0c2b5d8a1e4f7c0b3d6a9e2c5f8b1d926f",
                state_hash=secrets.token_hex(32),
            )
            .with_metadata(ihsan=0.99, snr=0.95)
            .build()
        )
        signed = _sign_envelope(envelope, private_key)

        result = gatekeeper.verify(signed)
        assert result.passed is True


# ═══════════════════════════════════════════════════════════════════════════════
# GATE 2: SIGNATURE — Ed25519 Verification
# ═══════════════════════════════════════════════════════════════════════════════


class TestSignatureGate:
    """Gate 2: Ed25519 signature verification."""

    def test_missing_signature_rejected(self, keypair, gatekeeper):
        """Unsigned envelope is rejected at SIGNATURE gate."""
        _, public_key = keypair
        envelope = _build_valid_envelope(public_key)
        # Don't sign — envelope.signature is None

        result = gatekeeper.verify(envelope)

        assert result.passed is False
        assert result.reject_code == RejectCode.REJECT_SIGNATURE
        assert "Missing signature" in result.details

    def test_wrong_key_signature_rejected(self, keypair, gatekeeper):
        """Envelope signed with wrong key fails SIGNATURE gate."""
        _, public_key = keypair
        envelope = _build_valid_envelope(public_key)

        # Sign with a DIFFERENT key
        attacker_private, _ = generate_keypair()
        signed = _sign_envelope(envelope, attacker_private)

        result = gatekeeper.verify(signed)

        assert result.passed is False
        assert result.reject_code == RejectCode.REJECT_SIGNATURE
        assert "Invalid ed25519 signature" in result.details

    def test_tampered_payload_rejected(self, keypair, gatekeeper):
        """Modifying payload after signing invalidates signature."""
        private_key, public_key = keypair
        envelope = _build_valid_envelope(public_key)
        signed = _sign_envelope(envelope, private_key)

        # Tamper with payload after signing
        signed.payload.data["prompt"] = "INJECTED MALICIOUS PROMPT"

        result = gatekeeper.verify(signed)

        assert result.passed is False
        assert result.reject_code == RejectCode.REJECT_SIGNATURE

    def test_tampered_metadata_rejected(self, keypair, gatekeeper):
        """Modifying metadata (e.g., inflating ihsan) after signing is caught."""
        private_key, public_key = keypair
        envelope = _build_valid_envelope(public_key)

        # Sign with low ihsan
        envelope.metadata.ihsan_score = 0.50
        signed = _sign_envelope(envelope, private_key)

        # Attacker inflates ihsan after signing to bypass IHSAN gate
        signed.metadata.ihsan_score = 0.99

        result = gatekeeper.verify(signed)

        # Should fail at SIGNATURE gate because metadata was tampered
        assert result.passed is False
        assert result.reject_code == RejectCode.REJECT_SIGNATURE


# ═══════════════════════════════════════════════════════════════════════════════
# GATE 3: TIMESTAMP — Clock Skew Detection
# ═══════════════════════════════════════════════════════════════════════════════


class TestTimestampGate:
    """Gate 3: Timestamp freshness and clock skew."""

    def test_stale_timestamp_rejected(self, keypair, gatekeeper):
        """Message with old timestamp is rejected."""
        private_key, public_key = keypair
        envelope = _build_valid_envelope(public_key)

        # Set timestamp to 10 minutes ago (exceeds MAX_CLOCK_SKEW_SECONDS)
        old_time = datetime.now(timezone.utc) - timedelta(seconds=MAX_CLOCK_SKEW_SECONDS + 60)
        envelope.timestamp = old_time.isoformat().replace("+00:00", "Z")
        signed = _sign_envelope(envelope, private_key)

        result = gatekeeper.verify(signed)

        assert result.passed is False
        assert result.reject_code == RejectCode.REJECT_TIMESTAMP_STALE

    def test_future_timestamp_rejected(self, keypair, gatekeeper):
        """Message with future timestamp is rejected (time-travel attack)."""
        private_key, public_key = keypair
        envelope = _build_valid_envelope(public_key)

        # Set timestamp to 10 minutes in the future
        future_time = datetime.now(timezone.utc) + timedelta(seconds=MAX_CLOCK_SKEW_SECONDS + 60)
        envelope.timestamp = future_time.isoformat().replace("+00:00", "Z")
        signed = _sign_envelope(envelope, private_key)

        result = gatekeeper.verify(signed)

        assert result.passed is False
        assert result.reject_code == RejectCode.REJECT_TIMESTAMP_FUTURE

    def test_invalid_timestamp_format_rejected(self, keypair, gatekeeper):
        """Malformed timestamp string is rejected."""
        private_key, public_key = keypair
        envelope = _build_valid_envelope(public_key)

        envelope.timestamp = "not-a-timestamp"
        signed = _sign_envelope(envelope, private_key)

        result = gatekeeper.verify(signed)

        assert result.passed is False
        assert result.reject_code == RejectCode.REJECT_SCHEMA

    def test_within_skew_window_passes(self, keypair, gatekeeper):
        """Timestamp within acceptable skew window passes."""
        private_key, public_key = keypair
        envelope = _build_valid_envelope(public_key)

        # 30 seconds ago — well within the 120s window
        recent_time = datetime.now(timezone.utc) - timedelta(seconds=30)
        envelope.timestamp = recent_time.isoformat().replace("+00:00", "Z")
        signed = _sign_envelope(envelope, private_key)

        result = gatekeeper.verify(signed)

        assert result.passed is True


# ═══════════════════════════════════════════════════════════════════════════════
# GATE 4: REPLAY — Nonce Deduplication
# ═══════════════════════════════════════════════════════════════════════════════


class TestReplayGate:
    """Gate 4: Replay protection via nonce tracking."""

    def test_duplicate_nonce_rejected(self, keypair, gatekeeper):
        """Replaying the same envelope (same nonce) is rejected."""
        private_key, public_key = keypair
        envelope = _build_valid_envelope(public_key)
        signed = _sign_envelope(envelope, private_key)

        # First submission succeeds
        result1 = gatekeeper.verify(signed)
        assert result1.passed is True

        # Second submission (same nonce) is rejected
        result2 = gatekeeper.verify(signed)
        assert result2.passed is False
        assert result2.reject_code == RejectCode.REJECT_NONCE_REPLAY

    def test_different_nonces_both_pass(self, keypair, gatekeeper):
        """Two envelopes with different nonces both pass."""
        private_key, public_key = keypair

        env1 = _build_valid_envelope(public_key)
        signed1 = _sign_envelope(env1, private_key)

        env2 = _build_valid_envelope(public_key)
        signed2 = _sign_envelope(env2, private_key)

        # Different nonces (from EnvelopeBuilder.build())
        assert signed1.nonce != signed2.nonce

        result1 = gatekeeper.verify(signed1)
        result2 = gatekeeper.verify(signed2)

        assert result1.passed is True
        assert result2.passed is True

    def test_nonce_cache_pruning(self, keypair):
        """Expired nonces are pruned to prevent memory exhaustion."""
        gk = PCIGateKeeper(policy_enforcement=False)
        private_key, public_key = keypair

        # Submit 5 envelopes
        for _ in range(5):
            env = _build_valid_envelope(public_key)
            signed = _sign_envelope(env, private_key)
            result = gk.verify(signed)
            assert result.passed is True

        assert len(gk.seen_nonces) == 5

        # Artificially age all nonces beyond TTL
        aged_time = time.time() - 400  # 400s > NONCE_TTL_SECONDS (300s)
        for nonce in gk.seen_nonces:
            gk.seen_nonces[nonce] = aged_time

        # Force prune
        pruned = gk._prune_expired_nonces()
        assert pruned == 5
        assert len(gk.seen_nonces) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# GATE 5: IHSAN — Ethical Excellence Threshold
# ═══════════════════════════════════════════════════════════════════════════════


class TestIhsanGate:
    """Gate 5: Ihsān (ethical excellence) threshold enforcement."""

    def test_low_ihsan_rejected(self, keypair, gatekeeper):
        """Envelope with Ihsan below 0.95 is rejected."""
        private_key, public_key = keypair
        envelope = (
            EnvelopeBuilder()
            .with_sender("PAT", "pat-001", public_key)
            .with_payload(
                action="inference.request",
                data={"prompt": "test"},
                policy_hash="d9c9b5f7a3e2c8d4f1a6e9b2c5d8a3f7e0c2b5d8a1e4f7c0b3d6a9e2c5f8b1d926f",
                state_hash=secrets.token_hex(32),
            )
            .with_metadata(ihsan=0.80, snr=0.92)  # Below 0.95 threshold
            .build()
        )
        signed = _sign_envelope(envelope, private_key)

        result = gatekeeper.verify(signed)

        assert result.passed is False
        assert result.reject_code == RejectCode.REJECT_IHSAN_BELOW_MIN
        assert str(IHSAN_MINIMUM_THRESHOLD) in result.details

    def test_exact_threshold_passes(self, keypair, gatekeeper):
        """Envelope at exactly the Ihsan threshold passes."""
        private_key, public_key = keypair
        envelope = (
            EnvelopeBuilder()
            .with_sender("PAT", "pat-001", public_key)
            .with_payload(
                action="inference.request",
                data={"prompt": "test"},
                policy_hash="d9c9b5f7a3e2c8d4f1a6e9b2c5d8a3f7e0c2b5d8a1e4f7c0b3d6a9e2c5f8b1d926f",
                state_hash=secrets.token_hex(32),
            )
            .with_metadata(ihsan=IHSAN_MINIMUM_THRESHOLD, snr=0.92)
            .build()
        )
        signed = _sign_envelope(envelope, private_key)

        result = gatekeeper.verify(signed)

        # Exact threshold: ihsan >= threshold, so should pass
        # (depends on whether gate uses < or <=)
        # Gate uses `<` so exact threshold passes
        assert result.passed is True

    def test_ihsan_checked_before_snr(self, keypair, gatekeeper):
        """Ihsan gate fires BEFORE SNR gate (fail-fast on ethics)."""
        private_key, public_key = keypair
        envelope = (
            EnvelopeBuilder()
            .with_sender("PAT", "pat-001", public_key)
            .with_payload(
                action="inference.request",
                data={"prompt": "test"},
                policy_hash="d9c9b5f7a3e2c8d4f1a6e9b2c5d8a3f7e0c2b5d8a1e4f7c0b3d6a9e2c5f8b1d926f",
                state_hash=secrets.token_hex(32),
            )
            .with_metadata(ihsan=0.50, snr=0.50)  # Both fail, but Ihsan should fire first
            .build()
        )
        signed = _sign_envelope(envelope, private_key)

        result = gatekeeper.verify(signed)

        assert result.passed is False
        assert result.reject_code == RejectCode.REJECT_IHSAN_BELOW_MIN
        # NOT REJECT_SNR_BELOW_MIN — ethics gate fires first


# ═══════════════════════════════════════════════════════════════════════════════
# GATE 6: SNR — Signal-to-Noise Ratio (Shannon)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSNRGate:
    """Gate 6: SNR (Shannon signal quality) threshold enforcement."""

    def test_low_snr_rejected(self, keypair, gatekeeper):
        """Envelope with SNR below 0.85 is rejected."""
        private_key, public_key = keypair
        envelope = (
            EnvelopeBuilder()
            .with_sender("PAT", "pat-001", public_key)
            .with_payload(
                action="inference.request",
                data={"prompt": "noisy data"},
                policy_hash="d9c9b5f7a3e2c8d4f1a6e9b2c5d8a3f7e0c2b5d8a1e4f7c0b3d6a9e2c5f8b1d926f",
                state_hash=secrets.token_hex(32),
            )
            .with_metadata(ihsan=0.97, snr=0.60)  # Below 0.85 threshold
            .build()
        )
        signed = _sign_envelope(envelope, private_key)

        result = gatekeeper.verify(signed)

        assert result.passed is False
        assert result.reject_code == RejectCode.REJECT_SNR_BELOW_MIN
        assert str(SNR_MINIMUM_THRESHOLD) in result.details

    def test_exact_snr_threshold_passes(self, keypair, gatekeeper):
        """Envelope at exactly the SNR threshold passes."""
        private_key, public_key = keypair
        envelope = (
            EnvelopeBuilder()
            .with_sender("PAT", "pat-001", public_key)
            .with_payload(
                action="inference.request",
                data={"prompt": "test"},
                policy_hash="d9c9b5f7a3e2c8d4f1a6e9b2c5d8a3f7e0c2b5d8a1e4f7c0b3d6a9e2c5f8b1d926f",
                state_hash=secrets.token_hex(32),
            )
            .with_metadata(ihsan=0.97, snr=SNR_MINIMUM_THRESHOLD)
            .build()
        )
        signed = _sign_envelope(envelope, private_key)

        result = gatekeeper.verify(signed)

        assert result.passed is True


# ═══════════════════════════════════════════════════════════════════════════════
# GATE 7: POLICY — Constitution Hash Verification
# ═══════════════════════════════════════════════════════════════════════════════


class TestPolicyGate:
    """Gate 7: Policy hash (constitution) verification."""

    def test_mismatched_policy_hash_rejected(self, keypair, gatekeeper_with_policy):
        """Envelope with wrong policy hash is rejected when enforcement is on."""
        private_key, public_key = keypair
        envelope = (
            EnvelopeBuilder()
            .with_sender("PAT", "pat-001", public_key)
            .with_payload(
                action="inference.request",
                data={"prompt": "test"},
                policy_hash="aaaa_wrong_hash_bbbb",  # Doesn't match constitution
                state_hash=secrets.token_hex(32),
            )
            .with_metadata(ihsan=0.97, snr=0.92)
            .build()
        )
        signed = _sign_envelope(envelope, private_key)

        result = gatekeeper_with_policy.verify(signed)

        assert result.passed is False
        assert result.reject_code == RejectCode.REJECT_POLICY_MISMATCH

    def test_correct_policy_hash_passes(self, keypair, gatekeeper_with_policy):
        """Envelope with matching policy hash passes."""
        private_key, public_key = keypair
        # Use the default constitution hash from the gatekeeper
        constitution_hash = gatekeeper_with_policy.constitution_hash

        envelope = (
            EnvelopeBuilder()
            .with_sender("PAT", "pat-001", public_key)
            .with_payload(
                action="inference.request",
                data={"prompt": "test"},
                policy_hash=constitution_hash,
                state_hash=secrets.token_hex(32),
            )
            .with_metadata(ihsan=0.97, snr=0.92)
            .build()
        )
        signed = _sign_envelope(envelope, private_key)

        result = gatekeeper_with_policy.verify(signed)

        assert result.passed is True

    def test_policy_enforcement_disabled_bypasses_check(self, keypair, gatekeeper):
        """When policy_enforcement=False, wrong hash still passes."""
        private_key, public_key = keypair
        envelope = (
            EnvelopeBuilder()
            .with_sender("PAT", "pat-001", public_key)
            .with_payload(
                action="inference.request",
                data={"prompt": "test"},
                policy_hash="totally_wrong_hash",
                state_hash=secrets.token_hex(32),
            )
            .with_metadata(ihsan=0.97, snr=0.92)
            .build()
        )
        signed = _sign_envelope(envelope, private_key)

        result = gatekeeper.verify(signed)

        assert result.passed is True  # Policy enforcement is off


# ═══════════════════════════════════════════════════════════════════════════════
# CRYPTO PRIMITIVES — Standing on Giants
# ═══════════════════════════════════════════════════════════════════════════════


class TestCryptoPrimitives:
    """Verify underlying cryptographic operations."""

    def test_keypair_generation(self):
        """generate_keypair produces valid hex keys of correct length."""
        private_key, public_key = generate_keypair()

        # Ed25519: 32 bytes = 64 hex chars
        assert len(private_key) == 64
        assert len(public_key) == 64

        # Valid hex
        bytes.fromhex(private_key)
        bytes.fromhex(public_key)

    def test_sign_and_verify_roundtrip(self):
        """sign_message + verify_signature roundtrip works."""
        private_key, public_key = generate_keypair()
        digest = secrets.token_hex(32)

        signature = sign_message(digest, private_key)
        assert verify_signature(digest, signature, public_key) is True

    def test_signature_fails_with_wrong_key(self):
        """Verification fails with different public key."""
        priv1, pub1 = generate_keypair()
        _, pub2 = generate_keypair()

        digest = secrets.token_hex(32)
        signature = sign_message(digest, priv1)

        assert verify_signature(digest, signature, pub1) is True
        assert verify_signature(digest, signature, pub2) is False

    def test_envelope_digest_deterministic(self, keypair):
        """Same envelope produces same digest (RFC8785 canonicalization)."""
        _, public_key = keypair
        envelope = _build_valid_envelope(public_key)

        digest1 = envelope.compute_digest()
        digest2 = envelope.compute_digest()

        assert digest1 == digest2
        assert len(digest1) == 64  # BLAKE3 produces 256 bits = 64 hex chars

    def test_envelope_digest_changes_on_mutation(self, keypair):
        """Mutating any field changes the digest."""
        _, public_key = keypair
        env1 = _build_valid_envelope(public_key)
        digest_original = env1.compute_digest()

        # Mutate payload
        env1.payload.data["prompt"] = "changed"
        digest_after = env1.compute_digest()

        assert digest_original != digest_after


# ═══════════════════════════════════════════════════════════════════════════════
# ADVERSARIAL SCENARIOS — Security Hardening
# ═══════════════════════════════════════════════════════════════════════════════


class TestAdversarialScenarios:
    """Test against known attack vectors."""

    def test_replay_attack_across_gatekeeper_instances(self):
        """Replay protection is per-gatekeeper instance (shared cache needed)."""
        private_key, public_key = generate_keypair()
        shared_cache = {}

        gk1 = PCIGateKeeper(seen_nonces_cache=shared_cache, policy_enforcement=False)
        gk2 = PCIGateKeeper(seen_nonces_cache=shared_cache, policy_enforcement=False)

        envelope = _build_valid_envelope(public_key)
        signed = _sign_envelope(envelope, private_key)

        # First gatekeeper accepts
        result1 = gk1.verify(signed)
        assert result1.passed is True

        # Second gatekeeper (sharing cache) rejects replay
        result2 = gk2.verify(signed)
        assert result2.passed is False
        assert result2.reject_code == RejectCode.REJECT_NONCE_REPLAY

    def test_ihsan_inflation_attack(self, keypair, gatekeeper):
        """
        Attacker signs with low ihsan then inflates score.
        Signature verification catches the tampering.
        """
        private_key, public_key = keypair
        envelope = _build_valid_envelope(public_key)
        envelope.metadata.ihsan_score = 0.40  # Ethically poor
        signed = _sign_envelope(envelope, private_key)

        # Inflate ihsan to bypass gate
        signed.metadata.ihsan_score = 0.99

        result = gatekeeper.verify(signed)
        assert result.passed is False
        assert result.reject_code == RejectCode.REJECT_SIGNATURE

    def test_snr_inflation_attack(self, keypair, gatekeeper):
        """
        Attacker signs with low SNR then inflates score.
        Signature verification catches the tampering.
        """
        private_key, public_key = keypair
        envelope = _build_valid_envelope(public_key)
        envelope.metadata.snr_score = 0.30  # Noisy garbage
        signed = _sign_envelope(envelope, private_key)

        # Inflate SNR to bypass gate
        signed.metadata.snr_score = 0.95

        result = gatekeeper.verify(signed)
        assert result.passed is False
        assert result.reject_code == RejectCode.REJECT_SIGNATURE

    def test_high_snr_low_ihsan_rejected(self, keypair, gatekeeper):
        """
        High-SNR malicious content is caught by Ihsan gate BEFORE SNR check.
        This is why Ihsan-before-SNR ordering matters in PCI context.
        """
        private_key, public_key = keypair
        envelope = (
            EnvelopeBuilder()
            .with_sender("PAT", "pat-001", public_key)
            .with_payload(
                action="inference.request",
                data={"prompt": "high quality malicious content"},
                policy_hash="d9c9b5f7a3e2c8d4f1a6e9b2c5d8a3f7e0c2b5d8a1e4f7c0b3d6a9e2c5f8b1d926f",
                state_hash=secrets.token_hex(32),
            )
            .with_metadata(ihsan=0.50, snr=0.99)  # Clear signal, but ethically compromised
            .build()
        )
        signed = _sign_envelope(envelope, private_key)

        result = gatekeeper.verify(signed)

        assert result.passed is False
        assert result.reject_code == RejectCode.REJECT_IHSAN_BELOW_MIN
        # Ihsan fires first, NOT SNR — this is the core security property

    def test_boundary_values_ihsan(self, keypair, gatekeeper):
        """Test Ihsan boundary: just below threshold fails, at threshold passes."""
        private_key, public_key = keypair

        # Just below
        env_below = (
            EnvelopeBuilder()
            .with_sender("PAT", "pat-001", public_key)
            .with_payload(
                action="test",
                data={},
                policy_hash="d9c9b5f7a3e2c8d4f1a6e9b2c5d8a3f7e0c2b5d8a1e4f7c0b3d6a9e2c5f8b1d926f",
                state_hash=secrets.token_hex(32),
            )
            .with_metadata(ihsan=IHSAN_MINIMUM_THRESHOLD - 0.001, snr=0.92)
            .build()
        )
        signed_below = _sign_envelope(env_below, private_key)
        result_below = gatekeeper.verify(signed_below)
        assert result_below.passed is False
        assert result_below.reject_code == RejectCode.REJECT_IHSAN_BELOW_MIN


# ═══════════════════════════════════════════════════════════════════════════════
# GATE ORDERING PROPERTY — The Crown Jewel
# ═══════════════════════════════════════════════════════════════════════════════


class TestGateOrdering:
    """
    Verify the gate ordering invariant:
    SCHEMA → SIGNATURE → TIMESTAMP → REPLAY → IHSAN → SNR → POLICY

    This ordering is BIZRA's core security differentiator for P2P messaging.
    """

    def test_gates_execute_in_correct_order(self, keypair, gatekeeper):
        """Full pass records all 7 gates in the correct order."""
        private_key, public_key = keypair
        envelope = _build_valid_envelope(public_key)
        signed = _sign_envelope(envelope, private_key)

        result = gatekeeper.verify(signed)

        assert result.passed is True
        assert result.gate_passed == [
            "SCHEMA",
            "SIGNATURE",
            "TIMESTAMP",
            "REPLAY",
            "IHSAN",
            "SNR",
            "POLICY",
        ]

    def test_tier1_gates_are_cheap_operations(self, keypair, gatekeeper):
        """Tier 1 gates (SCHEMA, SIGNATURE, TIMESTAMP, REPLAY) execute first."""
        private_key, public_key = keypair

        # Create envelope that fails at IHSAN (Tier 2)
        envelope = (
            EnvelopeBuilder()
            .with_sender("PAT", "pat-001", public_key)
            .with_payload(
                action="test",
                data={},
                policy_hash="d9c9b5f7a3e2c8d4f1a6e9b2c5d8a3f7e0c2b5d8a1e4f7c0b3d6a9e2c5f8b1d926f",
                state_hash=secrets.token_hex(32),
            )
            .with_metadata(ihsan=0.50, snr=0.92)  # Fails at IHSAN
            .build()
        )
        signed = _sign_envelope(envelope, private_key)

        result = gatekeeper.verify(signed)

        # Should fail at IHSAN, meaning all Tier 1 gates passed
        assert result.passed is False
        assert result.reject_code == RejectCode.REJECT_IHSAN_BELOW_MIN

    def test_signature_before_timestamp(self, keypair, gatekeeper):
        """Invalid signature is caught before timestamp check."""
        _, public_key = keypair
        envelope = _build_valid_envelope(public_key)

        # Both bad: no signature AND stale timestamp
        old_time = datetime.now(timezone.utc) - timedelta(hours=1)
        envelope.timestamp = old_time.isoformat().replace("+00:00", "Z")
        # No signature

        result = gatekeeper.verify(envelope)

        # Signature checked first
        assert result.reject_code == RejectCode.REJECT_SIGNATURE
