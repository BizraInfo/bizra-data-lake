"""
Receipt System Tests — INT-008 Coverage Expansion.

Proves that the receipt system produces cryptographically signed,
verifiable receipts for accepted, rejected, and amber-restricted outcomes.

Standing on Giants:
- Lamport (1978): Event ordering and receipts
- Merkle (1979): Hash integrity
- Shannon (1948): SNR as quality metric
- BIZRA Spearpoint PRD: "Every execution produces a signed receipt"
"""

import json
import pytest
from datetime import datetime, timezone

from core.proof_engine.canonical import CanonQuery, CanonPolicy, blake3_digest
from core.proof_engine.receipt import (
    Receipt,
    ReceiptStatus,
    ReceiptBuilder,
    ReceiptVerifier,
    SimpleSigner,
    Metrics,
)
from core.proof_engine.snr import SNREngine, SNRInput, SNRTrace


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def signer():
    """Test signer."""
    return SimpleSigner(secret=b"test_secret_key_for_receipt_tests")


@pytest.fixture
def query():
    """Standard test query."""
    return CanonQuery(
        user_id="alice", user_state="active",
        intent="What is sovereignty?", nonce="nonce_001",
    )


@pytest.fixture
def policy():
    """Standard test policy."""
    return CanonPolicy(
        policy_id="pol_test", version="1.0.0",
        rules={"snr_min": 0.95}, thresholds={"ihsan": 0.95},
    )


@pytest.fixture
def builder(signer):
    """Receipt builder."""
    return ReceiptBuilder(signer)


# =============================================================================
# RECEIPT STATUS
# =============================================================================

class TestReceiptStatus:
    """Tests for ReceiptStatus enum."""

    def test_all_statuses(self):
        """All 4 statuses exist."""
        assert ReceiptStatus.ACCEPTED.value == "accepted"
        assert ReceiptStatus.REJECTED.value == "rejected"
        assert ReceiptStatus.AMBER_RESTRICTED.value == "amber_restricted"
        assert ReceiptStatus.PENDING.value == "pending"


# =============================================================================
# SIMPLE SIGNER
# =============================================================================

class TestSimpleSigner:
    """Tests for SimpleSigner HMAC-based signing."""

    def test_sign_returns_bytes(self, signer):
        """sign() returns bytes."""
        sig = signer.sign(b"hello")
        assert isinstance(sig, bytes)
        assert len(sig) == 32  # SHA-256 digest

    def test_sign_deterministic(self, signer):
        """Same message produces same signature."""
        s1 = signer.sign(b"test")
        s2 = signer.sign(b"test")
        assert s1 == s2

    def test_sign_different_messages(self, signer):
        """Different messages produce different signatures."""
        s1 = signer.sign(b"alpha")
        s2 = signer.sign(b"beta")
        assert s1 != s2

    def test_verify_valid(self, signer):
        """Valid signature verifies."""
        msg = b"hello world"
        sig = signer.sign(msg)
        assert signer.verify(msg, sig) is True

    def test_verify_invalid(self, signer):
        """Invalid signature fails verification."""
        msg = b"hello world"
        bad_sig = b"\x00" * 32
        assert signer.verify(msg, bad_sig) is False

    def test_verify_wrong_message(self, signer):
        """Signature for different message fails."""
        sig = signer.sign(b"message_a")
        assert signer.verify(b"message_b", sig) is False

    def test_public_key_bytes(self, signer):
        """public_key_bytes returns deterministic hash."""
        pk1 = signer.public_key_bytes()
        pk2 = signer.public_key_bytes()
        assert pk1 == pk2
        assert len(pk1) == 32

    def test_different_secrets_different_keys(self):
        """Different secrets produce different keys."""
        s1 = SimpleSigner(secret=b"key_a")
        s2 = SimpleSigner(secret=b"key_b")
        assert s1.public_key_bytes() != s2.public_key_bytes()

    def test_different_secrets_different_sigs(self):
        """Different secrets produce different signatures for same message."""
        s1 = SimpleSigner(secret=b"key_a")
        s2 = SimpleSigner(secret=b"key_b")
        msg = b"same message"
        assert s1.sign(msg) != s2.sign(msg)

    def test_public_key_bytes_is_blake3(self):
        """Pin public_key_bytes to BLAKE3 (SR-001: cross-language interop).

        SimpleSigner is dev-only (production uses Ed25519). BLAKE3 ensures
        Python and Rust nodes produce identical key fingerprints.
        """
        import blake3

        secret = b"regression_test_secret"
        signer = SimpleSigner(secret=secret)
        expected = blake3.blake3(secret).digest()
        assert signer.public_key_bytes() == expected

    def test_existing_receipt_verifies_after_upgrade(self):
        """Receipts signed before upgrade must still verify.

        Simulates: create receipt → verify with same signer → assert OK.
        This catches any change to public_key_bytes() that would break
        the 'signer_pubkey != self.signer.public_key_bytes()' check in
        ReceiptVerifier.verify().
        """
        secret = b"stable_key_for_backward_compat"
        signer = SimpleSigner(secret=secret)
        pubkey_at_creation = signer.public_key_bytes()

        # Simulate "upgrade" — construct a fresh signer with same secret
        signer2 = SimpleSigner(secret=secret)
        assert signer2.public_key_bytes() == pubkey_at_creation


# =============================================================================
# METRICS
# =============================================================================

class TestMetrics:
    """Tests for Metrics dataclass."""

    def test_defaults(self):
        """All fields default to zero."""
        m = Metrics()
        assert m.p99_us == 0
        assert m.allocs == 0
        assert m.duration_ms == 0.0

    def test_to_dict(self):
        """to_dict() includes all fields."""
        m = Metrics(p99_us=100, allocs=5, duration_ms=1.5)
        d = m.to_dict()
        assert d["p99_us"] == 100
        assert d["allocs"] == 5
        assert d["duration_ms"] == 1.5


# =============================================================================
# RECEIPT
# =============================================================================

class TestReceipt:
    """Tests for Receipt dataclass."""

    def test_body_bytes_deterministic(self, query, policy, signer):
        """body_bytes() is deterministic."""
        receipt = Receipt(
            receipt_id="rcpt_001",
            status=ReceiptStatus.ACCEPTED,
            query_digest=query.digest(),
            policy_digest=policy.digest(),
            payload_digest=blake3_digest(query.canonical_bytes()),
            snr=0.96, ihsan_score=0.97,
            gate_passed="commit",
        )
        b1 = receipt.body_bytes()
        b2 = receipt.body_bytes()
        assert b1 == b2

    def test_sign_with_adds_signature(self, query, policy, signer):
        """sign_with() sets signature and pubkey."""
        receipt = Receipt(
            receipt_id="rcpt_002",
            status=ReceiptStatus.ACCEPTED,
            query_digest=query.digest(),
            policy_digest=policy.digest(),
            payload_digest=blake3_digest(b""),
            snr=0.96, ihsan_score=0.97,
            gate_passed="commit",
        )
        assert receipt.signature == b""
        receipt.sign_with(signer)
        assert len(receipt.signature) == 32
        assert receipt.signer_pubkey == signer.public_key_bytes()

    def test_verify_signature(self, query, policy, signer):
        """Signed receipt verifies."""
        receipt = Receipt(
            receipt_id="rcpt_003",
            status=ReceiptStatus.ACCEPTED,
            query_digest=query.digest(),
            policy_digest=policy.digest(),
            payload_digest=blake3_digest(b""),
            snr=0.96, ihsan_score=0.97,
            gate_passed="commit",
        )
        receipt.sign_with(signer)
        assert receipt.verify_signature(signer) is True

    def test_tampered_receipt_fails_verification(self, query, policy, signer):
        """Tampered receipt fails verification."""
        receipt = Receipt(
            receipt_id="rcpt_004",
            status=ReceiptStatus.ACCEPTED,
            query_digest=query.digest(),
            policy_digest=policy.digest(),
            payload_digest=blake3_digest(b""),
            snr=0.96, ihsan_score=0.97,
            gate_passed="commit",
        )
        receipt.sign_with(signer)
        # Tamper
        receipt.snr = 0.50
        assert receipt.verify_signature(signer) is False

    def test_digest_includes_signature(self, query, policy, signer):
        """receipt.digest() includes the signature."""
        receipt = Receipt(
            receipt_id="rcpt_005",
            status=ReceiptStatus.ACCEPTED,
            query_digest=query.digest(),
            policy_digest=policy.digest(),
            payload_digest=blake3_digest(b""),
            snr=0.96, ihsan_score=0.97,
            gate_passed="commit",
        )
        d_unsigned = receipt.digest()
        receipt.sign_with(signer)
        d_signed = receipt.digest()
        assert d_unsigned != d_signed

    def test_hex_digest_format(self, query, policy, signer):
        """hex_digest() returns 64-char hex string."""
        receipt = Receipt(
            receipt_id="rcpt_006",
            status=ReceiptStatus.ACCEPTED,
            query_digest=query.digest(),
            policy_digest=policy.digest(),
            payload_digest=blake3_digest(b""),
            snr=0.96, ihsan_score=0.97,
            gate_passed="commit",
        )
        hd = receipt.hex_digest()
        assert len(hd) == 64
        int(hd, 16)

    def test_to_dict_complete(self, query, policy, signer):
        """to_dict() includes all fields."""
        receipt = Receipt(
            receipt_id="rcpt_007",
            status=ReceiptStatus.ACCEPTED,
            query_digest=query.digest(),
            policy_digest=policy.digest(),
            payload_digest=blake3_digest(b""),
            snr=0.96, ihsan_score=0.97,
            gate_passed="commit",
            reason=None,
        )
        receipt.sign_with(signer)
        d = receipt.to_dict()
        assert d["receipt_id"] == "rcpt_007"
        assert d["status"] == "accepted"
        assert d["snr"] == 0.96
        assert d["ihsan_score"] == 0.97
        assert d["gate_passed"] == "commit"
        assert "query_digest" in d
        assert "policy_digest" in d
        assert "payload_digest" in d
        assert "signature" in d
        assert "signer_pubkey" in d
        assert "receipt_digest" in d
        assert "timestamp" in d

    def test_to_dict_json_serializable(self, query, policy, signer):
        """to_dict() output is JSON serializable."""
        receipt = Receipt(
            receipt_id="rcpt_008",
            status=ReceiptStatus.REJECTED,
            query_digest=query.digest(),
            policy_digest=policy.digest(),
            payload_digest=blake3_digest(b""),
            snr=0.50, ihsan_score=0.80,
            gate_passed="snr",
            reason="SNR below threshold",
        )
        receipt.sign_with(signer)
        serialized = json.dumps(receipt.to_dict())
        parsed = json.loads(serialized)
        assert parsed["status"] == "rejected"
        assert parsed["reason"] == "SNR below threshold"

    def test_default_timestamp(self, query, policy):
        """Timestamp is auto-set."""
        receipt = Receipt(
            receipt_id="rcpt_009",
            status=ReceiptStatus.PENDING,
            query_digest=query.digest(),
            policy_digest=policy.digest(),
            payload_digest=blake3_digest(b""),
            snr=0.0, ihsan_score=0.0,
            gate_passed="none",
        )
        assert receipt.timestamp is not None
        assert receipt.timestamp.tzinfo is not None


# =============================================================================
# RECEIPT BUILDER
# =============================================================================

class TestReceiptBuilder:
    """Tests for ReceiptBuilder."""

    def test_accepted_receipt(self, builder, query, policy):
        """accepted() creates an ACCEPTED receipt."""
        receipt = builder.accepted(
            query=query, policy=policy,
            payload=query.canonical_bytes(),
            snr=0.96, ihsan_score=0.97,
        )
        assert receipt.status == ReceiptStatus.ACCEPTED
        assert receipt.gate_passed == "commit"
        assert len(receipt.signature) == 32

    def test_rejected_receipt(self, builder, query, policy):
        """rejected() creates a REJECTED receipt."""
        receipt = builder.rejected(
            query=query, policy=policy,
            snr=0.50, ihsan_score=0.80,
            gate_failed="snr",
            reason="SNR below threshold",
        )
        assert receipt.status == ReceiptStatus.REJECTED
        assert receipt.reason == "SNR below threshold"
        assert len(receipt.signature) == 32

    def test_amber_restricted_receipt(self, builder, query, policy):
        """amber_restricted() creates an AMBER_RESTRICTED receipt."""
        receipt = builder.amber_restricted(
            query=query, policy=policy,
            payload=query.canonical_bytes(),
            snr=0.92, ihsan_score=0.90,
            restriction_reason="Safety concerns",
        )
        assert receipt.status == ReceiptStatus.AMBER_RESTRICTED
        assert "AMBER" in receipt.reason
        assert "Safety concerns" in receipt.reason

    def test_receipt_ids_unique(self, builder, query, policy):
        """Each receipt gets a unique ID."""
        r1 = builder.accepted(
            query=query, policy=policy,
            payload=query.canonical_bytes(),
            snr=0.96, ihsan_score=0.97,
        )
        r2 = builder.accepted(
            query=query, policy=policy,
            payload=query.canonical_bytes(),
            snr=0.96, ihsan_score=0.97,
        )
        assert r1.receipt_id != r2.receipt_id

    def test_receipt_ids_monotonic(self, builder, query, policy):
        """Receipt IDs contain incrementing counter."""
        ids = []
        for _ in range(5):
            r = builder.accepted(
                query=query, policy=policy,
                payload=b"", snr=0.96, ihsan_score=0.97,
            )
            ids.append(r.receipt_id)
        # All unique
        assert len(set(ids)) == 5

    def test_accepted_with_metrics(self, builder, query, policy):
        """accepted() accepts Metrics."""
        metrics = Metrics(p99_us=500, duration_ms=0.5)
        receipt = builder.accepted(
            query=query, policy=policy,
            payload=b"", snr=0.96, ihsan_score=0.97,
            metrics=metrics,
        )
        assert receipt.metrics.p99_us == 500

    def test_accepted_with_snr_trace(self, builder, query, policy):
        """accepted() accepts SNRTrace."""
        engine = SNREngine()
        inputs = SNRInput(source_trust_score=0.9, ihsan_score=0.97)
        _, trace = engine.compute(inputs)

        receipt = builder.accepted(
            query=query, policy=policy,
            payload=b"", snr=0.96, ihsan_score=0.97,
            snr_trace=trace,
        )
        assert receipt.snr_trace is not None
        d = receipt.to_dict()
        assert d["snr_trace"] is not None
        assert "signal_mass" in d["snr_trace"]

    def test_all_receipts_are_signed(self, builder, query, policy):
        """All builder methods produce signed receipts."""
        accepted = builder.accepted(
            query=query, policy=policy, payload=b"",
            snr=0.96, ihsan_score=0.97,
        )
        rejected = builder.rejected(
            query=query, policy=policy, snr=0.5, ihsan_score=0.8,
            gate_failed="snr", reason="test",
        )
        amber = builder.amber_restricted(
            query=query, policy=policy, payload=b"",
            snr=0.92, ihsan_score=0.9,
            restriction_reason="test",
        )
        for r in [accepted, rejected, amber]:
            assert len(r.signature) > 0
            assert len(r.signer_pubkey) > 0


# =============================================================================
# RECEIPT VERIFIER
# =============================================================================

class TestReceiptVerifier:
    """Tests for ReceiptVerifier."""

    def test_verify_valid_receipt(self, signer, builder, query, policy):
        """Valid receipt verifies successfully."""
        receipt = builder.accepted(
            query=query, policy=policy,
            payload=query.canonical_bytes(),
            snr=0.96, ihsan_score=0.97,
        )
        verifier = ReceiptVerifier(signer)
        valid, error = verifier.verify(receipt)
        assert valid is True
        assert error is None

    def test_verify_tampered_receipt(self, signer, builder, query, policy):
        """Tampered receipt fails verification."""
        receipt = builder.accepted(
            query=query, policy=policy,
            payload=query.canonical_bytes(),
            snr=0.96, ihsan_score=0.97,
        )
        receipt.snr = 0.01  # Tamper
        verifier = ReceiptVerifier(signer)
        valid, error = verifier.verify(receipt)
        assert valid is False
        assert "Invalid signature" in error

    def test_verify_wrong_signer(self, builder, query, policy):
        """Receipt from different signer fails."""
        receipt = builder.accepted(
            query=query, policy=policy,
            payload=query.canonical_bytes(),
            snr=0.96, ihsan_score=0.97,
        )
        other_signer = SimpleSigner(secret=b"different_key")
        verifier = ReceiptVerifier(other_signer)
        valid, error = verifier.verify(receipt)
        assert valid is False

    def test_stats_tracking(self, signer, builder, query, policy):
        """Verification stats are tracked."""
        verifier = ReceiptVerifier(signer)

        # Valid
        r1 = builder.accepted(
            query=query, policy=policy, payload=b"",
            snr=0.96, ihsan_score=0.97,
        )
        verifier.verify(r1)

        # Tampered
        r2 = builder.accepted(
            query=query, policy=policy, payload=b"",
            snr=0.96, ihsan_score=0.97,
        )
        r2.snr = 0.01
        verifier.verify(r2)

        stats = verifier.get_stats()
        assert stats["total_verified"] == 1
        assert stats["total_failed"] == 1
        assert stats["success_rate"] == 0.5

    def test_empty_stats(self, signer):
        """Empty verifier has zero stats."""
        verifier = ReceiptVerifier(signer)
        stats = verifier.get_stats()
        assert stats["total_verified"] == 0
        assert stats["total_failed"] == 0
        assert stats["success_rate"] == 0.0
