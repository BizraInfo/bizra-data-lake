"""
Property-Based Tests — Crypto Roundtrip Invariants
===================================================

Standing on Giants:
- QuickCheck (Claessen & Hughes, 2000) — Foundational PBT framework
- Hypothesis (MacIver, 2016) — Python PBT with shrinking & stateful testing
- Bernstein et al. (2012) — Ed25519: high-speed high-security signatures
- BLAKE3 (2020) — Fast cryptographic hash with domain separation

Invariants verified:
1. Ed25519 sign → verify roundtrip: ∀ (key, msg), verify(sign(msg, key)) == True
2. Ed25519 cross-key rejection: ∀ (k1, k2, msg), k1 ≠ k2 → verify_k2(sign_k1(msg)) == False
3. Signature determinism: ∀ (key, msg), sign(msg, key) == sign(msg, key)
4. Tamper detection: ∀ (key, msg, bit), flip(sig, bit) → verify fails
5. Domain-separated digest determinism & collision resistance
6. Timing-safe comparison reflexivity and symmetry
"""

import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from core.pci.crypto import (
    domain_separated_digest,
    generate_keypair,
    sign_message,
    verify_signature,
    timing_safe_compare,
    timing_safe_compare_hex,
)


# ── Strategies ──────────────────────────────────────────────────────────

# Arbitrary byte messages (1-1024 bytes)
message_bytes = st.binary(min_size=1, max_size=1024)

# Printable text messages
message_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=1,
    max_size=512,
)

# Valid hex strings (even length, 1-64 bytes → 2-128 hex chars)
hex_strings = st.binary(min_size=1, max_size=64).map(lambda b: b.hex())


# ── Ed25519 Roundtrip Invariants ────────────────────────────────────────

class TestEd25519Roundtrip:
    """Property: sign then verify always succeeds with the same keypair."""

    @given(data=message_bytes)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_sign_verify_roundtrip(self, data: bytes):
        """∀ (keypair, msg): verify(sign(msg, priv), pub) == True"""
        priv, pub = generate_keypair()
        digest = domain_separated_digest(data)
        sig = sign_message(digest, priv)
        assert verify_signature(digest, sig, pub), (
            f"Roundtrip failed: key={pub[:8]}... digest={digest[:16]}..."
        )

    @given(data=message_bytes)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_cross_key_rejection(self, data: bytes):
        """∀ (k1 ≠ k2, msg): verify_k2(sign_k1(msg)) == False"""
        priv1, pub1 = generate_keypair()
        priv2, pub2 = generate_keypair()
        assume(pub1 != pub2)

        digest = domain_separated_digest(data)
        sig = sign_message(digest, priv1)
        assert not verify_signature(digest, sig, pub2), (
            "Cross-key verification should NEVER succeed"
        )

    @given(data=message_bytes)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_signature_determinism(self, data: bytes):
        """∀ (key, msg): sign(msg, key) is deterministic (Ed25519 is deterministic)."""
        priv, pub = generate_keypair()
        digest = domain_separated_digest(data)
        sig1 = sign_message(digest, priv)
        sig2 = sign_message(digest, priv)
        assert sig1 == sig2, "Ed25519 signatures must be deterministic"

    @given(data=message_bytes, bit_pos=st.integers(min_value=0, max_value=511))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_tamper_detection(self, data: bytes, bit_pos: int):
        """∀ (key, msg, bit): flipping any bit in signature → verify fails."""
        priv, pub = generate_keypair()
        digest = domain_separated_digest(data)
        sig = sign_message(digest, priv)

        # Flip one bit in the signature
        sig_bytes = bytearray(bytes.fromhex(sig))
        byte_idx = bit_pos // 8
        if byte_idx < len(sig_bytes):
            sig_bytes[byte_idx] ^= 1 << (bit_pos % 8)
            tampered_sig = sig_bytes.hex()
            # Tampered signature must fail (unless the flip is a no-op, which is
            # impossible since we XOR a single bit)
            if tampered_sig != sig:
                assert not verify_signature(digest, tampered_sig, pub), (
                    "Tampered signature must be rejected"
                )


# ── Domain-Separated Digest Invariants ──────────────────────────────────

class TestDomainDigest:
    """Properties of the BLAKE3 domain-separated digest."""

    @given(data=message_bytes)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_digest_determinism(self, data: bytes):
        """∀ msg: digest(msg) == digest(msg)"""
        d1 = domain_separated_digest(data)
        d2 = domain_separated_digest(data)
        assert d1 == d2

    @given(data=message_bytes)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_digest_is_hex(self, data: bytes):
        """∀ msg: digest(msg) is valid hex string."""
        d = domain_separated_digest(data)
        bytes.fromhex(d)  # Will raise ValueError if not valid hex

    @given(a=message_bytes, b=message_bytes)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_digest_collision_resistance(self, a: bytes, b: bytes):
        """∀ a ≠ b: digest(a) ≠ digest(b) (probabilistically — BLAKE3 256-bit)."""
        assume(a != b)
        assert domain_separated_digest(a) != domain_separated_digest(b), (
            "BLAKE3 collision detected — this should be astronomically improbable"
        )


# ── Timing-Safe Compare Invariants ──────────────────────────────────────

class TestTimingSafeCompare:
    """Properties of constant-time comparison."""

    @given(data=st.binary(min_size=1, max_size=256))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_reflexivity(self, data: bytes):
        """∀ x: compare(x, x) == True"""
        assert timing_safe_compare(data, data)

    @given(a=st.binary(min_size=1, max_size=128), b=st.binary(min_size=1, max_size=128))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_inequality(self, a: bytes, b: bytes):
        """∀ a ≠ b: compare(a, b) == False"""
        assume(a != b)
        assert not timing_safe_compare(a, b)

    @given(data=hex_strings)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_hex_reflexivity(self, data: str):
        """∀ x: compare_hex(x, x) == True"""
        assert timing_safe_compare_hex(data, data)


# ── Keypair Generation Invariants ───────────────────────────────────────

class TestKeypairGeneration:
    """Properties of Ed25519 key generation."""

    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    @given(st.just(None))
    def test_keypair_uniqueness(self, _):
        """Each generate_keypair() call produces a unique key."""
        k1_priv, k1_pub = generate_keypair()
        k2_priv, k2_pub = generate_keypair()
        assert k1_priv != k2_priv, "Private keys must be unique"
        assert k1_pub != k2_pub, "Public keys must be unique"

    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    @given(st.just(None))
    def test_keypair_format(self, _):
        """Keys are 32-byte hex strings (64 hex chars)."""
        priv, pub = generate_keypair()
        assert len(priv) == 64, f"Private key hex length: {len(priv)}, expected 64"
        assert len(pub) == 64, f"Public key hex length: {len(pub)}, expected 64"
        bytes.fromhex(priv)  # Valid hex
        bytes.fromhex(pub)   # Valid hex
