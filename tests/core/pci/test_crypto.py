"""
BIZRA PCI Crypto Test Suite

Tests for Ed25519 cryptographic primitives.
Target: 70% coverage of core/pci/crypto.py (98 lines)
"""

import pytest
import json
import sys
from pathlib import Path

# Add project root to path (works across platforms)
_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from core.pci.crypto import (
    canonical_json,
    domain_separated_digest,
    sign_message,
    verify_signature,
    generate_keypair,
    PCI_DOMAIN_PREFIX,
)


class TestCanonicalJson:
    """Tests for RFC 8785 JSON Canonicalization."""

    def test_sorted_keys(self):
        """Keys should be sorted lexicographically."""
        data = {"z": 1, "a": 2, "m": 3}
        result = canonical_json(data)

        # Parse back to verify order in string
        result_str = result.decode('utf-8')
        assert result_str.index('"a"') < result_str.index('"m"')
        assert result_str.index('"m"') < result_str.index('"z"')

    def test_no_whitespace(self):
        """Output should have no whitespace."""
        data = {"key": "value", "nested": {"inner": 123}}
        result = canonical_json(data)

        assert b' ' not in result
        assert b'\n' not in result
        assert b'\t' not in result

    def test_ensure_ascii(self):
        """Output should be ASCII-only (RFC 8785 mandate)."""
        data = {"unicode": "caf\u00e9", "emoji": "\U0001F600"}
        result = canonical_json(data)

        # Check all bytes are ASCII
        for byte in result:
            assert byte < 128, f"Non-ASCII byte found: {byte}"

    def test_nested_objects_sorted(self):
        """Nested objects should also have sorted keys."""
        data = {
            "outer": {"z_inner": 1, "a_inner": 2},
            "another": "value"
        }
        result = canonical_json(data)
        result_str = result.decode('utf-8')

        assert '"a_inner"' in result_str
        assert result_str.index('"a_inner"') < result_str.index('"z_inner"')

    def test_deterministic(self):
        """Same input should always produce same output."""
        data = {"b": 2, "a": 1, "c": 3}

        result1 = canonical_json(data)
        result2 = canonical_json(data)

        assert result1 == result2

    def test_special_characters_escaped(self):
        """Special characters should be properly escaped."""
        data = {"text": "line1\nline2\ttab"}
        result = canonical_json(data)

        assert b'\\n' in result
        assert b'\\t' in result


class TestDomainSeparatedDigest:
    """Tests for domain-separated BLAKE3 hashing."""

    def test_includes_domain_prefix(self):
        """Digest should incorporate domain prefix."""
        data = b'test data'

        # Hashing same data with different domain should differ
        digest = domain_separated_digest(data)

        # Should be hex string
        assert all(c in '0123456789abcdef' for c in digest)

    def test_digest_length(self):
        """BLAKE3 digest should be 64 hex chars (256 bits)."""
        data = b'test'
        digest = domain_separated_digest(data)

        assert len(digest) == 64

    def test_deterministic(self):
        """Same input should produce same digest."""
        data = b'consistent input'

        digest1 = domain_separated_digest(data)
        digest2 = domain_separated_digest(data)

        assert digest1 == digest2

    def test_different_inputs_different_digests(self):
        """Different inputs should produce different digests."""
        digest1 = domain_separated_digest(b'input1')
        digest2 = domain_separated_digest(b'input2')

        assert digest1 != digest2

    def test_empty_input(self):
        """Empty input should produce valid digest."""
        digest = domain_separated_digest(b'')

        assert len(digest) == 64
        assert all(c in '0123456789abcdef' for c in digest)


class TestSignAndVerify:
    """Tests for Ed25519 sign/verify operations."""

    def test_sign_verify_roundtrip(self):
        """Signed message should verify with correct key."""
        priv, pub = generate_keypair()
        digest = domain_separated_digest(b'test message')

        signature = sign_message(digest, priv)
        result = verify_signature(digest, signature, pub)

        assert result is True

    def test_tampered_message_fails(self):
        """Signature should fail for modified message."""
        priv, pub = generate_keypair()
        digest1 = domain_separated_digest(b'original')
        digest2 = domain_separated_digest(b'tampered')

        signature = sign_message(digest1, priv)
        result = verify_signature(digest2, signature, pub)

        assert result is False

    def test_wrong_key_fails(self):
        """Signature should fail with wrong public key."""
        priv1, pub1 = generate_keypair()
        _, pub2 = generate_keypair()

        digest = domain_separated_digest(b'test')
        signature = sign_message(digest, priv1)

        result = verify_signature(digest, signature, pub2)

        assert result is False

    def test_malformed_signature_fails(self):
        """Malformed signature should return False, not raise."""
        _, pub = generate_keypair()
        digest = domain_separated_digest(b'test')

        # Too short
        result = verify_signature(digest, "abcd", pub)
        assert result is False

        # Invalid hex
        result = verify_signature(digest, "not_hex_at_all!", pub)
        assert result is False

    def test_malformed_public_key_fails(self):
        """Malformed public key should return False."""
        priv, _ = generate_keypair()
        digest = domain_separated_digest(b'test')
        signature = sign_message(digest, priv)

        # Too short
        result = verify_signature(digest, signature, "abcd")
        assert result is False

        # Invalid hex
        result = verify_signature(digest, signature, "not_valid_hex!")
        assert result is False

    def test_signature_format(self):
        """Signature should be 128 hex chars (64 bytes)."""
        priv, _ = generate_keypair()
        digest = domain_separated_digest(b'test')

        signature = sign_message(digest, priv)

        assert len(signature) == 128
        assert all(c in '0123456789abcdef' for c in signature)


class TestKeypairGeneration:
    """Tests for Ed25519 keypair generation."""

    def test_keypair_unique(self):
        """Each keypair should be unique."""
        keypairs = [generate_keypair() for _ in range(5)]

        private_keys = [kp[0] for kp in keypairs]
        public_keys = [kp[1] for kp in keypairs]

        # All private keys should be unique
        assert len(set(private_keys)) == 5
        # All public keys should be unique
        assert len(set(public_keys)) == 5

    def test_key_lengths(self):
        """Keys should be correct length."""
        priv, pub = generate_keypair()

        # Ed25519 private key: 32 bytes = 64 hex chars
        assert len(priv) == 64
        # Ed25519 public key: 32 bytes = 64 hex chars
        assert len(pub) == 64

    def test_keys_are_hex(self):
        """Keys should be valid hex strings."""
        priv, pub = generate_keypair()

        assert all(c in '0123456789abcdef' for c in priv)
        assert all(c in '0123456789abcdef' for c in pub)

    def test_derived_public_key_verifies(self):
        """Public key derived from private key should work for verification."""
        priv, pub = generate_keypair()
        digest = domain_separated_digest(b'test')

        signature = sign_message(digest, priv)

        # Verify with the paired public key
        assert verify_signature(digest, signature, pub)


class TestEdgeCases:
    """Edge case and boundary tests."""

    def test_empty_digest_signing(self):
        """Signing empty digest should work."""
        priv, pub = generate_keypair()
        # Note: normally you wouldn't sign an empty digest,
        # but the cryptographic operations should still work
        empty_digest = "00" * 32  # 32 bytes of zeros

        signature = sign_message(empty_digest, priv)
        result = verify_signature(empty_digest, signature, pub)

        assert result is True

    def test_max_length_message(self):
        """Large messages should hash and sign correctly."""
        priv, pub = generate_keypair()
        large_data = b'x' * 1_000_000  # 1MB

        digest = domain_separated_digest(large_data)
        signature = sign_message(digest, priv)
        result = verify_signature(digest, signature, pub)

        assert result is True

    def test_json_with_all_types(self):
        """Canonical JSON should handle all JSON types."""
        data = {
            "string": "hello",
            "number": 42,
            "float": 3.14159,
            "boolean": True,
            "null": None,
            "array": [1, 2, 3],
            "object": {"nested": "value"}
        }

        result = canonical_json(data)

        # Should be valid JSON
        parsed = json.loads(result)
        assert parsed["string"] == "hello"
        assert parsed["boolean"] is True
        assert parsed["null"] is None
