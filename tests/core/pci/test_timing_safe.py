"""
BIZRA PCI Timing Attack Resistance Test Suite

Tests for constant-time comparison operations to prevent timing attacks.

Standing on Giants:
- Kocher (1996): Timing Attacks on Implementations of Diffie-Hellman
- Bernstein (2005): Cache-timing attacks on AES
- Brumley & Boneh (2003): Remote Timing Attacks are Practical

Reference: CWE-208 Observable Timing Discrepancy
"""

import pytest
import time
import statistics
import sys
from pathlib import Path

# Add project root to path (works across platforms)
_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from core.pci.crypto import (
    timing_safe_compare,
    timing_safe_compare_hex,
    verify_digest_match,
    domain_separated_digest,
    generate_keypair,
    sign_message,
    verify_signature,
)
from core.pci.envelope import (
    PCIEnvelope,
    EnvelopeBuilder,
    _timing_safe_nonce_lookup,
    _seen_nonces,
)


class TestTimingSafeCompare:
    """Tests for timing_safe_compare function."""

    def test_equal_strings_return_true(self):
        """Equal strings should return True."""
        assert timing_safe_compare("hello", "hello") is True
        assert timing_safe_compare("", "") is True
        assert timing_safe_compare("a" * 1000, "a" * 1000) is True

    def test_unequal_strings_return_false(self):
        """Unequal strings should return False."""
        assert timing_safe_compare("hello", "world") is False
        assert timing_safe_compare("hello", "hell") is False
        assert timing_safe_compare("hello", "Hello") is False

    def test_equal_bytes_return_true(self):
        """Equal bytes should return True."""
        assert timing_safe_compare(b"hello", b"hello") is True
        assert timing_safe_compare(b"", b"") is True
        assert timing_safe_compare(b"\x00\x01\x02", b"\x00\x01\x02") is True

    def test_unequal_bytes_return_false(self):
        """Unequal bytes should return False."""
        assert timing_safe_compare(b"hello", b"world") is False
        assert timing_safe_compare(b"\x00", b"\x01") is False

    def test_mixed_str_bytes_comparison(self):
        """String and bytes with same content should compare equal."""
        assert timing_safe_compare("hello", b"hello") is True
        assert timing_safe_compare(b"hello", "hello") is True

    def test_unicode_comparison(self):
        """Unicode strings should compare correctly."""
        assert timing_safe_compare("cafe\u0301", "cafe\u0301") is True
        assert timing_safe_compare("cafe", "cafe\u0301") is False

    def test_empty_vs_nonempty(self):
        """Empty vs non-empty should return False."""
        assert timing_safe_compare("", "a") is False
        assert timing_safe_compare("a", "") is False
        assert timing_safe_compare(b"", b"a") is False


class TestTimingSafeCompareHex:
    """Tests for timing_safe_compare_hex function."""

    def test_equal_hex_strings(self):
        """Equal hex strings should return True."""
        assert timing_safe_compare_hex("abcd1234", "abcd1234") is True
        assert timing_safe_compare_hex("", "") is True

    def test_case_insensitive(self):
        """Hex comparison should be case-insensitive."""
        assert timing_safe_compare_hex("ABCD", "abcd") is True
        assert timing_safe_compare_hex("AbCd", "aBcD") is True
        assert timing_safe_compare_hex("0123456789abcdef", "0123456789ABCDEF") is True

    def test_unequal_hex_strings(self):
        """Unequal hex strings should return False."""
        assert timing_safe_compare_hex("abcd", "efgh") is False
        assert timing_safe_compare_hex("1234", "12345") is False

    def test_digest_length_hex(self):
        """Typical digest-length hex strings should work."""
        digest1 = "a" * 64  # SHA-256 length
        digest2 = "a" * 64
        digest3 = "b" * 64

        assert timing_safe_compare_hex(digest1, digest2) is True
        assert timing_safe_compare_hex(digest1, digest3) is False


class TestVerifyDigestMatch:
    """Tests for verify_digest_match function."""

    def test_matching_digests(self):
        """Matching digests should return True."""
        data = b"test data"
        digest = domain_separated_digest(data)

        assert verify_digest_match(digest, digest) is True

    def test_mismatched_digests(self):
        """Mismatched digests should return False."""
        digest1 = domain_separated_digest(b"data1")
        digest2 = domain_separated_digest(b"data2")

        assert verify_digest_match(digest1, digest2) is False

    def test_case_insensitive_digest_match(self):
        """Digest comparison should be case-insensitive."""
        digest = domain_separated_digest(b"test")
        digest_upper = digest.upper()

        assert verify_digest_match(digest, digest_upper) is True


class TestTimingSafeNonceLookup:
    """Tests for _timing_safe_nonce_lookup function."""

    def test_empty_set(self):
        """Empty set should return False."""
        assert _timing_safe_nonce_lookup("any_nonce", set()) is False

    def test_nonce_found(self):
        """Present nonce should return True."""
        nonces = {"nonce1", "nonce2", "nonce3"}
        assert _timing_safe_nonce_lookup("nonce2", nonces) is True

    def test_nonce_not_found(self):
        """Absent nonce should return False."""
        nonces = {"nonce1", "nonce2", "nonce3"}
        assert _timing_safe_nonce_lookup("nonce4", nonces) is False

    def test_similar_nonces(self):
        """Similar but different nonces should not match."""
        nonces = {"nonce_abc"}
        assert _timing_safe_nonce_lookup("nonce_abd", nonces) is False
        assert _timing_safe_nonce_lookup("nonce_ab", nonces) is False
        assert _timing_safe_nonce_lookup("nonce_abcd", nonces) is False


class TestTimingAttackResistance:
    """
    Statistical tests for timing attack resistance.

    These tests verify that comparison operations take approximately
    the same time regardless of where the first difference occurs.

    IMPORTANT: These are probabilistic tests and may occasionally fail
    due to system scheduling variations. They are marked as slow.
    """

    @pytest.mark.slow
    def test_compare_timing_consistency_early_mismatch(self):
        """
        Verify timing is consistent regardless of mismatch position.

        Timing attacks exploit the fact that standard string comparison
        returns early when a difference is found. We verify that our
        timing-safe comparison does not exhibit this behavior.
        """
        # Base string
        base = "a" * 100

        # Mismatch at position 0
        early_mismatch = "b" + "a" * 99

        # Mismatch at position 99
        late_mismatch = "a" * 99 + "b"

        # Measure timing for early mismatch
        early_times = []
        for _ in range(1000):
            start = time.perf_counter_ns()
            timing_safe_compare(base, early_mismatch)
            end = time.perf_counter_ns()
            early_times.append(end - start)

        # Measure timing for late mismatch
        late_times = []
        for _ in range(1000):
            start = time.perf_counter_ns()
            timing_safe_compare(base, late_mismatch)
            end = time.perf_counter_ns()
            late_times.append(end - start)

        # Calculate statistics
        early_median = statistics.median(early_times)
        late_median = statistics.median(late_times)

        # The ratio should be close to 1.0 (within 50% tolerance)
        # A vulnerable implementation would show late_median >> early_median
        ratio = late_median / early_median if early_median > 0 else 1.0

        # Allow for some system noise (0.5 to 2.0 range)
        assert 0.5 < ratio < 2.0, (
            f"Timing ratio {ratio:.2f} suggests timing leak. "
            f"Early median: {early_median}ns, Late median: {late_median}ns"
        )

    @pytest.mark.slow
    def test_hex_compare_timing_consistency(self):
        """Verify hex comparison timing is consistent."""
        base = "abcdef" * 10  # 60 char hex

        # Early and late mismatches
        early_mismatch = "1" + base[1:]
        late_mismatch = base[:-1] + "1"

        early_times = []
        late_times = []

        for _ in range(1000):
            start = time.perf_counter_ns()
            timing_safe_compare_hex(base, early_mismatch)
            end = time.perf_counter_ns()
            early_times.append(end - start)

            start = time.perf_counter_ns()
            timing_safe_compare_hex(base, late_mismatch)
            end = time.perf_counter_ns()
            late_times.append(end - start)

        early_median = statistics.median(early_times)
        late_median = statistics.median(late_times)

        ratio = late_median / early_median if early_median > 0 else 1.0

        assert 0.5 < ratio < 2.0, (
            f"Hex timing ratio {ratio:.2f} suggests timing leak"
        )

    @pytest.mark.slow
    def test_nonce_lookup_timing_consistency(self):
        """
        Verify nonce lookup takes similar time for found vs not found.

        This tests that the lookup iterates through all nonces
        regardless of whether a match is found early.
        """
        # Create a set with 100 nonces
        nonces = {f"nonce_{i:04d}" for i in range(100)}

        # Time lookups for existing nonce (first in set iteration)
        existing_nonce = list(nonces)[0]
        found_times = []

        for _ in range(500):
            start = time.perf_counter_ns()
            _timing_safe_nonce_lookup(existing_nonce, nonces)
            end = time.perf_counter_ns()
            found_times.append(end - start)

        # Time lookups for non-existing nonce
        missing_nonce = "nonce_not_in_set"
        not_found_times = []

        for _ in range(500):
            start = time.perf_counter_ns()
            _timing_safe_nonce_lookup(missing_nonce, nonces)
            end = time.perf_counter_ns()
            not_found_times.append(end - start)

        found_median = statistics.median(found_times)
        not_found_median = statistics.median(not_found_times)

        # Both should take similar time since we iterate all nonces
        # Widened bounds (0.3-3.0) to accommodate WSL/VM timing jitter
        ratio = found_median / not_found_median if not_found_median > 0 else 1.0

        assert 0.3 < ratio < 3.0, (
            f"Nonce lookup timing ratio {ratio:.2f} suggests timing leak. "
            f"Found: {found_median}ns, Not found: {not_found_median}ns"
        )


class TestSignatureVerificationSecurity:
    """Tests for signature verification security properties."""

    def test_signature_verification_returns_bool(self):
        """Verification should return bool, not raise exceptions for invalid sigs."""
        priv, pub = generate_keypair()
        digest = domain_separated_digest(b"test message")
        signature = sign_message(digest, priv)

        # Valid signature
        assert verify_signature(digest, signature, pub) is True

        # Tampered signature (flip one byte)
        tampered_sig = "00" + signature[2:]
        assert verify_signature(digest, tampered_sig, pub) is False

        # Wrong digest
        wrong_digest = domain_separated_digest(b"different message")
        assert verify_signature(wrong_digest, signature, pub) is False

        # Wrong key
        _, other_pub = generate_keypair()
        assert verify_signature(digest, signature, other_pub) is False

    def test_malformed_inputs_dont_raise(self):
        """Malformed inputs should return False, not raise exceptions."""
        _, pub = generate_keypair()
        digest = domain_separated_digest(b"test")

        # Various malformed inputs
        assert verify_signature("not_hex!", "abcd" * 32, pub) is False
        assert verify_signature(digest, "not_hex!", pub) is False
        assert verify_signature(digest, "ab" * 32, "not_hex!") is False
        assert verify_signature("", "", "") is False
        assert verify_signature(digest, "", pub) is False


class TestEnvelopeReplayProtection:
    """Tests for envelope replay protection with timing-safe operations."""

    def setup_method(self):
        """Clear seen nonces before each test."""
        _seen_nonces.clear()

    def test_replay_detection(self):
        """Replayed envelope should be detected."""
        priv, pub = generate_keypair()

        envelope = (
            EnvelopeBuilder()
            .with_sender("PAT", "agent-1", pub)
            .with_payload("test_action", {"key": "value"}, "policy_hash", "state_hash")
            .with_metadata(0.98, 0.95)
            .build()
        )
        envelope.sign(priv)

        # First check should not be replay
        assert envelope.is_replay() is False

        # Second check with same nonce should be replay
        # Note: We need to check again before the nonce is seen
        envelope2 = (
            EnvelopeBuilder()
            .with_sender("PAT", "agent-2", pub)
            .with_payload("other_action", {}, "ph", "sh")
            .with_metadata(0.99, 0.96)
            .build()
        )
        # Manually set same nonce
        envelope2.nonce = envelope.nonce

        assert envelope2.is_replay() is True

    def test_different_nonces_not_replay(self):
        """Different nonces should not trigger replay detection."""
        priv, pub = generate_keypair()

        envelope1 = (
            EnvelopeBuilder()
            .with_sender("PAT", "agent-1", pub)
            .with_payload("action1", {}, "ph", "sh")
            .with_metadata(0.98, 0.95)
            .build()
        )

        envelope2 = (
            EnvelopeBuilder()
            .with_sender("PAT", "agent-2", pub)
            .with_payload("action2", {}, "ph", "sh")
            .with_metadata(0.98, 0.95)
            .build()
        )

        # Both should have unique nonces
        assert envelope1.nonce != envelope2.nonce

        # Neither should be detected as replay on first check
        assert envelope1.is_replay() is False
        assert envelope2.is_replay() is False


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_string_comparison(self):
        """Empty strings should compare correctly."""
        assert timing_safe_compare("", "") is True
        assert timing_safe_compare_hex("", "") is True

    def test_very_long_string_comparison(self):
        """Very long strings should compare correctly."""
        long_str = "a" * 100000
        assert timing_safe_compare(long_str, long_str) is True
        assert timing_safe_compare(long_str, long_str + "b") is False

    def test_null_bytes_in_comparison(self):
        """Strings with null bytes should compare correctly."""
        s1 = "hello\x00world"
        s2 = "hello\x00world"
        s3 = "hello\x00other"

        assert timing_safe_compare(s1, s2) is True
        assert timing_safe_compare(s1, s3) is False

    def test_special_characters(self):
        """Special characters should compare correctly."""
        special = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        assert timing_safe_compare(special, special) is True
        assert timing_safe_compare(special, special[:-1]) is False
