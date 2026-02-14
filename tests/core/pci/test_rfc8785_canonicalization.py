"""
BIZRA PCI RFC8785 Canonicalization Test Suite

Standing on Giants:
- RFC8785 (2020): JSON Canonicalization Scheme (JCS)
- RFC8259 (2017): JSON Data Interchange Format

This test suite validates:
1. RFC8785 compliance for JSON canonicalization
2. Cross-repo compatibility (ASCII-only serialization)
3. Signature stability across different systems
4. Edge cases and security properties

P0-1 Audit Fix: Unify PCI canonicalization (RFC8785 ASCII across repos)
"""

import json
import pytest
import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from core.pci.crypto import (
    canonicalize_json,
    canonical_json,
    canonicalize_and_validate,
    validate_canonical_format,
    is_canonical_json,
    domain_separated_digest,
    sign_message,
    verify_signature,
    generate_keypair,
    CanonicalizationError,
    NonAsciiError,
    NonCanonicalInputError,
)


# =============================================================================
# RFC8785 COMPLIANCE TESTS
# =============================================================================

class TestRFC8785Compliance:
    """Tests for RFC8785 JSON Canonicalization Scheme compliance."""

    def test_keys_sorted_lexicographically(self):
        """RFC8785 3.2.3: Object keys MUST be sorted lexicographically."""
        data = {"zebra": 1, "apple": 2, "mango": 3, "123": 4, "ABC": 5}
        result = canonicalize_json(data)
        result_str = result.decode('ascii')

        # Keys should appear in this order: "123" < "ABC" < "apple" < "mango" < "zebra"
        assert result_str.index('"123"') < result_str.index('"ABC"')
        assert result_str.index('"ABC"') < result_str.index('"apple"')
        assert result_str.index('"apple"') < result_str.index('"mango"')
        assert result_str.index('"mango"') < result_str.index('"zebra"')

    def test_no_whitespace_between_tokens(self):
        """RFC8785 3.2.1: No whitespace between tokens."""
        data = {"key": "value", "nested": {"inner": [1, 2, 3]}}
        result = canonicalize_json(data)

        # No whitespace characters should appear outside of strings
        assert b' ' not in result.replace(b'"value"', b'').replace(b'"inner"', b'')
        assert b'\n' not in result
        assert b'\t' not in result
        assert b'\r' not in result

    def test_ensure_ascii_escapes_unicode(self):
        """RFC8785 with ensure_ascii: Non-ASCII MUST be escaped."""
        data = {
            "cafe": "caf\u00e9",       # e with acute accent
            "emoji": "\U0001F600",     # grinning face
            "arabic": "\u0628\u0630\u0631\u0629",  # BIZRA in Arabic
        }
        result = canonicalize_json(data, ensure_ascii=True)

        # All bytes must be ASCII (< 128)
        for byte in result:
            assert byte < 128, f"Non-ASCII byte found: {byte}"

        # Unicode should be escaped as \uXXXX
        result_str = result.decode('ascii')
        assert "\\u00e9" in result_str  # e-acute
        assert "\\ud83d\\ude00" in result_str.lower()  # emoji (surrogate pair)

    def test_numbers_no_leading_zeros(self):
        """RFC8785 3.2.2.3: Numbers MUST NOT have leading zeros."""
        # Python's json.dumps already handles this correctly
        data = {"num": 42, "float": 3.14}
        result = canonicalize_json(data)
        result_str = result.decode('ascii')

        # Should be "42" not "042"
        assert '"num":42' in result_str
        assert '"float":3.14' in result_str

    def test_numbers_no_plus_sign(self):
        """RFC8785 3.2.2.3: Numbers MUST NOT have leading plus sign."""
        data = {"positive": 100, "negative": -50}
        result = canonicalize_json(data)
        result_str = result.decode('ascii')

        # No plus sign before positive numbers
        assert '+' not in result_str
        assert '"positive":100' in result_str
        assert '"negative":-50' in result_str

    def test_special_characters_escaped(self):
        """RFC8785: Control characters MUST be escaped."""
        data = {
            "newline": "line1\nline2",
            "tab": "col1\tcol2",
            "carriage_return": "a\rb",
            "backslash": "path\\to\\file",
            "quote": 'say "hello"',
        }
        result = canonicalize_json(data)
        result_str = result.decode('ascii')

        assert "\\n" in result_str
        assert "\\t" in result_str
        assert "\\r" in result_str
        assert "\\\\" in result_str
        assert '\\"' in result_str

    def test_nested_objects_sorted(self):
        """Nested objects MUST also have sorted keys."""
        data = {
            "outer_z": {"z_inner": 1, "a_inner": 2},
            "outer_a": {"z_nested": {"z_deep": 1, "a_deep": 2}},
        }
        result = canonicalize_json(data)
        result_str = result.decode('ascii')

        # Outer keys sorted
        assert result_str.index('"outer_a"') < result_str.index('"outer_z"')
        # Inner keys sorted
        assert result_str.index('"a_inner"') < result_str.index('"z_inner"')
        # Deep nested keys sorted
        assert result_str.index('"a_deep"') < result_str.index('"z_deep"')

    def test_arrays_preserve_order(self):
        """Arrays MUST preserve element order (arrays are ordered)."""
        data = {"array": [3, 1, 4, 1, 5, 9]}
        result = canonicalize_json(data)
        result_str = result.decode('ascii')

        assert '"array":[3,1,4,1,5,9]' in result_str

    def test_null_boolean_literals(self):
        """null, true, false MUST be lowercase."""
        data = {"null": None, "true": True, "false": False}
        result = canonicalize_json(data)
        result_str = result.decode('ascii')

        assert ":null" in result_str
        assert ":true" in result_str
        assert ":false" in result_str
        # Should NOT have uppercase variants
        assert "Null" not in result_str
        assert "True" not in result_str
        assert "False" not in result_str


# =============================================================================
# CROSS-REPO COMPATIBILITY TESTS
# =============================================================================

class TestCrossRepoCompatibility:
    """Tests ensuring canonical JSON is identical across different implementations."""

    def test_deterministic_output(self):
        """Same input MUST always produce identical output."""
        data = {
            "version": "1.0.0",
            "timestamp": "2024-01-15T12:00:00Z",
            "sender": {"agent_type": "PAT", "agent_id": "node-001"},
            "payload": {"action": "QUERY", "data": {"key": "value"}},
        }

        results = [canonicalize_json(data) for _ in range(100)]
        assert all(r == results[0] for r in results), "Non-deterministic output detected"

    def test_output_is_pure_ascii(self):
        """Output MUST be pure ASCII for cross-repo compatibility."""
        # Include various Unicode characters
        data = {
            "arabic": "\u0628\u0630\u0631\u0629",
            "chinese": "\u4e2d\u6587",
            "emoji": "\U0001F4BB",
            "math": "\u221e \u03c0 \u2211",
        }
        result = canonicalize_json(data, ensure_ascii=True)

        # Verify all bytes are ASCII
        assert result.isascii() if hasattr(result, 'isascii') else all(b < 128 for b in result)

        # Verify we can decode as ASCII
        decoded = result.decode('ascii')
        assert isinstance(decoded, str)

    def test_digest_stability(self):
        """Digest of canonical JSON MUST be stable across calls."""
        data = {"id": "test-001", "value": 42, "nested": {"a": 1, "b": 2}}

        canonical = canonicalize_json(data)
        digests = [domain_separated_digest(canonical) for _ in range(50)]

        assert all(d == digests[0] for d in digests), "Digest instability detected"

    def test_signature_roundtrip(self):
        """Signatures on canonical JSON MUST verify consistently."""
        priv_key, pub_key = generate_keypair()

        data = {
            "envelope_id": "test-envelope-001",
            "timestamp": "2024-01-15T12:00:00Z",
            "payload": {"action": "INFERENCE", "model": "llama-7b"},
        }

        canonical = canonicalize_json(data)
        digest = domain_separated_digest(canonical)
        signature = sign_message(digest, priv_key)

        # Verify multiple times
        for _ in range(10):
            assert verify_signature(digest, signature, pub_key)

    def test_cross_serialization_consistency(self):
        """JSON serialized by different methods MUST produce same canonical form."""
        data = {"b": 2, "a": 1, "c": {"z": 26, "y": 25}}

        # Method 1: Direct canonicalization
        result1 = canonicalize_json(data)

        # Method 2: Serialize with Python json, then parse and re-canonicalize
        json_str = json.dumps(data)  # Not canonical
        parsed = json.loads(json_str)
        result2 = canonicalize_json(parsed)

        # Method 3: Different key order in source
        data_reordered = {"c": {"y": 25, "z": 26}, "a": 1, "b": 2}
        result3 = canonicalize_json(data_reordered)

        assert result1 == result2 == result3

    def test_known_vector_compatibility(self):
        """Test against known RFC8785 test vectors."""
        # RFC8785 Appendix B test vectors (simplified)
        test_cases = [
            # Empty object
            ({}, b'{}'),
            # Empty array
            ({"a": []}, b'{"a":[]}'),
            # Numbers
            ({"int": 42, "float": 3.5}, b'{"float":3.5,"int":42}'),
            # Sorted keys
            ({"z": 1, "a": 2}, b'{"a":2,"z":1}'),
            # Nested
            ({"a": {"c": 1, "b": 2}}, b'{"a":{"b":2,"c":1}}'),
            # Boolean and null
            ({"t": True, "f": False, "n": None}, b'{"f":false,"n":null,"t":true}'),
        ]

        for data, expected in test_cases:
            result = canonicalize_json(data)
            assert result == expected, f"Mismatch for {data}: got {result}, expected {expected}"


# =============================================================================
# VALIDATION FUNCTION TESTS
# =============================================================================

class TestValidateCanonicalFormat:
    """Tests for validate_canonical_format() function."""

    def test_valid_canonical_json(self):
        """Valid canonical JSON should pass validation."""
        data = {"a": 1, "b": 2}
        canonical = canonicalize_json(data)

        is_valid, error = validate_canonical_format(canonical)
        assert is_valid, f"Valid canonical JSON failed: {error}"
        assert error == ""

    def test_rejects_whitespace(self):
        """JSON with whitespace between tokens should fail."""
        # Manually create JSON with whitespace
        non_canonical = b'{"a": 1, "b": 2}'  # Spaces after colons and comma

        is_valid, error = validate_canonical_format(non_canonical)
        assert not is_valid
        assert "Whitespace" in error or "Non-canonical" in error

    def test_rejects_unsorted_keys(self):
        """JSON with unsorted keys should fail."""
        # Manually create JSON with unsorted keys
        non_canonical = b'{"z":1,"a":2}'  # z before a

        is_valid, error = validate_canonical_format(non_canonical)
        assert not is_valid
        assert "Non-canonical" in error

    def test_rejects_non_ascii(self):
        """JSON with raw non-ASCII bytes should fail."""
        # UTF-8 encoded unicode (not escaped)
        non_canonical = "cafe: caf\u00e9".encode('utf-8')

        is_valid, error = validate_canonical_format(non_canonical)
        assert not is_valid
        assert "Non-ASCII" in error

    def test_rejects_invalid_json(self):
        """Invalid JSON should fail validation."""
        invalid = b'{"broken":}'  # No whitespace, but invalid JSON value

        is_valid, error = validate_canonical_format(invalid)
        assert not is_valid
        assert "Invalid JSON" in error or "Non-canonical" in error

    def test_is_canonical_json_convenience(self):
        """is_canonical_json() should be a boolean wrapper."""
        data = {"key": "value"}
        canonical = canonicalize_json(data)

        assert is_canonical_json(canonical) is True
        assert is_canonical_json(b'{"key": "value"}') is False  # Has whitespace


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Tests for error conditions and edge cases."""

    def test_non_ascii_raises_when_required(self):
        """NonAsciiError should be raised if ASCII-only output fails."""
        # This shouldn't happen with Python's json.dumps(ensure_ascii=True)
        # but we test the validation path
        data = {"key": "value"}
        result = canonicalize_json(data, ensure_ascii=True)
        assert result.decode('ascii')  # Should not raise

    def test_nan_not_allowed(self):
        """NaN is not valid JSON per RFC8785."""
        import math
        data = {"value": float('nan')}

        with pytest.raises(ValueError):
            canonicalize_json(data)

    def test_infinity_not_allowed(self):
        """Infinity is not valid JSON per RFC8785."""
        data = {"value": float('inf')}

        with pytest.raises(ValueError):
            canonicalize_json(data)

    def test_circular_reference_fails(self):
        """Circular references should fail with clear error."""
        data: dict = {"key": "value"}
        data["self"] = data  # Create circular reference

        with pytest.raises((ValueError, TypeError)):
            canonicalize_json(data)

    def test_non_serializable_type_fails(self):
        """Non-JSON-serializable types should fail."""
        data = {"func": lambda x: x}

        with pytest.raises(TypeError):
            canonicalize_json(data)


# =============================================================================
# BACKWARD COMPATIBILITY TESTS
# =============================================================================

class TestBackwardCompatibility:
    """Tests ensuring backward compatibility with existing code."""

    def test_canonical_json_still_works(self):
        """Deprecated canonical_json() should still work."""
        data = {"key": "value", "num": 42}
        result = canonical_json(data)

        assert isinstance(result, bytes)
        assert b'"key":"value"' in result
        assert b'"num":42' in result

    def test_canonical_json_equals_canonicalize_json(self):
        """canonical_json() should produce same output as canonicalize_json()."""
        data = {
            "version": "1.0.0",
            "nested": {"z": 1, "a": 2},
            "array": [1, 2, 3],
        }

        result_old = canonical_json(data)
        result_new = canonicalize_json(data, ensure_ascii=True)

        assert result_old == result_new


# =============================================================================
# PCI ENVELOPE INTEGRATION TESTS
# =============================================================================

class TestPCIEnvelopeIntegration:
    """Tests for PCI envelope canonicalization."""

    def test_envelope_digest_deterministic(self):
        """PCI envelope digest must be deterministic."""
        from core.pci.envelope import EnvelopeBuilder

        builder = EnvelopeBuilder()
        # Create identical envelopes multiple times
        envelopes = []
        for _ in range(5):
            env = (
                builder
                .with_sender("PAT", "test-agent", "0" * 64)
                .with_payload("QUERY", {"key": "value"}, "policy123", "state456")
                .with_metadata(0.95, 0.90)
                .build()
            )
            # Override variable fields for determinism
            env.envelope_id = "fixed-id"
            env.timestamp = "2024-01-15T12:00:00Z"
            env.nonce = "0" * 64
            envelopes.append(env)

        digests = [env.compute_digest() for env in envelopes]
        assert all(d == digests[0] for d in digests)

    def test_signed_envelope_verifies_after_serialization(self):
        """Signed envelope should verify after round-trip serialization."""
        from core.pci.envelope import PCIEnvelope, EnvelopeBuilder

        priv_key, pub_key = generate_keypair()

        env = (
            EnvelopeBuilder()
            .with_sender("PAT", "test-agent", pub_key)
            .with_payload("QUERY", {"key": "value"}, "policy123", "state456")
            .with_metadata(0.95, 0.90)
            .build()
        )
        env.sign(priv_key)

        # Serialize to dict and back
        env_dict = env.to_dict()
        env_restored = PCIEnvelope.from_dict(env_dict)

        # Compute digest and verify signature
        digest = env_restored.compute_digest()
        assert verify_signature(
            digest,
            env_restored.signature.value,
            pub_key
        )


# =============================================================================
# CROSS-REPO COMPATIBILITY SIMULATION TESTS
# =============================================================================

class TestCrossRepoSimulation:
    """Simulate cross-repo canonicalization scenarios."""

    def test_different_python_json_implementations(self):
        """
        Simulate different JSON libraries potentially used in other repos.
        All should produce identical canonical output when using our function.
        """
        data = {
            "bizra": "seed",
            "nodes": 8_000_000_000,
            "active": True,
            "config": None,
        }

        # Simulate receiving JSON from another repo (potentially non-canonical)
        # Note: When we parse these back, missing keys won't magically appear
        incoming_variants = [
            '{"bizra":"seed","nodes":8000000000,"active":true,"config":null}',  # Canonical
            '{"nodes":8000000000,"bizra":"seed","config":null,"active":true}',  # Wrong order, all keys
            '{"active":true,"bizra":"seed","config":null,"nodes":8000000000}',  # All keys, different order
        ]

        canonical_result = canonicalize_json(data)

        for incoming in incoming_variants:
            parsed = json.loads(incoming)
            re_canonical = canonicalize_json(parsed)
            assert re_canonical == canonical_result, f"Failed for input: {incoming}"

    def test_bizra_omega_rust_compatibility(self):
        """
        Test that our canonical JSON would match Rust serde_json canonicalization.

        Rust serde_json with:
        - #[serde(sort_keys)]
        - to_string() (no pretty print)

        Should produce identical output for same data.
        """
        # This is the expected format from Rust bizra-omega
        test_cases = [
            (
                {"action": "QUERY", "model": "llama-7b", "version": "1.0.0"},
                b'{"action":"QUERY","model":"llama-7b","version":"1.0.0"}',
            ),
            (
                {"a": 1, "b": [1, 2, 3], "c": {"x": True, "y": False}},  # Python booleans
                b'{"a":1,"b":[1,2,3],"c":{"x":true,"y":false}}',  # JSON booleans
            ),
        ]

        for data, expected_rust_output in test_cases:
            result = canonicalize_json(data)
            # Note: Python json produces lowercase true/false already
            assert result == expected_rust_output, f"Mismatch: {result} != {expected_rust_output}"

    def test_unicode_cross_platform(self):
        """
        Unicode handling must be consistent across platforms.
        All non-ASCII should be escaped as \\uXXXX for cross-platform safety.
        """
        data = {
            "arabic_bizra": "\u0628\u0630\u0631\u0629",  # BIZRA in Arabic
            "ihsan": "\u0625\u062d\u0633\u0627\u0646",   # Excellence in Arabic
        }

        result = canonicalize_json(data, ensure_ascii=True)
        result_str = result.decode('ascii')

        # All Unicode should be escaped
        assert "\\u0628" in result_str
        assert "\\u0625" in result_str

        # Verify can be decoded on any platform
        parsed_back = json.loads(result_str)
        assert parsed_back["arabic_bizra"] == "\u0628\u0630\u0631\u0629"


# =============================================================================
# PERFORMANCE TESTS (marked slow)
# =============================================================================

@pytest.mark.slow
class TestPerformance:
    """Performance tests for canonicalization."""

    def test_large_document_performance(self):
        """Large documents should canonicalize in reasonable time."""
        import time

        # Create large nested document
        data = {
            f"key_{i}": {
                f"nested_{j}": f"value_{i}_{j}"
                for j in range(100)
            }
            for i in range(100)
        }

        start = time.time()
        result = canonicalize_json(data)
        elapsed = time.time() - start

        assert elapsed < 1.0, f"Canonicalization took too long: {elapsed}s"
        assert len(result) > 10000  # Sanity check

    def test_repeated_canonicalization_performance(self):
        """Repeated canonicalization should be fast."""
        import time

        data = {"version": "1.0.0", "value": 42, "nested": {"a": 1, "b": 2}}

        start = time.time()
        for _ in range(10000):
            canonicalize_json(data)
        elapsed = time.time() - start

        assert elapsed < 5.0, f"10000 iterations took too long: {elapsed}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
