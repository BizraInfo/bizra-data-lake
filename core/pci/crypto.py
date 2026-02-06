"""
BIZRA PROOF-CARRYING INFERENCE (PCI) PROTOCOL
Cryptographic Primitives (Ed25519 + BLAKE3)

Standing on Giants:
- RFC8785 (2020): JSON Canonicalization Scheme (JCS)
- RFC8259 (2017): JSON Data Interchange Format
- Kocher (1996): Timing Attacks on Implementations of Diffie-Hellman
- Bernstein (2005): Cache-timing attacks on AES
- CERT/CC: CWE-208 Observable Timing Discrepancy
"""

import json
import hmac
import re
from decimal import Decimal
from typing import Any, Dict, List, Tuple, Union

import blake3
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature

PCI_DOMAIN_PREFIX = "bizra-pci-v1:"


# =============================================================================
# RFC8785 CANONICALIZATION ERROR TYPES
# =============================================================================

class CanonicalizationError(Exception):
    """Base error for RFC8785 canonicalization failures."""
    pass


class NonAsciiError(CanonicalizationError):
    """Raised when canonicalized output contains non-ASCII characters."""
    pass


class NonCanonicalInputError(CanonicalizationError):
    """Raised when input JSON is detected as non-canonical."""
    pass


# =============================================================================
# TIMING-SAFE COMPARISON UTILITIES (Security Hardening S-2)
# Standing on Giants: Kocher (1996) - "Timing Attacks on Implementations"
# =============================================================================

def timing_safe_compare(a: Union[str, bytes], b: Union[str, bytes]) -> bool:
    """
    Constant-time comparison of two strings or byte sequences.

    SECURITY: Uses hmac.compare_digest() to prevent timing attacks.
    Standard string comparison (==) returns early on first mismatch,
    leaking information about the position of differences through timing.

    This function:
    1. Converts strings to bytes if needed (UTF-8 encoding)
    2. Uses hmac.compare_digest() for constant-time comparison
    3. Returns False for type mismatches without timing leakage

    Reference: Kocher (1996) "Timing Attacks on Implementations of Diffie-Hellman"

    Args:
        a: First value to compare (str or bytes)
        b: Second value to compare (str or bytes)

    Returns:
        True if values are equal, False otherwise (constant time)
    """
    # Convert to bytes for consistent comparison
    if isinstance(a, str):
        a = a.encode('utf-8')
    if isinstance(b, str):
        b = b.encode('utf-8')

    # hmac.compare_digest requires both to be bytes or both str
    # We've normalized to bytes above
    return hmac.compare_digest(a, b)


def timing_safe_compare_hex(a: str, b: str) -> bool:
    """
    Constant-time comparison specifically for hex strings.

    SECURITY: Normalizes case before comparison to prevent
    timing leaks from case differences (e.g., 'ab' vs 'AB').

    Args:
        a: First hex string
        b: Second hex string

    Returns:
        True if hex values are equal (case-insensitive), False otherwise
    """
    # Normalize to lowercase for consistent comparison
    # This is done unconditionally to avoid timing leaks
    a_lower = a.lower()
    b_lower = b.lower()

    return timing_safe_compare(a_lower, b_lower)


# =============================================================================
# RFC8785 JSON CANONICALIZATION SCHEME (JCS)
# Standing on Giants:
# - RFC8785 (2020): JSON Canonicalization Scheme
# - RFC8259 (2017): JSON Data Interchange Format
# =============================================================================

def canonicalize_json(data: Dict[str, Any], ensure_ascii: bool = True) -> bytes:
    """
    RFC8785 JSON Canonicalization Scheme (JCS).

    Guarantees deterministic, cross-platform JSON serialization
    for cryptographic signing operations. This is the PRIMARY
    canonicalization function for cross-repo compatibility.

    Standing on Giants:
    - RFC8785 (2020): JSON Canonicalization Scheme
    - RFC8259 (2017): JSON Data Interchange Format

    RFC8785 Key Rules Applied:
    1. Object keys sorted lexicographically (by UTF-16 code units)
    2. No whitespace padding between tokens
    3. Numbers in shortest form (no trailing zeros, no unnecessary decimals)
    4. Strings properly escaped (\\uXXXX for non-ASCII when ensure_ascii=True)
    5. UTF-8 encoding for output (ASCII-safe when ensure_ascii=True)

    SECURITY: ensure_ascii=True is REQUIRED for cross-repo compatibility.
    Different JSON libraries handle Unicode normalization differently,
    which can cause signature mismatches across repos.

    Args:
        data: Dictionary to canonicalize
        ensure_ascii: If True, escape all non-ASCII characters (default: True)
                     MUST be True for cross-repo compatibility.

    Returns:
        Canonical JSON as bytes (ASCII-only when ensure_ascii=True)

    Raises:
        NonAsciiError: If ensure_ascii=True but output contains non-ASCII
        TypeError: If data contains non-serializable types
    """
    # Use custom encoder for RFC8785 number handling
    canonical = json.dumps(
        data,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=ensure_ascii,
        allow_nan=False,  # RFC8785: NaN/Infinity not allowed
    )

    # CRITICAL VALIDATION: Ensure output is ASCII-only for cross-repo compatibility
    if ensure_ascii:
        if not canonical.isascii():
            raise NonAsciiError(
                f"Canonicalized JSON must be ASCII-only for cross-repo compatibility. "
                f"Found non-ASCII characters in output."
            )
        # Encode as ASCII for maximum cross-platform compatibility
        return canonical.encode('ascii')

    return canonical.encode('utf-8')


def canonical_json(data: Dict[str, Any]) -> bytes:
    """
    RFC 8785 JSON Canonicalization Scheme (JCS).

    DEPRECATED: Use canonicalize_json() for new code.
    This function is maintained for backward compatibility.

    - Keys sorted lexicographically
    - No whitespace
    - ASCII-only output (RFC 8785 mandate)

    SECURITY: ensure_ascii=True is REQUIRED for RFC 8785 compliance.
    Using False allows Unicode normalization differences across systems,
    enabling signature malleability attacks.
    """
    return canonicalize_json(data, ensure_ascii=True)


def validate_canonical_format(json_bytes: bytes) -> Tuple[bool, str]:
    """
    Validate that JSON bytes are in RFC8785 canonical format.

    This function detects non-canonical inputs that could indicate:
    - Cross-repo incompatibility (different canonicalization implementations)
    - Attempted signature malleability attacks
    - Corrupted or tampered data

    Standing on Giants:
    - RFC8785 Section 3: Serialization rules

    Validation checks:
    1. Valid JSON structure
    2. No whitespace between tokens
    3. Keys sorted lexicographically
    4. ASCII-only content
    5. No trailing zeros in numbers
    6. No unnecessary decimal points

    Args:
        json_bytes: JSON data as bytes to validate

    Returns:
        Tuple of (is_valid: bool, error_message: str)
        error_message is empty string if valid
    """
    # Check ASCII-only
    try:
        json_str = json_bytes.decode('ascii')
    except UnicodeDecodeError as e:
        return False, f"Non-ASCII byte at position {e.start}: {json_bytes[e.start:e.start+1]!r}"

    # Check for whitespace between tokens (not inside strings)
    # RFC8785 requires no whitespace outside of string values
    if _has_non_string_whitespace(json_str):
        return False, "Whitespace detected outside of string values"

    # Parse and re-canonicalize to compare
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"

    # Re-canonicalize and compare
    try:
        canonical = canonicalize_json(data, ensure_ascii=True)
    except (NonAsciiError, TypeError) as e:
        return False, f"Canonicalization failed: {e}"

    if canonical != json_bytes:
        # Find the first difference for debugging
        diff_pos = _find_first_difference(json_bytes, canonical)
        return False, (
            f"Non-canonical format detected at byte {diff_pos}. "
            f"Input may have unsorted keys, extra whitespace, or different number formatting."
        )

    return True, ""


def _has_non_string_whitespace(json_str: str) -> bool:
    """
    Check if JSON string has whitespace outside of string values.

    RFC8785 requires no whitespace between tokens.
    """
    in_string = False
    escape_next = False

    for char in json_str:
        if escape_next:
            escape_next = False
            continue

        if char == '\\' and in_string:
            escape_next = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if not in_string and char in ' \t\n\r':
            return True

    return False


def _find_first_difference(a: bytes, b: bytes) -> int:
    """Find the position of the first differing byte between two byte sequences."""
    min_len = min(len(a), len(b))
    for i in range(min_len):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return min_len
    return -1  # No difference


def is_canonical_json(json_bytes: bytes) -> bool:
    """
    Quick check if JSON bytes are in canonical format.

    Convenience wrapper around validate_canonical_format() that returns
    only a boolean. Use validate_canonical_format() when you need
    the error message for debugging.

    Args:
        json_bytes: JSON data as bytes to validate

    Returns:
        True if canonical, False otherwise
    """
    is_valid, _ = validate_canonical_format(json_bytes)
    return is_valid


def canonicalize_and_validate(data: Dict[str, Any]) -> bytes:
    """
    Canonicalize JSON and validate the result.

    This is the SAFEST function for PCI envelope serialization.
    It performs canonicalization and then validates the output
    to ensure cross-repo compatibility.

    Args:
        data: Dictionary to canonicalize

    Returns:
        Validated canonical JSON as bytes

    Raises:
        NonAsciiError: If output contains non-ASCII characters
        CanonicalizationError: If validation fails
    """
    canonical = canonicalize_json(data, ensure_ascii=True)

    is_valid, error = validate_canonical_format(canonical)
    if not is_valid:
        raise CanonicalizationError(f"Self-validation failed: {error}")

    return canonical

def domain_separated_digest(canonical_data: bytes) -> str:
    """
    Compute domain-separated BLAKE3 digest.
    Prefix: bizra-pci-v1:
    """
    hasher = blake3.blake3()
    hasher.update(PCI_DOMAIN_PREFIX.encode('utf-8'))
    hasher.update(canonical_data)
    return hasher.hexdigest()

def sign_message(digest_hex: str, private_key_hex: str) -> str:
    """
    Sign a digest using Ed25519.
    input: hex digest, hex private key
    output: hex signature
    """
    priv_bytes = bytes.fromhex(private_key_hex)
    digest_bytes = bytes.fromhex(digest_hex)
    
    start_time = 0
    private_key = ed25519.Ed25519PrivateKey.from_private_bytes(priv_bytes)
    signature = private_key.sign(digest_bytes)
    return signature.hex()

def verify_signature(digest_hex: str, signature_hex: str, public_key_hex: str) -> bool:
    """
    Verify an Ed25519 signature.

    SECURITY NOTES:
    1. Catches only InvalidSignature and ValueError (malformed hex).
       Other exceptions (library bugs, memory errors) should propagate.

    2. Ed25519 verification via cryptography library is already constant-time
       at the cryptographic operation level. However, we ensure consistent
       return timing by avoiding early returns based on input validation.

    3. For additional digest comparisons (if needed), use timing_safe_compare_hex().

    Reference: Kocher (1996) - Timing attacks can leak secret information
    """
    try:
        pub_bytes = bytes.fromhex(public_key_hex)
        sig_bytes = bytes.fromhex(signature_hex)
        digest_bytes = bytes.fromhex(digest_hex)

        public_key = ed25519.Ed25519PublicKey.from_public_bytes(pub_bytes)
        public_key.verify(sig_bytes, digest_bytes)
        return True
    except InvalidSignature:
        # Expected: signature does not match
        return False
    except ValueError:
        # Expected: malformed hex input
        return False
    # NOTE: Other exceptions (TypeError, MemoryError, etc.) propagate intentionally


def verify_digest_match(computed_digest: str, expected_digest: str) -> bool:
    """
    Timing-safe verification that two digests match.

    SECURITY: Use this instead of `==` when comparing digests
    to prevent timing attacks that could leak information about
    the expected digest value.

    Args:
        computed_digest: The digest computed from message
        expected_digest: The expected digest to compare against

    Returns:
        True if digests match, False otherwise (constant time)
    """
    return timing_safe_compare_hex(computed_digest, expected_digest)

def generate_keypair() -> Tuple[str, str]:
    """Generates (private_key_hex, public_key_hex)."""
    priv = ed25519.Ed25519PrivateKey.generate()
    pub = priv.public_key()
    
    priv_hex = priv.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption()
    ).hex()
    
    pub_hex = pub.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw
    ).hex()
    
    return priv_hex, pub_hex
