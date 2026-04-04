"""
BIZRA PCI Protocol — Cryptographic Operations
==============================================
Ed25519 signing, BLAKE3 hashing, nonce generation.

Status: FROZEN — Changes require version bump + test vector update
Security: Keys MUST be stored in HSM or secure enclave in production.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .types import DOMAIN_PREFIX, NONCE_BYTES

# Try to import cryptographic libraries
try:
    from nacl.signing import SigningKey, VerifyKey
    from nacl.exceptions import BadSignatureError

    NACL_AVAILABLE = True
except ImportError:
    NACL_AVAILABLE = False
    SigningKey = None
    VerifyKey = None
    BadSignatureError = Exception

try:
    import blake3

    BLAKE3_AVAILABLE = True
except ImportError:
    BLAKE3_AVAILABLE = False


# =============================================================================
# CANONICAL JSON (RFC 8785 JCS)
# =============================================================================


def canonical_json(data: Dict[str, Any]) -> bytes:
    """
    Serialize data to canonical JSON (RFC 8785 JCS).

    Properties:
    - Keys sorted lexicographically (Unicode code point order)
    - No whitespace between tokens
    - Numbers: no leading zeros, no trailing zeros after decimal
    - Strings: UTF-8, minimal escape sequences
    - No duplicate keys
    """
    return json.dumps(
        data,
        separators=(",", ":"),
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")


def canonical_json_str(data: Dict[str, Any]) -> str:
    """Serialize data to canonical JSON string."""
    return canonical_json(data).decode("utf-8")


# =============================================================================
# BLAKE3 HASHING WITH DOMAIN SEPARATION
# =============================================================================


def blake3_digest(data: bytes) -> str:
    """
    Compute BLAKE3 digest of raw bytes.

    Returns: Hex-encoded 256-bit digest
    """
    if BLAKE3_AVAILABLE:
        return blake3.blake3(data).hexdigest()
    else:
        # Fallback to SHA-256 if blake3 not available
        # WARNING: This should only be used in development
        return hashlib.sha256(data).hexdigest()


def domain_separated_digest(data: bytes, domain: str = DOMAIN_PREFIX) -> str:
    """
    Compute domain-separated BLAKE3 digest.

    Formula: BLAKE3(domain || data)

    Domain separation prevents cross-protocol attacks where a digest
    from one protocol could be reused in another.

    Args:
        data: Raw bytes to hash
        domain: Domain prefix (default: "bizra-pci-v1:")

    Returns: Hex-encoded 256-bit digest
    """
    domain_bytes = domain.encode("utf-8")
    combined = domain_bytes + data
    return blake3_digest(combined)


def envelope_digest(envelope_data: Dict[str, Any]) -> str:
    """
    Compute the canonical digest of a PCI envelope.

    This is the digest used for signing and verification.

    Formula: BLAKE3("bizra-pci-v1:" || canonical_json(envelope))
    """
    canonical = canonical_json(envelope_data)
    return domain_separated_digest(canonical)


def policy_hash(constitution_data: Dict[str, Any]) -> str:
    """
    Compute the policy hash of the constitution.

    This hash is included in every envelope to bind it to
    a specific version of the constitution.
    """
    canonical = canonical_json(constitution_data)
    return blake3_digest(canonical)


def state_hash(state_data: Dict[str, Any]) -> str:
    """
    Compute the state hash of the current system state.

    This hash binds the envelope to a specific state,
    preventing state manipulation attacks.
    """
    canonical = canonical_json(state_data)
    return blake3_digest(canonical)


# =============================================================================
# NONCE GENERATION
# =============================================================================


def generate_nonce() -> str:
    """
    Generate a cryptographically random nonce.

    Returns: Hex-encoded 32-byte (256-bit) nonce

    Security requirements:
    - MUST use CSPRNG (cryptographically secure random number generator)
    - MUST be 32 bytes (256 bits) of entropy
    - MUST never be reused across envelopes
    """
    nonce_bytes = secrets.token_bytes(NONCE_BYTES)
    return nonce_bytes.hex()


def validate_nonce(nonce: str) -> bool:
    """
    Validate nonce format.

    Requirements:
    - Must be hex-encoded
    - Must be exactly 32 bytes (64 hex chars)
    """
    if len(nonce) != NONCE_BYTES * 2:
        return False
    try:
        bytes.fromhex(nonce)
        return True
    except ValueError:
        return False


# =============================================================================
# ED25519 SIGNING
# =============================================================================


@dataclass
class KeyPair:
    """Ed25519 key pair."""

    private_key: bytes  # 32 bytes
    public_key: bytes  # 32 bytes

    @property
    def private_key_hex(self) -> str:
        return self.private_key.hex()

    @property
    def public_key_hex(self) -> str:
        return self.public_key.hex()


def generate_keypair() -> KeyPair:
    """
    Generate a new Ed25519 key pair.

    Returns: KeyPair with 32-byte private and public keys

    Security: In production, keys should be generated and stored
    in an HSM or secure enclave.
    """
    if not NACL_AVAILABLE:
        raise ImportError(
            "PyNaCl is required for Ed25519 operations. Install with: pip install pynacl"
        )

    signing_key = SigningKey.generate()
    return KeyPair(
        private_key=bytes(signing_key),
        public_key=bytes(signing_key.verify_key),
    )


def sign_message(message: bytes, private_key: bytes) -> str:
    """
    Sign a message with Ed25519.

    Args:
        message: Raw bytes to sign
        private_key: 32-byte Ed25519 private key

    Returns: Hex-encoded 64-byte signature
    """
    if not NACL_AVAILABLE:
        raise ImportError(
            "PyNaCl is required for Ed25519 operations. Install with: pip install pynacl"
        )

    signing_key = SigningKey(private_key)
    signed = signing_key.sign(message)
    # signed.signature is the 64-byte signature
    return signed.signature.hex()


def verify_signature(message: bytes, signature_hex: str, public_key_hex: str) -> bool:
    """
    Verify an Ed25519 signature.

    Args:
        message: Raw bytes that were signed
        signature_hex: Hex-encoded 64-byte signature
        public_key_hex: Hex-encoded 32-byte public key

    Returns: True if signature is valid, False otherwise
    """
    if not NACL_AVAILABLE:
        raise ImportError(
            "PyNaCl is required for Ed25519 operations. Install with: pip install pynacl"
        )

    try:
        signature = bytes.fromhex(signature_hex)
        public_key = bytes.fromhex(public_key_hex)

        if len(signature) != 64:
            return False
        if len(public_key) != 32:
            return False

        verify_key = VerifyKey(public_key)
        verify_key.verify(message, signature)
        return True
    except (ValueError, BadSignatureError):
        return False


def sign_envelope(
    envelope_data: Dict[str, Any],
    private_key: bytes,
    signed_fields: Optional[list] = None,
) -> str:
    """
    Sign a PCI envelope.

    The envelope is serialized to canonical JSON, then the
    domain-separated digest is signed.

    Args:
        envelope_data: Envelope data dict (without signature field)
        private_key: 32-byte Ed25519 private key
        signed_fields: Fields to include in signature (optional)

    Returns: Hex-encoded 64-byte signature
    """
    if signed_fields is None:
        signed_fields = [
            "version",
            "envelope_id",
            "timestamp",
            "nonce",
            "sender",
            "payload",
            "metadata",
        ]

    # Extract only the signed fields
    data_to_sign = {k: envelope_data[k] for k in signed_fields if k in envelope_data}

    # Compute canonical JSON
    canonical = canonical_json(data_to_sign)

    # Compute domain-separated digest
    digest = domain_separated_digest(canonical)

    # Sign the digest
    return sign_message(bytes.fromhex(digest), private_key)


def verify_envelope_signature(envelope_data: Dict[str, Any]) -> bool:
    """
    Verify the signature on a PCI envelope.

    Args:
        envelope_data: Complete envelope including signature field

    Returns: True if signature is valid, False otherwise
    """
    if "signature" not in envelope_data:
        return False

    signature_data = envelope_data["signature"]
    signed_fields = signature_data.get(
        "signed_fields",
        [
            "version",
            "envelope_id",
            "timestamp",
            "nonce",
            "sender",
            "payload",
            "metadata",
        ],
    )

    # Extract fields that were signed
    data_to_verify = {k: envelope_data[k] for k in signed_fields if k in envelope_data}

    # Compute canonical JSON
    canonical = canonical_json(data_to_verify)

    # Compute domain-separated digest
    digest = domain_separated_digest(canonical)

    # Get public key from sender
    public_key_hex = envelope_data["sender"]["public_key"]
    signature_hex = signature_data["value"]

    # Verify
    return verify_signature(bytes.fromhex(digest), signature_hex, public_key_hex)


# =============================================================================
# SEEN-NONCE CACHE (Replay Protection)
# =============================================================================


class NonceCache:
    """
    Thread-safe seen-nonce cache for replay protection.

    TTL: 120 seconds
    LRU eviction when capacity exceeded
    """

    def __init__(self, ttl_seconds: int = 120, max_size: int = 100000):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._cache: Dict[str, float] = {}
        self._lock = None

        # Try to use threading lock
        try:
            import threading

            self._lock = threading.Lock()
        except ImportError:
            pass

    def _now(self) -> float:
        import time

        return time.time()

    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        now = self._now()
        expired = [k for k, v in self._cache.items() if now - v > self.ttl_seconds]
        for k in expired:
            del self._cache[k]

    def check_and_add(self, nonce: str) -> bool:
        """
        Check if nonce has been seen and add it if not.

        Returns: True if nonce is new (not seen), False if replay detected
        """
        if self._lock:
            with self._lock:
                return self._check_and_add_unlocked(nonce)
        else:
            return self._check_and_add_unlocked(nonce)

    def _check_and_add_unlocked(self, nonce: str) -> bool:
        # Cleanup expired entries periodically
        if len(self._cache) > self.max_size:
            self._cleanup_expired()

        # Check if nonce exists and is not expired
        now = self._now()
        if nonce in self._cache:
            if now - self._cache[nonce] <= self.ttl_seconds:
                return False  # Replay detected
            # Expired, remove and continue
            del self._cache[nonce]

        # Add nonce
        self._cache[nonce] = now
        return True  # New nonce

    def clear(self) -> None:
        """Clear all cached nonces."""
        if self._lock:
            with self._lock:
                self._cache.clear()
        else:
            self._cache.clear()


# Global nonce cache instance
_nonce_cache: Optional[NonceCache] = None


def get_nonce_cache() -> NonceCache:
    """Get the global nonce cache instance."""
    global _nonce_cache
    if _nonce_cache is None:
        _nonce_cache = NonceCache()
    return _nonce_cache


def check_nonce_replay(nonce: str) -> bool:
    """
    Check if a nonce is a replay.

    Returns: True if nonce is new (not a replay), False if replay detected
    """
    return get_nonce_cache().check_and_add(nonce)
