"""
BIZRA Genesis Sealer - Cryptographic Seal Generation
=====================================================
BLAKE3 + Ed25519 seal generation with domain separation.

This module provides cryptographic sealing for genesis phases and blocks,
ensuring tamper-proof evidence chains with proper domain separation.

Domain Prefix: bizra-genesis-v1:
Algorithms: BLAKE3 (hashing), Ed25519 (signing)

Status: PRODUCTION
Alignment: BIZRA_SOT.md Section 3.1 (Ihsan IM >= 0.95)
"""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

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

    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False


# =============================================================================
# CONSTANTS
# =============================================================================

GENESIS_DOMAIN_PREFIX = "bizra-genesis-v1:"
GENESIS_VERSION = "1.0.0"
SEAL_DOMAIN = "bizra-genesis-seal-v1:"
ATTESTATION_DOMAIN = "bizra-genesis-attest-v1:"
NONCE_BYTES = 32


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class SealAttestation:
    """
    An attestation included in a genesis seal.

    Attributes:
        attester_id: Unique identifier of the attesting agent
        attester_public_key: Ed25519 public key (hex)
        timestamp: ISO8601 timestamp of attestation
        data: Attestation-specific data
        signature: Ed25519 signature over attestation (hex)
    """

    attester_id: str
    attester_public_key: str
    timestamp: str
    data: Dict[str, Any] = field(default_factory=dict)
    signature: str = ""

    def get_signing_payload(self) -> bytes:
        """Get the canonical bytes to sign."""
        payload = {
            "attester_id": self.attester_id,
            "attester_public_key": self.attester_public_key,
            "timestamp": self.timestamp,
            "data": self.data,
        }
        json_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        return (ATTESTATION_DOMAIN).encode("utf-8") + json_bytes

    def compute_digest(self) -> str:
        """Compute BLAKE3 digest of attestation."""
        payload = self.get_signing_payload()
        if HAS_BLAKE3:
            return blake3.blake3(payload).hexdigest()
        else:
            return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attester_id": self.attester_id,
            "attester_public_key": self.attester_public_key,
            "timestamp": self.timestamp,
            "data": self.data,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SealAttestation":
        return cls(
            attester_id=data["attester_id"],
            attester_public_key=data["attester_public_key"],
            timestamp=data["timestamp"],
            data=data.get("data", {}),
            signature=data.get("signature", ""),
        )


@dataclass
class GenesisSeal:
    """
    Cryptographic seal for genesis phases and blocks.

    A seal binds together:
    - A unique seal hash (BLAKE3)
    - A timestamp
    - A set of attestations from validators
    - The sealer's Ed25519 signature

    Attributes:
        seal_id: Unique seal identifier
        seal_hash: BLAKE3 hash of sealed content
        timestamp: ISO8601 timestamp
        attestations: List of validator attestations
        sealer_public_key: Ed25519 public key of sealer (hex)
        sealer_signature: Ed25519 signature (hex)
        version: Seal version
        nonce: Random nonce for uniqueness
        phase: Genesis phase being sealed (optional)
        block_hash: Related block hash (optional)
    """

    seal_id: str
    seal_hash: str
    timestamp: str
    attestations: List[SealAttestation] = field(default_factory=list)
    sealer_public_key: str = ""
    sealer_signature: str = ""
    version: str = GENESIS_VERSION
    nonce: str = ""
    phase: Optional[int] = None
    block_hash: Optional[str] = None

    def __post_init__(self):
        if not self.seal_id:
            self.seal_id = str(uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if not self.nonce:
            self.nonce = secrets.token_hex(NONCE_BYTES)

    def get_signing_payload(self) -> bytes:
        """Get the canonical bytes for sealer signature."""
        payload = {
            "seal_id": self.seal_id,
            "seal_hash": self.seal_hash,
            "timestamp": self.timestamp,
            "attestations": [a.to_dict() for a in self.attestations],
            "version": self.version,
            "nonce": self.nonce,
            "phase": self.phase,
            "block_hash": self.block_hash,
        }
        json_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        return (SEAL_DOMAIN).encode("utf-8") + json_bytes

    def compute_integrity_hash(self) -> str:
        """Compute integrity hash of entire seal (including signature)."""
        full_data = {
            "seal_id": self.seal_id,
            "seal_hash": self.seal_hash,
            "timestamp": self.timestamp,
            "attestations": [a.to_dict() for a in self.attestations],
            "sealer_public_key": self.sealer_public_key,
            "sealer_signature": self.sealer_signature,
            "version": self.version,
            "nonce": self.nonce,
            "phase": self.phase,
            "block_hash": self.block_hash,
        }
        json_bytes = json.dumps(full_data, sort_keys=True).encode("utf-8")
        prefixed = (GENESIS_DOMAIN_PREFIX + "integrity:").encode("utf-8") + json_bytes

        if HAS_BLAKE3:
            return blake3.blake3(prefixed).hexdigest()
        else:
            return hashlib.sha256(prefixed).hexdigest()

    def is_signed(self) -> bool:
        """Check if seal has been signed."""
        return bool(self.sealer_signature and self.sealer_public_key)

    def attestation_count(self) -> int:
        """Get number of attestations."""
        return len(self.attestations)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seal_id": self.seal_id,
            "seal_hash": self.seal_hash,
            "timestamp": self.timestamp,
            "attestations": [a.to_dict() for a in self.attestations],
            "sealer_public_key": self.sealer_public_key,
            "sealer_signature": self.sealer_signature,
            "version": self.version,
            "nonce": self.nonce,
            "phase": self.phase,
            "block_hash": self.block_hash,
            "integrity_hash": self.compute_integrity_hash(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenesisSeal":
        return cls(
            seal_id=data["seal_id"],
            seal_hash=data["seal_hash"],
            timestamp=data["timestamp"],
            attestations=[
                SealAttestation.from_dict(a) for a in data.get("attestations", [])
            ],
            sealer_public_key=data.get("sealer_public_key", ""),
            sealer_signature=data.get("sealer_signature", ""),
            version=data.get("version", GENESIS_VERSION),
            nonce=data.get("nonce", ""),
            phase=data.get("phase"),
            block_hash=data.get("block_hash"),
        )


# =============================================================================
# GENESIS SEALER
# =============================================================================


class GenesisSealer:
    """
    Cryptographic sealer for genesis phases and blocks.

    Provides BLAKE3 hashing with domain separation and Ed25519 signing
    for tamper-proof seal generation.

    Attributes:
        sealer_id: Unique identifier for this sealer
        private_key: Ed25519 private key (32 bytes)
        public_key_hex: Ed25519 public key (hex string)

    Usage:
        # Create sealer with private key
        sealer = GenesisSealer(private_key=my_private_key)

        # Create a seal
        seal = sealer.create_seal(content_hash, attestations)

        # Verify a seal
        is_valid = sealer.verify_seal(seal)

        # Seal a specific phase
        seal = sealer.seal_phase(phase_value, attestations)
    """

    def __init__(
        self,
        private_key: Optional[bytes] = None,
        sealer_id: Optional[str] = None,
    ):
        """
        Initialize the GenesisSealer.

        Args:
            private_key: 32-byte Ed25519 private key (generates new if None)
            sealer_id: Unique sealer ID (generates UUID if None)
        """
        self.sealer_id = sealer_id or str(uuid4())
        self._lock = threading.Lock()
        self._seal_count = 0

        if not NACL_AVAILABLE:
            raise ImportError(
                "PyNaCl is required for GenesisSealer. "
                "Install with: pip install pynacl"
            )

        if private_key is None:
            # Generate new key pair
            signing_key = SigningKey.generate()
            self._private_key = bytes(signing_key)
            self._public_key = bytes(signing_key.verify_key)
        else:
            if len(private_key) != 32:
                raise ValueError("Private key must be 32 bytes")
            self._private_key = private_key
            signing_key = SigningKey(private_key)
            self._public_key = bytes(signing_key.verify_key)

        self.public_key_hex = self._public_key.hex()

    def _next_seal_number(self) -> int:
        """Get the next monotonically increasing seal number."""
        with self._lock:
            self._seal_count += 1
            return self._seal_count

    def _domain_hash(self, data: bytes, domain: str = GENESIS_DOMAIN_PREFIX) -> str:
        """
        Compute domain-separated BLAKE3 hash.

        Args:
            data: Bytes to hash
            domain: Domain prefix for separation

        Returns:
            Hex-encoded BLAKE3 digest
        """
        prefixed = domain.encode("utf-8") + data

        if HAS_BLAKE3:
            return blake3.blake3(prefixed).hexdigest()
        else:
            return hashlib.sha256(prefixed).hexdigest()

    def _sign(self, message: bytes) -> str:
        """
        Sign a message with Ed25519.

        Args:
            message: Bytes to sign

        Returns:
            Hex-encoded 64-byte signature
        """
        signing_key = SigningKey(self._private_key)
        signed = signing_key.sign(message)
        return signed.signature.hex()

    def _verify_signature(
        self,
        message: bytes,
        signature_hex: str,
        public_key_hex: str,
    ) -> bool:
        """
        Verify an Ed25519 signature.

        Args:
            message: Original message bytes
            signature_hex: Hex-encoded signature
            public_key_hex: Hex-encoded public key

        Returns:
            True if valid, False otherwise
        """
        try:
            signature = bytes.fromhex(signature_hex)
            public_key = bytes.fromhex(public_key_hex)

            if len(signature) != 64 or len(public_key) != 32:
                return False

            verify_key = VerifyKey(public_key)
            verify_key.verify(message, signature)
            return True
        except (ValueError, BadSignatureError):
            return False

    def create_seal(
        self,
        content_hash: str,
        attestations: Optional[List[SealAttestation]] = None,
        phase: Optional[int] = None,
        block_hash: Optional[str] = None,
    ) -> GenesisSeal:
        """
        Create a new genesis seal.

        Args:
            content_hash: BLAKE3 hash of content being sealed
            attestations: List of validator attestations
            phase: Genesis phase number (optional)
            block_hash: Related block hash (optional)

        Returns:
            Signed GenesisSeal
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        nonce = secrets.token_hex(NONCE_BYTES)

        # Create unsigned seal
        seal = GenesisSeal(
            seal_id=str(uuid4()),
            seal_hash=content_hash,
            timestamp=timestamp,
            attestations=attestations or [],
            version=GENESIS_VERSION,
            nonce=nonce,
            phase=phase,
            block_hash=block_hash,
        )

        # Sign the seal
        signing_payload = seal.get_signing_payload()
        digest = self._domain_hash(signing_payload, SEAL_DOMAIN)
        signature = self._sign(bytes.fromhex(digest))

        seal.sealer_public_key = self.public_key_hex
        seal.sealer_signature = signature

        return seal

    def verify_seal(self, seal: GenesisSeal) -> Tuple[bool, str]:
        """
        Verify a genesis seal.

        Checks:
        1. Seal has signature
        2. Signature is valid Ed25519
        3. All attestation signatures are valid

        Args:
            seal: GenesisSeal to verify

        Returns:
            Tuple of (is_valid, reason)
        """
        # Check seal has signature
        if not seal.is_signed():
            return False, "Seal is not signed"

        # Verify sealer signature
        signing_payload = seal.get_signing_payload()
        digest = self._domain_hash(signing_payload, SEAL_DOMAIN)

        if not self._verify_signature(
            bytes.fromhex(digest),
            seal.sealer_signature,
            seal.sealer_public_key,
        ):
            return False, "Invalid sealer signature"

        # Verify all attestation signatures
        for i, attestation in enumerate(seal.attestations):
            if not attestation.signature:
                return False, f"Attestation {i} has no signature"

            attest_digest = attestation.compute_digest()
            if not self._verify_signature(
                bytes.fromhex(attest_digest),
                attestation.signature,
                attestation.attester_public_key,
            ):
                return False, f"Invalid signature on attestation {i}"

        return True, "Seal verified successfully"

    def seal_phase(
        self,
        phase: int,
        attestations: Optional[List[SealAttestation]] = None,
        phase_data: Optional[Dict[str, Any]] = None,
    ) -> GenesisSeal:
        """
        Create a seal for a specific genesis phase.

        Args:
            phase: Phase number (0-3)
            attestations: List of validator attestations
            phase_data: Additional phase-specific data

        Returns:
            Signed GenesisSeal for the phase
        """
        # Compute phase content hash
        content = {
            "phase": phase,
            "data": phase_data or {},
            "seal_number": self._next_seal_number(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        content_bytes = json.dumps(content, sort_keys=True).encode("utf-8")
        content_hash = self._domain_hash(
            content_bytes, GENESIS_DOMAIN_PREFIX + f"phase{phase}:"
        )

        return self.create_seal(
            content_hash=content_hash,
            attestations=attestations,
            phase=phase,
        )

    def create_attestation(
        self,
        data: Optional[Dict[str, Any]] = None,
    ) -> SealAttestation:
        """
        Create a signed attestation from this sealer.

        Args:
            data: Attestation-specific data

        Returns:
            Signed SealAttestation
        """
        attestation = SealAttestation(
            attester_id=self.sealer_id,
            attester_public_key=self.public_key_hex,
            timestamp=datetime.now(timezone.utc).isoformat(),
            data=data or {},
        )

        # Sign the attestation
        digest = attestation.compute_digest()
        attestation.signature = self._sign(bytes.fromhex(digest))

        return attestation

    def get_public_key(self) -> Tuple[str, bytes]:
        """
        Get the sealer's public key.

        Returns:
            Tuple of (hex_string, raw_bytes)
        """
        return self.public_key_hex, self._public_key

    def to_dict(self) -> Dict[str, Any]:
        """Serialize sealer metadata (excluding private key)."""
        return {
            "sealer_id": self.sealer_id,
            "public_key": self.public_key_hex,
            "seal_count": self._seal_count,
        }

    def __repr__(self) -> str:
        return (
            f"GenesisSealer(id={self.sealer_id}, "
            f"pubkey={self.public_key_hex[:16]}...)"
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def compute_genesis_block_hash(block_data: Dict[str, Any]) -> str:
    """
    Compute the hash of a genesis block.

    Args:
        block_data: Genesis block data dictionary

    Returns:
        BLAKE3 hex digest with domain separation
    """
    json_bytes = json.dumps(block_data, sort_keys=True).encode("utf-8")
    prefixed = (GENESIS_DOMAIN_PREFIX + "block:").encode("utf-8") + json_bytes

    if HAS_BLAKE3:
        return blake3.blake3(prefixed).hexdigest()
    else:
        return hashlib.sha256(prefixed).hexdigest()


def verify_attestation_signature(attestation: SealAttestation) -> bool:
    """
    Verify an attestation signature.

    Args:
        attestation: SealAttestation to verify

    Returns:
        True if signature is valid
    """
    if not NACL_AVAILABLE:
        raise ImportError("PyNaCl required for signature verification")

    if not attestation.signature:
        return False

    try:
        digest = attestation.compute_digest()
        signature = bytes.fromhex(attestation.signature)
        public_key = bytes.fromhex(attestation.attester_public_key)

        if len(signature) != 64 or len(public_key) != 32:
            return False

        verify_key = VerifyKey(public_key)
        verify_key.verify(bytes.fromhex(digest), signature)
        return True
    except (ValueError, BadSignatureError):
        return False
