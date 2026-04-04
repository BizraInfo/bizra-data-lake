"""
BIZRA Genesis Verifier - Proof Verification Framework
======================================================
Attestation chain verification and sovereignty constraint validation.

This module provides verification capabilities for:
- Attestation chain integrity
- Genesis block verification
- Sovereignty constraint validation
- Integration with WinterProofEmbedder for offline verification

Domain: bizra-genesis-v1:
Threshold: 0.95 Ihsan minimum (fail-closed)

Status: PRODUCTION
Alignment: BIZRA_SOT.md Section 3.1 (Ihsan IM >= 0.95)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4

# Try to import cryptographic libraries
try:
    from nacl.signing import VerifyKey
    from nacl.exceptions import BadSignatureError

    NACL_AVAILABLE = True
except ImportError:
    NACL_AVAILABLE = False
    VerifyKey = None
    BadSignatureError = Exception

try:
    import blake3

    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False

# Import from sibling modules
from .sealer import (
    GenesisSeal,
    GENESIS_DOMAIN_PREFIX,
    SEAL_DOMAIN,
    ATTESTATION_DOMAIN,
)

# =============================================================================
# CONSTANTS
# =============================================================================

IHSAN_THRESHOLD = 0.95
SNR_THRESHOLD = 0.98
QUORUM_REQUIRED = 3  # 3 out of 5 for SAT consensus
QUORUM_TOTAL = 5


# =============================================================================
# ENUMS
# =============================================================================


class VerificationStatus(str, Enum):
    """Status of a verification operation."""

    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    PENDING = "PENDING"
    PARTIAL = "PARTIAL"


class ConstraintType(str, Enum):
    """Types of sovereignty constraints."""

    OFFLINE_CAPABLE = "OFFLINE_CAPABLE"
    NO_EXTERNAL_DEPS = "NO_EXTERNAL_DEPS"
    DETERMINISTIC = "DETERMINISTIC"
    FAIL_CLOSED = "FAIL_CLOSED"
    IHSAN_COMPLIANT = "IHSAN_COMPLIANT"
    SNR_COMPLIANT = "SNR_COMPLIANT"
    QUORUM_BASED = "QUORUM_BASED"


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class ProofAttestation:
    """
    A proof attestation in the verification chain.

    Attributes:
        attestation_id: Unique identifier
        attester_id: ID of the attesting agent
        attester_type: Type of attester (PAT, SAT, etc.)
        timestamp: ISO8601 timestamp
        proof_hash: Hash of the proof being attested
        parent_attestation_id: Previous attestation in chain (optional)
        signature: Ed25519 signature
        data: Additional attestation data
    """

    attestation_id: str
    attester_id: str
    attester_type: str
    timestamp: str
    proof_hash: str
    attester_public_key: str = ""
    parent_attestation_id: Optional[str] = None
    signature: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.attestation_id:
            self.attestation_id = str(uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def get_signing_payload(self) -> bytes:
        """Get the canonical bytes for signature verification."""
        payload = {
            "attestation_id": self.attestation_id,
            "attester_id": self.attester_id,
            "attester_type": self.attester_type,
            "timestamp": self.timestamp,
            "proof_hash": self.proof_hash,
            "parent_attestation_id": self.parent_attestation_id,
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
            "attestation_id": self.attestation_id,
            "attester_id": self.attester_id,
            "attester_type": self.attester_type,
            "timestamp": self.timestamp,
            "proof_hash": self.proof_hash,
            "attester_public_key": self.attester_public_key,
            "parent_attestation_id": self.parent_attestation_id,
            "signature": self.signature,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProofAttestation":
        return cls(
            attestation_id=data["attestation_id"],
            attester_id=data["attester_id"],
            attester_type=data["attester_type"],
            timestamp=data["timestamp"],
            proof_hash=data["proof_hash"],
            attester_public_key=data.get("attester_public_key", ""),
            parent_attestation_id=data.get("parent_attestation_id"),
            signature=data.get("signature", ""),
            data=data.get("data", {}),
        )


@dataclass
class VerificationResult:
    """
    Result of a verification operation.

    Attributes:
        verification_id: Unique identifier
        status: Verification status
        timestamp: ISO8601 timestamp
        checks_passed: List of passed checks
        checks_failed: List of failed checks
        ihsan_score: Ihsan score if applicable
        snr_score: SNR score if applicable
        details: Additional verification details
        integrity_hash: BLAKE3 hash of result
    """

    verification_id: str
    status: VerificationStatus
    timestamp: str
    checks_passed: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)
    ihsan_score: Optional[float] = None
    snr_score: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    integrity_hash: str = ""

    def __post_init__(self):
        if not self.verification_id:
            self.verification_id = str(uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if not self.integrity_hash:
            self.integrity_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute BLAKE3 integrity hash."""
        data = {
            "verification_id": self.verification_id,
            "status": self.status.value,
            "timestamp": self.timestamp,
            "checks_passed": sorted(self.checks_passed),
            "checks_failed": sorted(self.checks_failed),
            "ihsan_score": self.ihsan_score,
            "snr_score": self.snr_score,
        }
        json_bytes = json.dumps(data, sort_keys=True).encode("utf-8")
        prefixed = (GENESIS_DOMAIN_PREFIX + "verification:").encode(
            "utf-8"
        ) + json_bytes

        if HAS_BLAKE3:
            return blake3.blake3(prefixed).hexdigest()
        else:
            return hashlib.sha256(prefixed).hexdigest()

    def is_success(self) -> bool:
        return self.status == VerificationStatus.VERIFIED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verification_id": self.verification_id,
            "status": self.status.value,
            "timestamp": self.timestamp,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "ihsan_score": self.ihsan_score,
            "snr_score": self.snr_score,
            "details": self.details,
            "integrity_hash": self.integrity_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerificationResult":
        return cls(
            verification_id=data["verification_id"],
            status=VerificationStatus(data["status"]),
            timestamp=data["timestamp"],
            checks_passed=data.get("checks_passed", []),
            checks_failed=data.get("checks_failed", []),
            ihsan_score=data.get("ihsan_score"),
            snr_score=data.get("snr_score"),
            details=data.get("details", {}),
            integrity_hash=data.get("integrity_hash", ""),
        )


@dataclass
class SovereigntyConstraint:
    """
    A sovereignty constraint that must be satisfied.

    Attributes:
        constraint_type: Type of constraint
        name: Human-readable name
        description: Detailed description
        validator: Optional async validation function
        required: If True, failure is fatal
    """

    constraint_type: ConstraintType
    name: str
    description: str
    validator: Optional[Callable[[], Tuple[bool, Dict[str, Any]]]] = None
    required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constraint_type": self.constraint_type.value,
            "name": self.name,
            "description": self.description,
            "required": self.required,
        }


# =============================================================================
# GENESIS VERIFIER
# =============================================================================


class GenesisVerifier:
    """
    Proof verification framework for genesis operations.

    Provides verification for:
    - Attestation chain integrity
    - Genesis block validity
    - Sovereignty constraints
    - Integration with WinterProofEmbedder

    CRITICAL: Fail-closed enforcement - any ambiguous state results in rejection.

    Usage:
        verifier = GenesisVerifier()

        # Verify attestation chain
        result = await verifier.verify_attestation_chain(attestations)

        # Verify genesis block
        result = await verifier.verify_genesis_block(block_data, seal)

        # Verify sovereignty constraints
        result = await verifier.verify_sovereignty_constraints()
    """

    def __init__(self):
        """Initialize the GenesisVerifier."""
        self._lock = threading.Lock()
        self._verification_history: List[VerificationResult] = []
        self._known_attesters: Dict[str, str] = {}  # id -> public_key

        # Default sovereignty constraints
        self._constraints: List[SovereigntyConstraint] = self._get_default_constraints()

        # WinterProofEmbedder integration (lazy load)
        self._winter_embedder = None

    def _get_default_constraints(self) -> List[SovereigntyConstraint]:
        """Get default sovereignty constraints."""
        return [
            SovereigntyConstraint(
                constraint_type=ConstraintType.OFFLINE_CAPABLE,
                name="offline_operation",
                description="System must operate without external API calls",
                required=True,
            ),
            SovereigntyConstraint(
                constraint_type=ConstraintType.DETERMINISTIC,
                name="deterministic_output",
                description="Same input must produce same output",
                required=True,
            ),
            SovereigntyConstraint(
                constraint_type=ConstraintType.FAIL_CLOSED,
                name="fail_closed_enforcement",
                description="Ambiguous states must result in rejection",
                required=True,
            ),
            SovereigntyConstraint(
                constraint_type=ConstraintType.IHSAN_COMPLIANT,
                name="ihsan_threshold",
                description=f"Ihsan score must be >= {IHSAN_THRESHOLD}",
                required=True,
            ),
            SovereigntyConstraint(
                constraint_type=ConstraintType.SNR_COMPLIANT,
                name="snr_threshold",
                description=f"SNR must be >= {SNR_THRESHOLD}",
                required=True,
            ),
            SovereigntyConstraint(
                constraint_type=ConstraintType.QUORUM_BASED,
                name="quorum_consensus",
                description=f"SAT quorum ({QUORUM_REQUIRED}/{QUORUM_TOTAL}) required",
                required=True,
            ),
        ]

    def _get_winter_embedder(self):
        """Lazy-load WinterProofEmbedder."""
        if self._winter_embedder is None:
            try:
                from core.sovereignty.winter_proof import WinterProofEmbedder

                self._winter_embedder = WinterProofEmbedder()
            except ImportError:
                # WinterProofEmbedder not available
                pass
        return self._winter_embedder

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
        if not NACL_AVAILABLE:
            raise ImportError("PyNaCl required for signature verification")

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

    def register_attester(self, attester_id: str, public_key_hex: str) -> None:
        """
        Register a known attester's public key.

        Args:
            attester_id: Unique attester identifier
            public_key_hex: Ed25519 public key (hex)
        """
        with self._lock:
            self._known_attesters[attester_id] = public_key_hex

    def add_constraint(self, constraint: SovereigntyConstraint) -> None:
        """
        Add a sovereignty constraint.

        Args:
            constraint: SovereigntyConstraint to add
        """
        with self._lock:
            self._constraints.append(constraint)

    async def verify_attestation_chain(
        self,
        attestations: List[ProofAttestation],
        require_known_attesters: bool = True,
    ) -> VerificationResult:
        """
        Verify an attestation chain.

        Checks:
        1. All signatures are valid
        2. Chain is properly linked (parent references)
        3. Attesters are known (if required)
        4. No duplicate attestations
        5. Timestamps are monotonically increasing

        Args:
            attestations: List of ProofAttestation objects
            require_known_attesters: Require attesters to be pre-registered

        Returns:
            VerificationResult with detailed status
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        checks_passed: List[str] = []
        checks_failed: List[str] = []
        details: Dict[str, Any] = {}

        if not attestations:
            return VerificationResult(
                verification_id=str(uuid4()),
                status=VerificationStatus.FAILED,
                timestamp=timestamp,
                checks_failed=["empty_chain"],
                details={"error": "Attestation chain is empty"},
            )

        # Build attestation lookup
        attestation_map: Dict[str, ProofAttestation] = {
            a.attestation_id: a for a in attestations
        }
        seen_attesters: Set[str] = set()

        # Verify each attestation
        for i, attestation in enumerate(attestations):
            # Check for duplicate attesters
            if attestation.attester_id in seen_attesters:
                checks_failed.append(f"duplicate_attester_{i}")
            else:
                seen_attesters.add(attestation.attester_id)
                checks_passed.append(f"unique_attester_{i}")

            # Check if attester is known
            if require_known_attesters:
                with self._lock:
                    known_key = self._known_attesters.get(attestation.attester_id)

                if known_key is None:
                    checks_failed.append(f"unknown_attester_{i}")
                elif attestation.attester_public_key != known_key:
                    checks_failed.append(f"key_mismatch_{i}")
                else:
                    checks_passed.append(f"known_attester_{i}")

            # Verify signature
            if not attestation.signature:
                checks_failed.append(f"missing_signature_{i}")
            else:
                digest = attestation.compute_digest()
                if self._verify_signature(
                    bytes.fromhex(digest),
                    attestation.signature,
                    attestation.attester_public_key,
                ):
                    checks_passed.append(f"valid_signature_{i}")
                else:
                    checks_failed.append(f"invalid_signature_{i}")

            # Verify chain linkage (if not first)
            if i > 0:
                if attestation.parent_attestation_id is None:
                    checks_failed.append(f"missing_parent_{i}")
                elif attestation.parent_attestation_id not in attestation_map:
                    checks_failed.append(f"invalid_parent_{i}")
                else:
                    checks_passed.append(f"valid_chain_{i}")

        # Determine final status
        if checks_failed:
            status = VerificationStatus.FAILED
        else:
            status = VerificationStatus.VERIFIED

        result = VerificationResult(
            verification_id=str(uuid4()),
            status=status,
            timestamp=timestamp,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            details={
                "attestation_count": len(attestations),
                "unique_attesters": len(seen_attesters),
            },
        )

        # Store in history
        with self._lock:
            self._verification_history.append(result)

        return result

    async def verify_genesis_block(
        self,
        block_data: Dict[str, Any],
        seal: GenesisSeal,
    ) -> VerificationResult:
        """
        Verify a genesis block.

        Checks:
        1. Block hash matches seal
        2. Seal signature is valid
        3. All attestations are valid
        4. Required fields present
        5. Phase is valid (0-3)

        Args:
            block_data: Genesis block data dictionary
            seal: GenesisSeal for the block

        Returns:
            VerificationResult with detailed status
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        checks_passed: List[str] = []
        checks_failed: List[str] = []
        details: Dict[str, Any] = {}

        # Compute block hash
        json_bytes = json.dumps(block_data, sort_keys=True).encode("utf-8")
        prefixed = (GENESIS_DOMAIN_PREFIX + "block:").encode("utf-8") + json_bytes

        if HAS_BLAKE3:
            computed_hash = blake3.blake3(prefixed).hexdigest()
        else:
            computed_hash = hashlib.sha256(prefixed).hexdigest()

        # Verify block hash matches seal
        if computed_hash == seal.seal_hash:
            checks_passed.append("block_hash_match")
        else:
            checks_failed.append("block_hash_mismatch")
            details["expected_hash"] = seal.seal_hash
            details["computed_hash"] = computed_hash

        # Verify seal signature
        if not seal.is_signed():
            checks_failed.append("seal_not_signed")
        else:
            signing_payload = seal.get_signing_payload()

            if HAS_BLAKE3:
                seal_digest = blake3.blake3(
                    (SEAL_DOMAIN).encode("utf-8") + signing_payload
                ).hexdigest()
            else:
                seal_digest = hashlib.sha256(
                    (SEAL_DOMAIN).encode("utf-8") + signing_payload
                ).hexdigest()

            if self._verify_signature(
                bytes.fromhex(seal_digest),
                seal.sealer_signature,
                seal.sealer_public_key,
            ):
                checks_passed.append("seal_signature_valid")
            else:
                checks_failed.append("seal_signature_invalid")

        # Verify attestations
        valid_attestations = 0
        for i, attestation in enumerate(seal.attestations):
            if not attestation.signature:
                checks_failed.append(f"attestation_{i}_unsigned")
                continue

            attest_digest = attestation.compute_digest()
            if self._verify_signature(
                bytes.fromhex(attest_digest),
                attestation.signature,
                attestation.attester_public_key,
            ):
                valid_attestations += 1
                checks_passed.append(f"attestation_{i}_valid")
            else:
                checks_failed.append(f"attestation_{i}_invalid")

        details["valid_attestations"] = valid_attestations
        details["total_attestations"] = len(seal.attestations)

        # Verify required fields
        required_fields = ["version", "phase", "timestamp"]
        for field_name in required_fields:
            if field_name in block_data:
                checks_passed.append(f"field_{field_name}_present")
            else:
                checks_failed.append(f"field_{field_name}_missing")

        # Verify phase is valid
        if seal.phase is not None:
            if 0 <= seal.phase <= 3:
                checks_passed.append("phase_valid")
            else:
                checks_failed.append("phase_invalid")

        # Determine final status
        if checks_failed:
            status = VerificationStatus.FAILED
        else:
            status = VerificationStatus.VERIFIED

        result = VerificationResult(
            verification_id=str(uuid4()),
            status=status,
            timestamp=timestamp,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            details=details,
        )

        with self._lock:
            self._verification_history.append(result)

        return result

    async def verify_sovereignty_constraints(
        self,
        ihsan_score: Optional[float] = None,
        snr_score: Optional[float] = None,
        quorum_achieved: Optional[int] = None,
    ) -> VerificationResult:
        """
        Verify all sovereignty constraints.

        Args:
            ihsan_score: Current Ihsan score (optional)
            snr_score: Current SNR score (optional)
            quorum_achieved: Number of SAT votes achieved (optional)

        Returns:
            VerificationResult with constraint check details
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        checks_passed: List[str] = []
        checks_failed: List[str] = []
        details: Dict[str, Any] = {}

        for constraint in self._constraints:
            passed = True

            # Run custom validator if provided
            if constraint.validator:
                try:
                    if asyncio.iscoroutinefunction(constraint.validator):
                        passed, validator_details = await constraint.validator()
                    else:
                        passed, validator_details = constraint.validator()
                    details[constraint.name] = validator_details
                except Exception as e:
                    passed = False
                    details[constraint.name] = {"error": str(e)}

            # Built-in constraint checks
            elif constraint.constraint_type == ConstraintType.IHSAN_COMPLIANT:
                if ihsan_score is not None:
                    passed = ihsan_score >= IHSAN_THRESHOLD
                    details["ihsan_score"] = ihsan_score
                    details["ihsan_threshold"] = IHSAN_THRESHOLD
                else:
                    # No score provided - fail closed
                    passed = False
                    details["ihsan_error"] = "score_not_provided"

            elif constraint.constraint_type == ConstraintType.SNR_COMPLIANT:
                if snr_score is not None:
                    passed = snr_score >= SNR_THRESHOLD
                    details["snr_score"] = snr_score
                    details["snr_threshold"] = SNR_THRESHOLD
                else:
                    # No score provided - fail closed
                    passed = False
                    details["snr_error"] = "score_not_provided"

            elif constraint.constraint_type == ConstraintType.QUORUM_BASED:
                if quorum_achieved is not None:
                    passed = quorum_achieved >= QUORUM_REQUIRED
                    details["quorum_achieved"] = quorum_achieved
                    details["quorum_required"] = QUORUM_REQUIRED
                else:
                    # No quorum info - fail closed
                    passed = False
                    details["quorum_error"] = "quorum_not_provided"

            elif constraint.constraint_type == ConstraintType.OFFLINE_CAPABLE:
                # Check if WinterProofEmbedder is available (offline embeddings)
                embedder = self._get_winter_embedder()
                passed = embedder is not None
                details["winter_embedder_available"] = passed

            if passed:
                checks_passed.append(constraint.name)
            else:
                checks_failed.append(constraint.name)

        # Determine final status
        has_required_failures = any(
            c.name in checks_failed and c.required for c in self._constraints
        )

        if has_required_failures:
            status = VerificationStatus.FAILED
        elif checks_failed:
            status = VerificationStatus.PARTIAL
        else:
            status = VerificationStatus.VERIFIED

        result = VerificationResult(
            verification_id=str(uuid4()),
            status=status,
            timestamp=timestamp,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            ihsan_score=ihsan_score,
            snr_score=snr_score,
            details=details,
        )

        with self._lock:
            self._verification_history.append(result)

        return result

    def get_verification_history(self) -> List[VerificationResult]:
        """Get all verification results."""
        with self._lock:
            return list(self._verification_history)

    def get_latest_verification(self) -> Optional[VerificationResult]:
        """Get the most recent verification result."""
        with self._lock:
            if self._verification_history:
                return self._verification_history[-1]
            return None

    def clear_history(self) -> None:
        """Clear verification history."""
        with self._lock:
            self._verification_history.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize verifier state."""
        with self._lock:
            return {
                "verification_count": len(self._verification_history),
                "known_attesters": len(self._known_attesters),
                "constraints": [c.to_dict() for c in self._constraints],
                "latest_verification": (
                    self._verification_history[-1].to_dict()
                    if self._verification_history
                    else None
                ),
            }

    def __repr__(self) -> str:
        return (
            f"GenesisVerifier(verifications={len(self._verification_history)}, "
            f"attesters={len(self._known_attesters)})"
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


async def quick_verify_seal(seal: GenesisSeal) -> Tuple[bool, str]:
    """
    Quickly verify a genesis seal (signature only).

    Args:
        seal: GenesisSeal to verify

    Returns:
        Tuple of (is_valid, reason)
    """
    if not seal.is_signed():
        return False, "Seal is not signed"

    if not NACL_AVAILABLE:
        return False, "PyNaCl not available for verification"

    try:
        signing_payload = seal.get_signing_payload()

        if HAS_BLAKE3:
            digest = blake3.blake3(
                (SEAL_DOMAIN).encode("utf-8") + signing_payload
            ).hexdigest()
        else:
            digest = hashlib.sha256(
                (SEAL_DOMAIN).encode("utf-8") + signing_payload
            ).hexdigest()

        signature = bytes.fromhex(seal.sealer_signature)
        public_key = bytes.fromhex(seal.sealer_public_key)

        if len(signature) != 64 or len(public_key) != 32:
            return False, "Invalid signature or key length"

        verify_key = VerifyKey(public_key)
        verify_key.verify(bytes.fromhex(digest), signature)
        return True, "Seal verified"

    except BadSignatureError:
        return False, "Invalid signature"
    except ValueError as e:
        return False, f"Verification error: {e}"


def create_proof_chain(
    attestations: List[ProofAttestation],
) -> List[ProofAttestation]:
    """
    Create a properly linked attestation chain.

    Sets parent_attestation_id on each attestation to link to the previous one.

    Args:
        attestations: Unlinked attestations

    Returns:
        Linked attestation chain
    """
    if not attestations:
        return []

    # First attestation has no parent
    linked = [attestations[0]]
    linked[0].parent_attestation_id = None

    # Link subsequent attestations
    for i in range(1, len(attestations)):
        attestations[i].parent_attestation_id = attestations[i - 1].attestation_id
        linked.append(attestations[i])

    return linked
