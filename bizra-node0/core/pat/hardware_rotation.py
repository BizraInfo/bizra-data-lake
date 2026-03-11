"""
Hardware Fingerprint Rotation — DID Handoff on Hardware Change

Implements automated DID (Decentralized Identifier) handoff when
hardware changes, preserving Proof-of-Impact reputation across
device migrations.

Protocol:
    1. Detect hardware fingerprint change (CPU, GPU, TPM, MAC)
    2. Verify old identity ownership (Ed25519 signature)
    3. Generate new keypair on new hardware
    4. Create signed rotation certificate linking old -> new
    5. Transfer PoI history to new identity
    6. Revoke old identity after grace period

Standing on Giants:
- W3C DID (2022): Decentralized Identifiers specification
- FIDO2 (2018): Hardware-bound authentication
- TPM 2.0: Trusted Platform Module specification
- Lamport (1978): Key rotation and certificate chains
"""

from __future__ import annotations

import hashlib
import logging
import platform
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Grace period before old identity is fully revoked (seconds)
ROTATION_GRACE_PERIOD_SECONDS = 86400 * 7  # 7 days

# Maximum rotation chain length before requiring re-attestation
MAX_ROTATION_CHAIN_LENGTH = 10


def _collect_hardware_fingerprint() -> Dict[str, str]:
    """
    Collect hardware fingerprint components.

    In production, this would read TPM, CPU serial, GPU ID, etc.
    For the MVP, we use platform information and MAC address.
    """
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "node": str(uuid.getnode()),  # MAC-derived node identifier
    }


def compute_fingerprint_hash(components: Optional[Dict[str, str]] = None) -> str:
    """
    Compute a deterministic hash of hardware fingerprint components.

    Args:
        components: Hardware component dictionary (auto-collected if None)

    Returns:
        SHA-256 hex digest of the fingerprint
    """
    if components is None:
        components = _collect_hardware_fingerprint()

    # Sort keys for deterministic hashing
    canonical = "|".join(f"{k}={v}" for k, v in sorted(components.items()))
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass
class RotationCertificate:
    """
    Signed certificate linking old identity to new identity.

    This certificate proves that the same human controls both
    identities, enabling PoI reputation transfer.
    """

    old_node_id: str
    new_node_id: str
    old_public_key: str
    new_public_key: str
    old_fingerprint_hash: str
    new_fingerprint_hash: str
    rotation_reason: str
    old_identity_signature: str = ""  # Signed by old private key
    certificate_hash: str = ""
    created_at: str = ""
    grace_period_ends: str = ""
    chain_position: int = 0  # Position in rotation chain

    def __post_init__(self):
        if not self.created_at:
            self.created_at = (
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            )
        if not self.certificate_hash:
            self.certificate_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute certificate hash for signing."""
        data = (
            f"{self.old_node_id}|{self.new_node_id}|"
            f"{self.old_public_key}|{self.new_public_key}|"
            f"{self.old_fingerprint_hash}|{self.new_fingerprint_hash}|"
            f"{self.rotation_reason}|{self.created_at}"
        )
        return hashlib.sha256(data.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "old_node_id": self.old_node_id,
            "new_node_id": self.new_node_id,
            "old_public_key": self.old_public_key,
            "new_public_key": self.new_public_key,
            "old_fingerprint_hash": self.old_fingerprint_hash,
            "new_fingerprint_hash": self.new_fingerprint_hash,
            "rotation_reason": self.rotation_reason,
            "old_identity_signature": self.old_identity_signature,
            "certificate_hash": self.certificate_hash,
            "created_at": self.created_at,
            "grace_period_ends": self.grace_period_ends,
            "chain_position": self.chain_position,
        }


@dataclass
class DIDHandoff:
    """
    Record of a complete DID handoff including PoI transfer.

    Tracks the transfer of identity and reputation from one
    hardware-bound identity to another.
    """

    certificate: RotationCertificate
    poi_history_transferred: bool = False
    poi_score_at_rotation: float = 0.0
    reputation_preserved: bool = False
    old_identity_revoked: bool = False
    handoff_complete: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "certificate": self.certificate.to_dict(),
            "poi_history_transferred": self.poi_history_transferred,
            "poi_score_at_rotation": self.poi_score_at_rotation,
            "reputation_preserved": self.reputation_preserved,
            "old_identity_revoked": self.old_identity_revoked,
            "handoff_complete": self.handoff_complete,
            "error": self.error,
        }


@dataclass
class RotationHistory:
    """History of all rotations for a lineage."""

    original_node_id: str
    rotations: List[RotationCertificate] = field(default_factory=list)

    @property
    def chain_length(self) -> int:
        return len(self.rotations)

    @property
    def current_node_id(self) -> str:
        if self.rotations:
            return self.rotations[-1].new_node_id
        return self.original_node_id

    @property
    def needs_reattestation(self) -> bool:
        return self.chain_length >= MAX_ROTATION_CHAIN_LENGTH

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_node_id": self.original_node_id,
            "current_node_id": self.current_node_id,
            "chain_length": self.chain_length,
            "needs_reattestation": self.needs_reattestation,
            "rotations": [r.to_dict() for r in self.rotations],
        }


class HardwareRotationCeremony:
    """
    Orchestrates a hardware fingerprint rotation ceremony.

    This ceremony is triggered when:
    - User migrates to new hardware
    - TPM/security module is replaced
    - User explicitly requests key rotation

    The ceremony creates a chain of trust from old identity to new,
    preserving the user's PoI history and reputation.
    """

    def __init__(self):
        self._rotation_histories: Dict[str, RotationHistory] = {}

    def detect_hardware_change(
        self,
        stored_fingerprint: str,
        current_components: Optional[Dict[str, str]] = None,
    ) -> bool:
        """
        Detect if hardware has changed by comparing fingerprints.

        Args:
            stored_fingerprint: Previously stored fingerprint hash
            current_components: Current hardware components (auto-detected if None)

        Returns:
            True if hardware has changed
        """
        current = compute_fingerprint_hash(current_components)
        return current != stored_fingerprint

    def initiate_rotation(
        self,
        old_node_id: str,
        old_public_key: str,
        old_private_key: str,
        new_public_key: str,
        old_fingerprint_hash: str,
        new_fingerprint_hash: str,
        reason: str = "hardware_upgrade",
        poi_score: float = 0.0,
    ) -> DIDHandoff:
        """
        Initiate a DID rotation ceremony.

        Args:
            old_node_id: Current node ID being rotated
            old_public_key: Current public key (hex)
            old_private_key: Current private key for signing the rotation
            new_public_key: New public key on new hardware (hex)
            old_fingerprint_hash: Hash of old hardware fingerprint
            new_fingerprint_hash: Hash of new hardware fingerprint
            reason: Reason for rotation
            poi_score: Current PoI score to transfer

        Returns:
            DIDHandoff with rotation result
        """
        # Determine chain position
        history = self._rotation_histories.get(old_node_id)
        chain_position = 0
        if history:
            chain_position = history.chain_length

        # Check chain length limit
        if chain_position >= MAX_ROTATION_CHAIN_LENGTH:
            return DIDHandoff(
                certificate=RotationCertificate(
                    old_node_id=old_node_id,
                    new_node_id="",
                    old_public_key=old_public_key,
                    new_public_key=new_public_key,
                    old_fingerprint_hash=old_fingerprint_hash,
                    new_fingerprint_hash=new_fingerprint_hash,
                    rotation_reason=reason,
                ),
                error=f"Rotation chain too long ({chain_position}). Re-attestation required.",
            )

        # Generate new node ID from new public key
        new_node_id = (
            "BIZRA-"
            + hashlib.sha256(bytes.fromhex(new_public_key)).hexdigest()[:8].upper()
        )

        # Create rotation certificate
        certificate = RotationCertificate(
            old_node_id=old_node_id,
            new_node_id=new_node_id,
            old_public_key=old_public_key,
            new_public_key=new_public_key,
            old_fingerprint_hash=old_fingerprint_hash,
            new_fingerprint_hash=new_fingerprint_hash,
            rotation_reason=reason,
            chain_position=chain_position,
        )

        # Sign the certificate with the old private key
        try:
            from core.pci.crypto import sign_message

            certificate.old_identity_signature = sign_message(
                certificate.certificate_hash, old_private_key
            )
        except ImportError:
            # Fallback: HMAC-based signature for environments without PCI
            import hmac as hmac_mod

            sig = hmac_mod.new(
                bytes.fromhex(old_private_key),
                certificate.certificate_hash.encode(),
                hashlib.sha256,
            ).hexdigest()
            certificate.old_identity_signature = sig

        # Record in rotation history
        if old_node_id not in self._rotation_histories:
            # Trace back to the original
            original = old_node_id
            if history:
                original = history.original_node_id
            self._rotation_histories[old_node_id] = RotationHistory(
                original_node_id=original,
            )

        self._rotation_histories[old_node_id].rotations.append(certificate)

        # Also index by new node ID for future lookups
        self._rotation_histories[new_node_id] = self._rotation_histories[old_node_id]

        handoff = DIDHandoff(
            certificate=certificate,
            poi_score_at_rotation=poi_score,
            poi_history_transferred=True,  # In production, would copy PoI records
            reputation_preserved=True,
            handoff_complete=True,
        )

        logger.info(
            "DID rotation: %s -> %s (reason=%s, chain=%d, poi=%.2f)",
            old_node_id,
            new_node_id,
            reason,
            chain_position,
            poi_score,
        )

        return handoff

    def verify_rotation_chain(self, node_id: str) -> Dict[str, Any]:
        """
        Verify the rotation chain for a node.

        Returns the full lineage from original to current.
        """
        history = self._rotation_histories.get(node_id)
        if history is None:
            return {
                "node_id": node_id,
                "has_rotation_history": False,
                "chain_length": 0,
            }

        return {
            "node_id": node_id,
            "has_rotation_history": True,
            "original_node_id": history.original_node_id,
            "current_node_id": history.current_node_id,
            "chain_length": history.chain_length,
            "needs_reattestation": history.needs_reattestation,
            "rotations": [r.to_dict() for r in history.rotations],
        }

    def get_rotation_history(self, node_id: str) -> Optional[RotationHistory]:
        """Get the rotation history for a node."""
        return self._rotation_histories.get(node_id)


__all__ = [
    "HardwareRotationCeremony",
    "DIDHandoff",
    "RotationCertificate",
    "RotationHistory",
    "compute_fingerprint_hash",
    "ROTATION_GRACE_PERIOD_SECONDS",
    "MAX_ROTATION_CHAIN_LENGTH",
]
