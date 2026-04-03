"""
BIZRA PCI Protocol — CommitReceipt
==================================
Immutable receipt for every verification decision.

Status: FROZEN — Changes require version bump + test vector update
Semantics: Append-only, monotonically increasing offset
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import threading

from .crypto import (
    canonical_json,
    domain_separated_digest,
    sign_message,
    verify_signature,
)
from .types import (
    PCI_VERSION,
    CommitRef,
    CommitRefType,
    Gate,
    Quorum,
    Verification,
    VerificationTier,
    VerifierSignature,
    generate_receipt_id,
    utc_now_iso,
)


@dataclass
class CommitReceipt:
    """
    Immutable receipt for verification decisions.
    
    Every decision (accept or reject) produces a receipt.
    Receipts are append-only with monotonically increasing offsets.
    """
    version: str
    receipt_id: str
    timestamp: str
    envelope_digest: str
    commit_ref: CommitRef
    verification: Verification
    verifier_set: List[VerifierSignature]
    quorum: Quorum
    audit_digest: str
    policy_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for wire format."""
        return {
            "version": self.version,
            "receipt_id": self.receipt_id,
            "timestamp": self.timestamp,
            "envelope_digest": self.envelope_digest,
            "commit_ref": self.commit_ref.to_dict(),
            "verification": self.verification.to_dict(),
            "verifier_set": [v.to_dict() for v in self.verifier_set],
            "quorum": self.quorum.to_dict(),
            "audit_digest": self.audit_digest,
            "policy_hash": self.policy_hash,
        }

    def to_canonical_bytes(self) -> bytes:
        """Serialize to canonical JSON bytes."""
        return canonical_json(self.to_dict())

    def to_canonical_json(self) -> str:
        """Serialize to canonical JSON string."""
        return self.to_canonical_bytes().decode('utf-8')

    def is_quorum_met(self) -> bool:
        """Check if quorum requirements are satisfied."""
        return self.quorum.is_met()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CommitReceipt":
        """Deserialize from dictionary."""
        return cls(
            version=data["version"],
            receipt_id=data["receipt_id"],
            timestamp=data["timestamp"],
            envelope_digest=data["envelope_digest"],
            commit_ref=CommitRef.from_dict(data["commit_ref"]),
            verification=Verification.from_dict(data["verification"]),
            verifier_set=[VerifierSignature.from_dict(v) for v in data["verifier_set"]],
            quorum=Quorum.from_dict(data["quorum"]),
            audit_digest=data["audit_digest"],
            policy_hash=data["policy_hash"],
        )


# =============================================================================
# RECEIPT GENERATOR
# =============================================================================

class ReceiptGenerator:
    """
    Generates CommitReceipts for verification decisions.
    
    Thread-safe with monotonically increasing offsets.
    """
    
    def __init__(self, ref_type: CommitRefType = CommitRefType.EVENTLOG):
        self._ref_type = ref_type
        self._offset = 0
        self._lock = threading.Lock()

    def _next_offset(self) -> int:
        """Get the next monotonically increasing offset."""
        with self._lock:
            self._offset += 1
            return self._offset

    def create_receipt(
        self,
        envelope_digest: str,
        verification_tier: VerificationTier,
        latency_ms: float,
        gates_passed: List[Gate],
        ihsan_score: float,
        snr_score: float,
        verifier_id: str,
        verifier_public_key: str,
        verifier_private_key: bytes,
        policy_hash: str,
        audit_data: Optional[Dict[str, Any]] = None,
        block_hash: Optional[str] = None,
        quorum_required: int = 1,
    ) -> CommitReceipt:
        """
        Create a new commit receipt.
        
        Args:
            envelope_digest: BLAKE3 digest of the verified envelope
            verification_tier: Verification confidence tier
            latency_ms: Total verification latency in milliseconds
            gates_passed: List of gates that passed
            ihsan_score: Final Ihsān score
            snr_score: Final SNR score
            verifier_id: ID of the verifying SAT agent
            verifier_public_key: Public key of the verifier (hex)
            verifier_private_key: Private key for signing the receipt
            policy_hash: BLAKE3 of the active constitution
            audit_data: Optional audit data for digest
            block_hash: Optional block hash (for blockgraph type)
            quorum_required: Number of verifiers required
        
        Returns: Signed CommitReceipt
        """
        timestamp = utc_now_iso()
        offset = self._next_offset()
        
        # Compute audit digest
        audit_content = {
            "envelope_digest": envelope_digest,
            "gates_passed": [g.value for g in gates_passed],
            "ihsan_score": ihsan_score,
            "snr_score": snr_score,
            "latency_ms": latency_ms,
            "timestamp": timestamp,
            **(audit_data or {}),
        }
        audit_bytes = canonical_json(audit_content)
        audit_digest = domain_separated_digest(audit_bytes, "bizra-pci-audit-v1:")
        
        # Create verifier signature
        verifier_sig_data = {
            "envelope_digest": envelope_digest,
            "audit_digest": audit_digest,
            "timestamp": timestamp,
        }
        verifier_sig_bytes = canonical_json(verifier_sig_data)
        verifier_sig_digest = domain_separated_digest(verifier_sig_bytes, "bizra-pci-verifier-v1:")
        verifier_signature = sign_message(bytes.fromhex(verifier_sig_digest), verifier_private_key)
        
        return CommitReceipt(
            version=PCI_VERSION,
            receipt_id=generate_receipt_id(),
            timestamp=timestamp,
            envelope_digest=envelope_digest,
            commit_ref=CommitRef(
                type=self._ref_type,
                offset=offset,
                block_hash=block_hash,
            ),
            verification=Verification(
                tier=verification_tier,
                latency_ms=latency_ms,
                gates_passed=gates_passed,
                ihsan_score=ihsan_score,
                snr_score=snr_score,
            ),
            verifier_set=[
                VerifierSignature(
                    sat_id=verifier_id,
                    public_key=verifier_public_key,
                    signature=verifier_signature,
                    timestamp=timestamp,
                ),
            ],
            quorum=Quorum(
                required=quorum_required,
                achieved=1,
            ),
            audit_digest=audit_digest,
            policy_hash=policy_hash,
        )

    def add_verifier_signature(
        self,
        receipt: CommitReceipt,
        verifier_id: str,
        verifier_public_key: str,
        verifier_private_key: bytes,
    ) -> CommitReceipt:
        """
        Add another verifier signature to a receipt.
        
        Returns a new receipt with the additional signature and updated quorum.
        """
        timestamp = utc_now_iso()
        
        # Create verifier signature
        verifier_sig_data = {
            "envelope_digest": receipt.envelope_digest,
            "audit_digest": receipt.audit_digest,
            "timestamp": timestamp,
        }
        verifier_sig_bytes = canonical_json(verifier_sig_data)
        verifier_sig_digest = domain_separated_digest(verifier_sig_bytes, "bizra-pci-verifier-v1:")
        verifier_signature = sign_message(bytes.fromhex(verifier_sig_digest), verifier_private_key)
        
        new_verifier_set = receipt.verifier_set + [
            VerifierSignature(
                sat_id=verifier_id,
                public_key=verifier_public_key,
                signature=verifier_signature,
                timestamp=timestamp,
            ),
        ]
        
        return CommitReceipt(
            version=receipt.version,
            receipt_id=receipt.receipt_id,
            timestamp=receipt.timestamp,
            envelope_digest=receipt.envelope_digest,
            commit_ref=receipt.commit_ref,
            verification=receipt.verification,
            verifier_set=new_verifier_set,
            quorum=Quorum(
                required=receipt.quorum.required,
                achieved=len(new_verifier_set),
            ),
            audit_digest=receipt.audit_digest,
            policy_hash=receipt.policy_hash,
        )


# =============================================================================
# RECEIPT STORE (Append-Only)
# =============================================================================

class ReceiptStore:
    """
    Append-only receipt store.
    
    In production, this would be backed by a persistent store
    (Redis, PostgreSQL, or BlockGraph).
    """
    
    def __init__(self):
        self._receipts: List[CommitReceipt] = []
        self._by_envelope_digest: Dict[str, CommitReceipt] = {}
        self._by_receipt_id: Dict[str, CommitReceipt] = {}
        self._lock = threading.Lock()

    def append(self, receipt: CommitReceipt) -> None:
        """Append a receipt to the store."""
        with self._lock:
            self._receipts.append(receipt)
            self._by_envelope_digest[receipt.envelope_digest] = receipt
            self._by_receipt_id[receipt.receipt_id] = receipt

    def get_by_envelope_digest(self, digest: str) -> Optional[CommitReceipt]:
        """Get receipt by envelope digest."""
        with self._lock:
            return self._by_envelope_digest.get(digest)

    def get_by_receipt_id(self, receipt_id: str) -> Optional[CommitReceipt]:
        """Get receipt by receipt ID."""
        with self._lock:
            return self._by_receipt_id.get(receipt_id)

    def get_by_offset(self, offset: int) -> Optional[CommitReceipt]:
        """Get receipt by offset."""
        with self._lock:
            for receipt in self._receipts:
                if receipt.commit_ref.offset == offset:
                    return receipt
            return None

    def get_all(self) -> List[CommitReceipt]:
        """Get all receipts."""
        with self._lock:
            return list(self._receipts)

    def count(self) -> int:
        """Get the number of receipts."""
        with self._lock:
            return len(self._receipts)

    def get_latest_offset(self) -> int:
        """Get the latest offset."""
        with self._lock:
            if not self._receipts:
                return 0
            return max(r.commit_ref.offset for r in self._receipts)


# Global receipt generator and store
_receipt_generator: Optional[ReceiptGenerator] = None
_receipt_store: Optional[ReceiptStore] = None


def get_receipt_generator() -> ReceiptGenerator:
    """Get the global receipt generator."""
    global _receipt_generator
    if _receipt_generator is None:
        _receipt_generator = ReceiptGenerator()
    return _receipt_generator


def get_receipt_store() -> ReceiptStore:
    """Get the global receipt store."""
    global _receipt_store
    if _receipt_store is None:
        _receipt_store = ReceiptStore()
    return _receipt_store
