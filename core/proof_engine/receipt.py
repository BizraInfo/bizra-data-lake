"""
Receipt System — Signed Success + Rejection Receipts

Fail-closed without panics; audit trail for denials.

Every execution produces a cryptographically signed receipt:
- Accepted: Query processed successfully
- Rejected: Query failed gates (with reason)
- AmberRestricted: Partial processing with restrictions

Receipts are the proof of execution.
"""

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple

from core.proof_engine.canonical import (
    CanonPolicy,
    CanonQuery,
    blake3_digest,
    canonical_bytes,
)
from core.proof_engine.snr import SNRTrace


class ReceiptStatus(Enum):
    """Receipt status codes."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    AMBER_RESTRICTED = "amber_restricted"
    PENDING = "pending"


class SovereignSigner(Protocol):
    """Protocol for signing receipts."""

    def sign(self, msg: bytes) -> bytes:
        """Sign a message and return signature."""
        ...

    def verify(self, msg: bytes, signature: bytes) -> bool:
        """Verify a signature."""
        ...

    def public_key_bytes(self) -> bytes:
        """Get public key bytes."""
        ...


class SimpleSigner:
    """
    Simple HMAC-based signer for development.

    Production should use Ed25519.
    """

    def __init__(self, secret: bytes):
        self.secret = secret

    def sign(self, msg: bytes) -> bytes:
        """Sign with HMAC-SHA256."""
        import hmac

        return hmac.new(self.secret, msg, hashlib.sha256).digest()

    def verify(self, msg: bytes, signature: bytes) -> bool:
        """Verify HMAC signature."""
        import hmac as hmac_module

        expected = self.sign(msg)
        return hmac_module.compare_digest(expected, signature)

    def public_key_bytes(self) -> bytes:
        """Return hash of secret as 'public key'."""
        return hashlib.sha256(self.secret).digest()


@dataclass
class Metrics:
    """Execution metrics for receipt."""

    p99_us: int = 0  # p99 latency in microseconds
    allocs: int = 0  # Allocation count
    cpu_cycles: int = 0  # CPU cycles (if available)
    memory_bytes: int = 0  # Peak memory usage
    duration_ms: float = 0.0  # Total duration

    def to_dict(self) -> Dict[str, Any]:
        return {
            "p99_us": self.p99_us,
            "allocs": self.allocs,
            "cpu_cycles": self.cpu_cycles,
            "memory_bytes": self.memory_bytes,
            "duration_ms": self.duration_ms,
        }


@dataclass
class Receipt:
    """
    Cryptographically signed execution receipt.

    The proof of what happened during query processing.
    """

    # Identity
    receipt_id: str

    # Status
    status: ReceiptStatus

    # Digests
    query_digest: bytes
    policy_digest: bytes
    payload_digest: bytes

    # Scores
    snr: float
    ihsan_score: float

    # Execution
    gate_passed: str  # Last gate passed (or failed)
    reason: Optional[str] = None  # Rejection reason

    # Metrics
    metrics: Metrics = field(default_factory=Metrics)

    # SNR trace (optional, for full audit)
    snr_trace: Optional[SNRTrace] = None

    # Signature
    signature: bytes = field(default_factory=bytes)
    signer_pubkey: bytes = field(default_factory=bytes)

    # Timestamp
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def body_bytes(self) -> bytes:
        """
        Get the receipt body for signing.

        Does not include signature itself.
        """
        data = {
            "receipt_id": self.receipt_id,
            "status": self.status.value,
            "query_digest": self.query_digest.hex(),
            "policy_digest": self.policy_digest.hex(),
            "payload_digest": self.payload_digest.hex(),
            "snr": self.snr,
            "ihsan_score": self.ihsan_score,
            "gate_passed": self.gate_passed,
            "reason": self.reason,
            "metrics": self.metrics.to_dict(),
            "timestamp": self.timestamp.isoformat(),
        }
        return canonical_bytes(data)

    def sign_with(self, signer: SovereignSigner) -> "Receipt":
        """Sign the receipt and return self."""
        body = self.body_bytes()
        self.signature = signer.sign(body)
        self.signer_pubkey = signer.public_key_bytes()
        return self

    def verify_signature(self, signer: SovereignSigner) -> bool:
        """Verify the receipt signature."""
        body = self.body_bytes()
        return signer.verify(body, self.signature)

    def digest(self) -> bytes:
        """Compute receipt digest (includes signature)."""
        data = self.body_bytes() + self.signature
        return blake3_digest(data)

    def hex_digest(self) -> str:
        """Compute hex-encoded digest."""
        return self.digest().hex()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "receipt_id": self.receipt_id,
            "status": self.status.value,
            "query_digest": self.query_digest.hex(),
            "policy_digest": self.policy_digest.hex(),
            "payload_digest": self.payload_digest.hex(),
            "snr": self.snr,
            "ihsan_score": self.ihsan_score,
            "gate_passed": self.gate_passed,
            "reason": self.reason,
            "metrics": self.metrics.to_dict(),
            "snr_trace": self.snr_trace.to_dict() if self.snr_trace else None,
            "signature": self.signature.hex(),
            "signer_pubkey": self.signer_pubkey.hex(),
            "timestamp": self.timestamp.isoformat(),
            "receipt_digest": self.hex_digest(),
        }


class ReceiptBuilder:
    """
    Builder for creating receipts.

    Ensures all required fields are set.
    """

    def __init__(self, signer: SovereignSigner):
        self.signer = signer
        self._counter = 0

    def _next_id(self) -> str:
        """Generate next receipt ID."""
        self._counter += 1
        return f"rcpt_{self._counter:012d}_{int(time.time() * 1000)}"

    def accepted(
        self,
        query: CanonQuery,
        policy: CanonPolicy,
        payload: bytes,
        snr: float,
        ihsan_score: float,
        gate_passed: str = "commit",
        metrics: Optional[Metrics] = None,
        snr_trace: Optional[SNRTrace] = None,
    ) -> Receipt:
        """Create an accepted receipt."""
        receipt = Receipt(
            receipt_id=self._next_id(),
            status=ReceiptStatus.ACCEPTED,
            query_digest=query.digest(),
            policy_digest=policy.digest(),
            payload_digest=blake3_digest(payload),
            snr=snr,
            ihsan_score=ihsan_score,
            gate_passed=gate_passed,
            metrics=metrics or Metrics(),
            snr_trace=snr_trace,
        )
        return receipt.sign_with(self.signer)

    def rejected(
        self,
        query: CanonQuery,
        policy: CanonPolicy,
        snr: float,
        ihsan_score: float,
        gate_failed: str,
        reason: str,
        metrics: Optional[Metrics] = None,
        snr_trace: Optional[SNRTrace] = None,
    ) -> Receipt:
        """Create a rejection receipt."""
        receipt = Receipt(
            receipt_id=self._next_id(),
            status=ReceiptStatus.REJECTED,
            query_digest=query.digest(),
            policy_digest=policy.digest(),
            payload_digest=blake3_digest(b""),  # No payload on rejection
            snr=snr,
            ihsan_score=ihsan_score,
            gate_passed=gate_failed,  # Last gate attempted
            reason=reason,
            metrics=metrics or Metrics(),
            snr_trace=snr_trace,
        )
        return receipt.sign_with(self.signer)

    def amber_restricted(
        self,
        query: CanonQuery,
        policy: CanonPolicy,
        payload: bytes,
        snr: float,
        ihsan_score: float,
        restriction_reason: str,
        gate_passed: str = "safety",
        metrics: Optional[Metrics] = None,
        snr_trace: Optional[SNRTrace] = None,
    ) -> Receipt:
        """Create an amber-restricted receipt."""
        receipt = Receipt(
            receipt_id=self._next_id(),
            status=ReceiptStatus.AMBER_RESTRICTED,
            query_digest=query.digest(),
            policy_digest=policy.digest(),
            payload_digest=blake3_digest(payload),
            snr=snr,
            ihsan_score=ihsan_score,
            gate_passed=gate_passed,
            reason=f"AMBER: {restriction_reason}",
            metrics=metrics or Metrics(),
            snr_trace=snr_trace,
        )
        return receipt.sign_with(self.signer)


class ReceiptVerifier:
    """
    Receipt verification service.

    Verifies signatures and checks consistency.
    """

    def __init__(self, signer: SovereignSigner):
        self.signer = signer
        self._verified: List[str] = []
        self._failed: List[str] = []

    def verify(self, receipt: Receipt) -> Tuple[bool, Optional[str]]:
        """
        Verify a receipt.

        Returns (valid, error_message).
        """
        # Verify signature
        if not receipt.verify_signature(self.signer):
            self._failed.append(receipt.receipt_id)
            return False, "Invalid signature"

        # Verify signer matches
        if receipt.signer_pubkey != self.signer.public_key_bytes():
            self._failed.append(receipt.receipt_id)
            return False, "Signer mismatch"

        # Verify SNR trace if present
        if receipt.snr_trace:
            from core.proof_engine.snr import SNREngine

            engine = SNREngine()
            if not engine.verify_trace(receipt.snr_trace):
                self._failed.append(receipt.receipt_id)
                return False, "SNR trace verification failed"

        self._verified.append(receipt.receipt_id)
        return True, None

    def get_stats(self) -> Dict[str, Any]:
        """Get verification statistics."""
        total = len(self._verified) + len(self._failed)
        return {
            "total_verified": len(self._verified),
            "total_failed": len(self._failed),
            "success_rate": len(self._verified) / total if total > 0 else 0.0,
        }
