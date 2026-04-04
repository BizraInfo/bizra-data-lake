"""
BIZRA PCI Protocol — RejectCode Registry
=========================================
Stable numeric IDs for cross-language compatibility and audit logging.

Status: FROZEN — Changes require version bump + test vector update
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Optional

from .types import AuditTrail


class RejectCode(IntEnum):
    """
    Stable numeric rejection codes.

    WARNING: These codes are part of the wire protocol.
    Never change existing code values. Only append new codes.
    """

    # Success
    SUCCESS = 0

    # CHEAP tier rejections (1-5)
    REJECT_SCHEMA = 1
    REJECT_SIGNATURE = 2
    REJECT_NONCE_REPLAY = 3
    REJECT_TIMESTAMP_STALE = 4
    REJECT_TIMESTAMP_FUTURE = 5

    # MEDIUM tier rejections (6-10)
    REJECT_IHSAN_BELOW_MIN = 6
    REJECT_SNR_BELOW_MIN = 7
    REJECT_BUDGET_EXCEEDED = 8
    REJECT_POLICY_MISMATCH = 9
    REJECT_STATE_MISMATCH = 10

    # ROLE/QUORUM rejections (11-12)
    REJECT_ROLE_VIOLATION = 11
    REJECT_QUORUM_FAILED = 12

    # EXPENSIVE tier rejections (13-14)
    REJECT_FATE_VIOLATION = 13
    REJECT_INVARIANT_FAILED = 14

    # Rate limiting (15)
    REJECT_RATE_LIMITED = 15

    # Internal error (99)
    REJECT_INTERNAL_ERROR = 99

    @property
    def is_success(self) -> bool:
        return self == RejectCode.SUCCESS

    @property
    def is_rejection(self) -> bool:
        return self != RejectCode.SUCCESS

    @property
    def description(self) -> str:
        """Human-readable description of the reject code."""
        descriptions = {
            RejectCode.SUCCESS: "Operation completed successfully",
            RejectCode.REJECT_SCHEMA: "Envelope failed JSON schema validation",
            RejectCode.REJECT_SIGNATURE: "Cryptographic signature invalid",
            RejectCode.REJECT_NONCE_REPLAY: "Nonce already seen within TTL window",
            RejectCode.REJECT_TIMESTAMP_STALE: "Timestamp outside acceptable skew (past)",
            RejectCode.REJECT_TIMESTAMP_FUTURE: "Timestamp too far in future",
            RejectCode.REJECT_IHSAN_BELOW_MIN: "Ihsān score below 0.95 threshold",
            RejectCode.REJECT_SNR_BELOW_MIN: "SNR score below tier threshold",
            RejectCode.REJECT_BUDGET_EXCEEDED: "Verification latency exceeded tier budget",
            RejectCode.REJECT_POLICY_MISMATCH: "policy_hash doesn't match current constitution",
            RejectCode.REJECT_STATE_MISMATCH: "state_hash doesn't match expected state",
            RejectCode.REJECT_ROLE_VIOLATION: "Agent attempted unauthorized action",
            RejectCode.REJECT_QUORUM_FAILED: "Insufficient verifier signatures",
            RejectCode.REJECT_FATE_VIOLATION: "FATE invariant check failed",
            RejectCode.REJECT_INVARIANT_FAILED: "Formal invariant verification failed",
            RejectCode.REJECT_RATE_LIMITED: "Too many requests from sender",
            RejectCode.REJECT_INTERNAL_ERROR: "Unexpected internal error (fail-closed)",
        }
        return descriptions.get(self, "Unknown rejection code")


@dataclass
class RejectionResponse:
    """
    Structured rejection response.

    Every rejection MUST produce this response for audit logging.
    """

    rejected: bool
    code: RejectCode
    message: str
    envelope_digest: str
    timestamp: str
    audit_trail: Optional[AuditTrail] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "rejected": self.rejected,
            "code": int(self.code),
            "name": self.code.name,
            "message": self.message,
            "envelope_digest": self.envelope_digest,
            "timestamp": self.timestamp,
        }
        if self.audit_trail:
            result["audit_trail"] = self.audit_trail.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RejectionResponse":
        audit_trail = None
        if "audit_trail" in data:
            audit_trail = AuditTrail.from_dict(data["audit_trail"])

        return cls(
            rejected=data["rejected"],
            code=RejectCode(data["code"]),
            message=data["message"],
            envelope_digest=data["envelope_digest"],
            timestamp=data["timestamp"],
            audit_trail=audit_trail,
        )

    @classmethod
    def success(cls, envelope_digest: str, timestamp: str) -> "RejectionResponse":
        """Create a success response."""
        return cls(
            rejected=False,
            code=RejectCode.SUCCESS,
            message="Operation completed successfully",
            envelope_digest=envelope_digest,
            timestamp=timestamp,
            audit_trail=None,
        )

    @classmethod
    def rejection(
        cls,
        code: RejectCode,
        message: str,
        envelope_digest: str,
        timestamp: str,
        audit_trail: Optional[AuditTrail] = None,
    ) -> "RejectionResponse":
        """Create a rejection response."""
        return cls(
            rejected=True,
            code=code,
            message=message,
            envelope_digest=envelope_digest,
            timestamp=timestamp,
            audit_trail=audit_trail,
        )


# =============================================================================
# REJECTION HELPERS
# =============================================================================


def reject_schema(
    envelope_digest: str, timestamp: str, details: str
) -> RejectionResponse:
    """Create a schema validation rejection."""
    from .types import Gate, GateTier

    return RejectionResponse.rejection(
        code=RejectCode.REJECT_SCHEMA,
        message=f"Schema validation failed: {details}",
        envelope_digest=envelope_digest,
        timestamp=timestamp,
        audit_trail=AuditTrail(
            gate=Gate.SCHEMA,
            tier=GateTier.CHEAP,
            latency_ms=0.0,
            details={"error": details},
        ),
    )


def reject_signature(envelope_digest: str, timestamp: str) -> RejectionResponse:
    """Create a signature verification rejection."""
    from .types import Gate, GateTier

    return RejectionResponse.rejection(
        code=RejectCode.REJECT_SIGNATURE,
        message="Cryptographic signature verification failed",
        envelope_digest=envelope_digest,
        timestamp=timestamp,
        audit_trail=AuditTrail(
            gate=Gate.SIGNATURE,
            tier=GateTier.CHEAP,
            latency_ms=0.0,
            details={"error": "Invalid Ed25519 signature"},
        ),
    )


def reject_ihsan(
    envelope_digest: str,
    timestamp: str,
    score: float,
    threshold: float = 0.95,
) -> RejectionResponse:
    """Create an Ihsān threshold rejection."""
    from .types import Gate, GateTier

    return RejectionResponse.rejection(
        code=RejectCode.REJECT_IHSAN_BELOW_MIN,
        message=f"Ihsān score {score:.2f} < required {threshold:.2f}",
        envelope_digest=envelope_digest,
        timestamp=timestamp,
        audit_trail=AuditTrail(
            gate=Gate.IHSAN,
            tier=GateTier.MEDIUM,
            latency_ms=0.0,
            details={"score": score, "threshold": threshold},
        ),
    )


def reject_snr(
    envelope_digest: str,
    timestamp: str,
    score: float,
    threshold: float,
) -> RejectionResponse:
    """Create an SNR threshold rejection."""
    from .types import Gate, GateTier

    return RejectionResponse.rejection(
        code=RejectCode.REJECT_SNR_BELOW_MIN,
        message=f"SNR score {score:.2f} < tier threshold {threshold:.2f}",
        envelope_digest=envelope_digest,
        timestamp=timestamp,
        audit_trail=AuditTrail(
            gate=Gate.SNR,
            tier=GateTier.MEDIUM,
            latency_ms=0.0,
            details={"score": score, "threshold": threshold},
        ),
    )


def reject_replay(
    envelope_digest: str, timestamp: str, nonce: str
) -> RejectionResponse:
    """Create a nonce replay rejection."""
    from .types import Gate, GateTier

    return RejectionResponse.rejection(
        code=RejectCode.REJECT_NONCE_REPLAY,
        message="Nonce already seen within TTL window (replay attack detected)",
        envelope_digest=envelope_digest,
        timestamp=timestamp,
        audit_trail=AuditTrail(
            gate=Gate.REPLAY,
            tier=GateTier.CHEAP,
            latency_ms=0.0,
            details={"nonce": nonce[:16] + "..."},  # Truncate for log safety
        ),
    )


def reject_timestamp_stale(
    envelope_digest: str,
    timestamp: str,
    envelope_ts: str,
    skew_seconds: float,
) -> RejectionResponse:
    """Create a stale timestamp rejection."""
    from .types import Gate, GateTier

    return RejectionResponse.rejection(
        code=RejectCode.REJECT_TIMESTAMP_STALE,
        message=f"Timestamp {envelope_ts} is {abs(skew_seconds):.1f}s in the past (max 120s)",
        envelope_digest=envelope_digest,
        timestamp=timestamp,
        audit_trail=AuditTrail(
            gate=Gate.TIMESTAMP,
            tier=GateTier.CHEAP,
            latency_ms=0.0,
            details={"envelope_timestamp": envelope_ts, "skew_seconds": skew_seconds},
        ),
    )


def reject_timestamp_future(
    envelope_digest: str,
    timestamp: str,
    envelope_ts: str,
    skew_seconds: float,
) -> RejectionResponse:
    """Create a future timestamp rejection."""
    from .types import Gate, GateTier

    return RejectionResponse.rejection(
        code=RejectCode.REJECT_TIMESTAMP_FUTURE,
        message=f"Timestamp {envelope_ts} is {skew_seconds:.1f}s in the future (max 120s)",
        envelope_digest=envelope_digest,
        timestamp=timestamp,
        audit_trail=AuditTrail(
            gate=Gate.TIMESTAMP,
            tier=GateTier.CHEAP,
            latency_ms=0.0,
            details={"envelope_timestamp": envelope_ts, "skew_seconds": skew_seconds},
        ),
    )


def reject_role_violation(
    envelope_digest: str,
    timestamp: str,
    agent_type: str,
    action: str,
) -> RejectionResponse:
    """Create a role violation rejection."""
    from .types import Gate, GateTier

    return RejectionResponse.rejection(
        code=RejectCode.REJECT_ROLE_VIOLATION,
        message=f"{agent_type} agent cannot perform action: {action}",
        envelope_digest=envelope_digest,
        timestamp=timestamp,
        audit_trail=AuditTrail(
            gate=Gate.ROLE,
            tier=GateTier.CHEAP,
            latency_ms=0.0,
            details={"agent_type": agent_type, "action": action},
        ),
    )


def reject_fate_violation(
    envelope_digest: str,
    timestamp: str,
    invariant: str,
    details: Dict[str, Any],
) -> RejectionResponse:
    """Create a FATE invariant violation rejection."""
    from .types import Gate, GateTier

    return RejectionResponse.rejection(
        code=RejectCode.REJECT_FATE_VIOLATION,
        message=f"FATE invariant failed: {invariant}",
        envelope_digest=envelope_digest,
        timestamp=timestamp,
        audit_trail=AuditTrail(
            gate=Gate.FATE,
            tier=GateTier.EXPENSIVE,
            latency_ms=0.0,
            details={"invariant": invariant, **details},
        ),
    )


def reject_internal_error(
    envelope_digest: str,
    timestamp: str,
    error: str,
) -> RejectionResponse:
    """Create an internal error rejection (fail-closed)."""
    from .types import Gate, GateTier

    return RejectionResponse.rejection(
        code=RejectCode.REJECT_INTERNAL_ERROR,
        message=f"Internal error (fail-closed): {error}",
        envelope_digest=envelope_digest,
        timestamp=timestamp,
        audit_trail=AuditTrail(
            gate=Gate.SCHEMA,  # Default gate
            tier=GateTier.CHEAP,
            latency_ms=0.0,
            details={"error": error},
        ),
    )
