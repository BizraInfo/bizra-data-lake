"""
BIZRA PCI Protocol — Type Definitions
=====================================
Proof-Carrying Inference type system for dual-agent architecture.

Status: FROZEN — Changes require version bump + test vector update
Alignment: BIZRA_SOT.md Section 3.1 (Ihsān IM ≥ 0.95)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from core.constants import (
    IHSAN_THRESHOLD as CONSTITUTION_IHSAN_THRESHOLD,
    SNR_THRESHOLD_BASE as CONSTITUTION_SNR_THRESHOLD,
)

# =============================================================================
# CONSTANTS
# =============================================================================

PCI_VERSION = "1.0.0"
DOMAIN_PREFIX = "bizra-pci-v1:"
NONCE_BYTES = 32
TIMESTAMP_SKEW_SECONDS = 120
IHSAN_THRESHOLD = CONSTITUTION_IHSAN_THRESHOLD
SNR_THRESHOLD_DEFAULT = CONSTITUTION_SNR_THRESHOLD

# Latency budgets (milliseconds)
LATENCY_BUDGET_CHEAP = 10
LATENCY_BUDGET_MEDIUM = 150
LATENCY_BUDGET_EXPENSIVE = 2000


# =============================================================================
# ENUMS
# =============================================================================

class AgentType(str, Enum):
    """Agent type in the dual-agent architecture."""
    PAT = "PAT"  # Prover/Builder
    SAT = "SAT"  # Verifier/Governor


class Urgency(str, Enum):
    """Request urgency level affecting latency budgets."""
    REAL_TIME = "REAL_TIME"
    NEAR_REAL_TIME = "NEAR_REAL_TIME"
    BATCH = "BATCH"
    DEFERRED = "DEFERRED"


class VerificationTier(str, Enum):
    """Verification confidence tier."""
    STATISTICAL = "STATISTICAL"
    INCREMENTAL = "INCREMENTAL"
    OPTIMISTIC = "OPTIMISTIC"
    FULL_ZK = "FULL_ZK"
    FORMAL = "FORMAL"


class GateTier(str, Enum):
    """Gate execution tier with latency budget."""
    CHEAP = "CHEAP"      # <10ms
    MEDIUM = "MEDIUM"    # <150ms
    EXPENSIVE = "EXPENSIVE"  # <2000ms


class Gate(str, Enum):
    """Verification gates in execution order."""
    # CHEAP tier
    SCHEMA = "SCHEMA"
    SIGNATURE = "SIGNATURE"
    TIMESTAMP = "TIMESTAMP"
    REPLAY = "REPLAY"
    ROLE = "ROLE"
    # MEDIUM tier
    SNR = "SNR"
    IHSAN = "IHSAN"
    POLICY = "POLICY"
    # EXPENSIVE tier
    FATE = "FATE"
    FORMAL = "FORMAL"

    @property
    def tier(self) -> GateTier:
        """Get the tier for this gate."""
        if self in (Gate.SCHEMA, Gate.SIGNATURE, Gate.TIMESTAMP, Gate.REPLAY, Gate.ROLE):
            return GateTier.CHEAP
        elif self in (Gate.SNR, Gate.IHSAN, Gate.POLICY):
            return GateTier.MEDIUM
        else:
            return GateTier.EXPENSIVE


class CommitRefType(str, Enum):
    """Type of commit reference."""
    EVENTLOG = "eventlog"
    BLOCKGRAPH = "blockgraph"


class SignatureAlgorithm(str, Enum):
    """Supported signature algorithms."""
    ED25519 = "ed25519"
    DILITHIUM5 = "dilithium5"  # Future: post-quantum


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Sender:
    """Envelope sender information."""
    agent_type: AgentType
    agent_id: str
    public_key: str  # Hex-encoded Ed25519 public key (32 bytes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_type": self.agent_type.value,
            "agent_id": self.agent_id,
            "public_key": self.public_key,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Sender":
        return cls(
            agent_type=AgentType(data["agent_type"]),
            agent_id=data["agent_id"],
            public_key=data["public_key"],
        )


@dataclass
class Payload:
    """Envelope payload containing the action and data."""
    action: str
    data: Dict[str, Any]
    policy_hash: str  # BLAKE3 of constitution
    state_hash: str   # BLAKE3 of current state

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "data": self.data,
            "policy_hash": self.policy_hash,
            "state_hash": self.state_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Payload":
        return cls(
            action=data["action"],
            data=data["data"],
            policy_hash=data["policy_hash"],
            state_hash=data["state_hash"],
        )


@dataclass
class Metadata:
    """Envelope metadata for verification gates."""
    ihsan_score: float
    snr_score: float
    urgency: Urgency = Urgency.NEAR_REAL_TIME

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ihsan_score": self.ihsan_score,
            "snr_score": self.snr_score,
            "urgency": self.urgency.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Metadata":
        return cls(
            ihsan_score=float(data["ihsan_score"]),
            snr_score=float(data["snr_score"]),
            urgency=Urgency(data.get("urgency", "NEAR_REAL_TIME")),
        )


@dataclass
class Signature:
    """Cryptographic signature of the envelope."""
    algorithm: SignatureAlgorithm
    value: str  # Hex-encoded signature
    signed_fields: List[str] = field(default_factory=lambda: [
        "version", "envelope_id", "timestamp", "nonce", "sender", "payload", "metadata"
    ])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm.value,
            "value": self.value,
            "signed_fields": self.signed_fields,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Signature":
        return cls(
            algorithm=SignatureAlgorithm(data["algorithm"]),
            value=data["value"],
            signed_fields=data.get("signed_fields", [
                "version", "envelope_id", "timestamp", "nonce", "sender", "payload", "metadata"
            ]),
        )


@dataclass
class CommitRef:
    """Reference to the committed location."""
    type: CommitRefType
    offset: int
    block_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "type": self.type.value,
            "offset": self.offset,
        }
        if self.block_hash:
            result["block_hash"] = self.block_hash
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CommitRef":
        return cls(
            type=CommitRefType(data["type"]),
            offset=data["offset"],
            block_hash=data.get("block_hash"),
        )


@dataclass
class Verification:
    """Verification result details."""
    tier: VerificationTier
    latency_ms: float
    gates_passed: List[Gate]
    ihsan_score: float
    snr_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier.value,
            "latency_ms": self.latency_ms,
            "gates_passed": [g.value for g in self.gates_passed],
            "ihsan_score": self.ihsan_score,
            "snr_score": self.snr_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Verification":
        return cls(
            tier=VerificationTier(data["tier"]),
            latency_ms=float(data["latency_ms"]),
            gates_passed=[Gate(g) for g in data["gates_passed"]],
            ihsan_score=float(data["ihsan_score"]),
            snr_score=float(data["snr_score"]),
        )


@dataclass
class VerifierSignature:
    """Signature from a SAT verifier."""
    sat_id: str
    public_key: str
    signature: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sat_id": self.sat_id,
            "public_key": self.public_key,
            "signature": self.signature,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerifierSignature":
        return cls(
            sat_id=data["sat_id"],
            public_key=data["public_key"],
            signature=data["signature"],
            timestamp=data["timestamp"],
        )


@dataclass
class Quorum:
    """Quorum requirements and achievement."""
    required: int
    achieved: int

    def is_met(self) -> bool:
        return self.achieved >= self.required

    def to_dict(self) -> Dict[str, int]:
        return {
            "required": self.required,
            "achieved": self.achieved,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> "Quorum":
        return cls(
            required=data["required"],
            achieved=data["achieved"],
        )


@dataclass
class AuditTrail:
    """Audit trail for rejection."""
    gate: Gate
    tier: GateTier
    latency_ms: float
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate": self.gate.value,
            "tier": self.tier.value,
            "latency_ms": self.latency_ms,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditTrail":
        return cls(
            gate=Gate(data["gate"]),
            tier=GateTier(data["tier"]),
            latency_ms=float(data["latency_ms"]),
            details=data["details"],
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def utc_now_iso() -> str:
    """Get current UTC time in ISO8601 format."""
    return datetime.now(timezone.utc).isoformat()


def generate_envelope_id() -> str:
    """Generate a unique envelope ID."""
    return str(uuid4())


def generate_receipt_id() -> str:
    """Generate a unique receipt ID."""
    return str(uuid4())
