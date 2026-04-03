"""
core/mission.py - Canonical Mission Contracts (Python Parity)

FROZEN: These are the Python equivalents of the four canonical contracts
defined authoritatively in src/mission.rs. This file MUST stay synchronized
with the Rust implementation.

Contracts:
    1. MissionEnvelope  - Operator-facing mission representation
    2. GateVerdict      - Constitutional gate output (PERMIT|REJECT|REVIEW|SCORE_ONLY)
    3. ReceiptArtifact  - Signed proof artifact with BLAKE3 chain
    4. ManifestArtifact - Daily proof-of-life heartbeat

Authority chain: Layer 1 defines -> Layer 2 interprets -> Layer 3 enforces
                 -> Layer 4 experiments -> Layer 5 reveals

Gate order (frozen):
    Ingress -> State -> Proposal -> Constitution -> Proof -> Receipt -> Refinement -> Reflex
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

# ── Frozen Gate Order ─────────────────────────────────────────────────────────

GATE_ORDER = (
    "Ingress",
    "State",
    "Proposal",
    "Constitution",
    "Proof",
    "Receipt",
    "Refinement",
    "Reflex",
)

# ── Enums ─────────────────────────────────────────────────────────────────────


class MissionState(str, Enum):
    DRAFT = "draft"
    SUBMITTED = "submitted"
    EVALUATING = "evaluating"
    PERMITTED = "permitted"
    EXECUTING = "executing"
    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"
    REVOKED = "revoked"


# Valid state transitions: (from, to)
MISSION_TRANSITIONS: frozenset[tuple[MissionState, MissionState]] = frozenset(
    {
        (MissionState.DRAFT, MissionState.SUBMITTED),
        (MissionState.SUBMITTED, MissionState.EVALUATING),
        (MissionState.EVALUATING, MissionState.PERMITTED),
        (MissionState.EVALUATING, MissionState.REJECTED),
        (MissionState.PERMITTED, MissionState.EXECUTING),
        (MissionState.EXECUTING, MissionState.COMPLETED),
        (MissionState.EXECUTING, MissionState.FAILED),
        (MissionState.PERMITTED, MissionState.REVOKED),
        (MissionState.EXECUTING, MissionState.REVOKED),
    }
)


class VerdictKind(str, Enum):
    PERMIT = "PERMIT"
    REJECT = "REJECT"
    REVIEW = "REVIEW"
    SCORE_ONLY = "SCORE_ONLY"


class RiskClass(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MissionClass(str, Enum):
    WORK = "work"
    QUERY = "query"
    MAINTENANCE = "maintenance"
    CONSTITUTIONAL = "constitutional"


class EvidenceSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class ReceiptState(str, Enum):
    ISSUED = "issued"
    CHAINED = "chained"
    VERIFIED = "verified"
    EXPIRED = "expired"
    REVOKED = "revoked"


class HeartbeatStatus(str, Enum):
    ALIVE = "alive"
    DEGRADED = "degraded"
    DEAD = "dead"


class LayerStatus(str, Enum):
    ACTIVE = "active"
    DEGRADED = "degraded"
    OFFLINE = "offline"


class BridgeStatus(str, Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class ChainStatus(str, Enum):
    INTACT = "intact"
    BROKEN = "broken"
    EMPTY = "empty"


# ── Supporting Dataclasses ────────────────────────────────────────────────────


@dataclass(frozen=True)
class EvidenceItem:
    code: str
    description: str
    severity: EvidenceSeverity


@dataclass(frozen=True)
class ReceiptAction:
    tool: str
    input: dict[str, Any]
    output: dict[str, Any]
    timestamp: datetime


@dataclass(frozen=True)
class ReceiptVerification:
    signature_valid: bool
    chain_intact: bool
    payload_intact: bool
    verified_at: datetime
    verified_by: str


@dataclass(frozen=True)
class ManifestVerification:
    signature_valid: bool
    chain_head_matches: bool
    receipt_count_matches: bool
    verified_at: datetime


@dataclass(frozen=True)
class SystemHealthSnapshot:
    uptime_seconds: int
    constitutional_layer: LayerStatus
    kernel_bridge: BridgeStatus
    receipt_chain: ChainStatus


# ── Contract 1: GateVerdict ───────────────────────────────────────────────────


@dataclass
class GateVerdict:
    """Canonical GateVerdict — produced by the authoritative kernel (Layer 1/2)."""

    verdict_id: str
    layer: int  # 1, 2, or 3
    kind: VerdictKind
    reason_code: str
    reason: str
    advisory_score: int  # 0-100
    evidence: list[EvidenceItem] = field(default_factory=list)
    issued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    authority_signature: Optional[str] = None

    @classmethod
    def permit(cls, reason_code: str, reason: str, score: int) -> GateVerdict:
        return cls(
            verdict_id=str(uuid.uuid4()),
            layer=1,
            kind=VerdictKind.PERMIT,
            reason_code=reason_code,
            reason=reason,
            advisory_score=score,
        )

    @classmethod
    def reject(
        cls,
        reason_code: str,
        reason: str,
        evidence: list[EvidenceItem] | None = None,
    ) -> GateVerdict:
        return cls(
            verdict_id=str(uuid.uuid4()),
            layer=1,
            kind=VerdictKind.REJECT,
            reason_code=reason_code,
            reason=reason,
            advisory_score=0,
            evidence=evidence or [],
        )

    @property
    def is_permit(self) -> bool:
        return self.kind == VerdictKind.PERMIT

    @property
    def is_blocking(self) -> bool:
        return self.kind in (VerdictKind.PERMIT, VerdictKind.REJECT)

    def integrity_hash(self) -> str:
        payload = f"{self.verdict_id}{self.issued_at.isoformat()}{self.reason_code}{self.reason}{self.advisory_score}"
        return hashlib.sha256(payload.encode()).hexdigest()


# ── Contract 2: MissionEnvelope ───────────────────────────────────────────────


class MissionTransitionError(Exception):
    def __init__(self, from_state: MissionState, to_state: MissionState):
        self.from_state = from_state
        self.to_state = to_state
        super().__init__(f"Invalid mission transition: {from_state.value} -> {to_state.value}")


@dataclass
class MissionEnvelope:
    """Canonical MissionEnvelope — the operator-facing mission representation."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: MissionState = MissionState.DRAFT
    description: str = ""
    mission_class: MissionClass = MissionClass.WORK
    scope: list[str] = field(default_factory=list)
    risk_class: RiskClass = RiskClass.LOW
    verdict: Optional[GateVerdict] = None
    operator_key: Optional[str] = None
    assigned_agent_id: Optional[str] = None
    max_tokens: int = 100_000
    max_steps: int = 50
    budget_used: int = 0
    submitted_at: Optional[datetime] = None
    evaluated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def new(
        cls,
        description: str,
        mission_class: MissionClass = MissionClass.WORK,
        risk_class: RiskClass = RiskClass.LOW,
    ) -> MissionEnvelope:
        return cls(description=description, mission_class=mission_class, risk_class=risk_class)

    def can_transition_to(self, target: MissionState) -> bool:
        return (self.state, target) in MISSION_TRANSITIONS

    def transition_to(self, target: MissionState) -> None:
        if not self.can_transition_to(target):
            raise MissionTransitionError(self.state, target)

        self.state = target
        now = datetime.now(timezone.utc)
        self.updated_at = now

        if target == MissionState.SUBMITTED:
            self.submitted_at = now
        elif target in (MissionState.PERMITTED, MissionState.REJECTED):
            self.evaluated_at = now
        elif target in (MissionState.COMPLETED, MissionState.FAILED):
            self.completed_at = now

    def apply_verdict(self, verdict: GateVerdict) -> None:
        target = MissionState.PERMITTED if verdict.is_permit else MissionState.REJECTED
        self.verdict = verdict
        self.transition_to(target)


# ── Contract 3: ReceiptArtifact ───────────────────────────────────────────────


def _blake3_hex(data: str) -> str:
    """Compute BLAKE3 hash. Falls back to SHA-256 if blake3 is not installed."""
    try:
        import blake3 as _blake3

        return _blake3.blake3(data.encode()).hexdigest()
    except ImportError:
        return hashlib.sha256(data.encode()).hexdigest()


@dataclass
class ReceiptArtifact:
    """Canonical ReceiptArtifact — the signed proof artifact."""

    receipt_id: str
    mission_id: str
    prev_receipt_hash: str
    receipt_state: ReceiptState
    verdict: GateVerdict
    actions: list[ReceiptAction]
    evidence_hash: str
    payload_hash: str
    authority_signature: str = ""
    issued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    verification: Optional[ReceiptVerification] = None

    @classmethod
    def new(
        cls,
        mission_id: str,
        prev_receipt_hash: str,
        verdict: GateVerdict,
        actions: list[ReceiptAction] | None = None,
    ) -> ReceiptArtifact:
        actions = actions or []
        receipt_id = str(uuid.uuid4())

        # Evidence hash
        import json

        evidence_payload = json.dumps(
            [
                {
                    "tool": a.tool,
                    "input": a.input,
                    "output": a.output,
                    "timestamp": a.timestamp.isoformat(),
                }
                for a in actions
            ]
        )
        evidence_hash = _blake3_hex(evidence_payload)

        # Payload hash
        payload = f"{receipt_id}{mission_id}{prev_receipt_hash}{verdict.verdict_id}"
        payload_hash = _blake3_hex(payload)

        return cls(
            receipt_id=receipt_id,
            mission_id=mission_id,
            prev_receipt_hash=prev_receipt_hash,
            receipt_state=ReceiptState.ISSUED,
            verdict=verdict,
            actions=actions,
            evidence_hash=evidence_hash,
            payload_hash=payload_hash,
        )

    def verify_chain(self, expected_prev_hash: str) -> bool:
        return self.prev_receipt_hash == expected_prev_hash

    def verify_payload(self) -> bool:
        payload = f"{self.receipt_id}{self.mission_id}{self.prev_receipt_hash}{self.verdict.verdict_id}"
        expected = _blake3_hex(payload)
        return expected == self.payload_hash


# ── Contract 4: ManifestArtifact ──────────────────────────────────────────────


@dataclass
class ManifestArtifact:
    """Canonical ManifestArtifact — the daily proof-of-life artifact."""

    id: str
    period_start: datetime
    period_end: datetime
    heartbeat_status: HeartbeatStatus
    receipt_chain_head: str
    receipt_count: int
    permit_count: int
    reject_count: int
    review_count: int
    system_health: SystemHealthSnapshot
    deployment_id: str
    public_proof_hash: str
    authority_signature: str = ""
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    verification: Optional[ManifestVerification] = None

    @classmethod
    def generate(
        cls,
        period_start: datetime,
        period_end: datetime,
        receipt_chain_head: str,
        receipt_count: int,
        permit_count: int,
        reject_count: int,
        review_count: int,
        system_health: SystemHealthSnapshot,
        deployment_id: str,
    ) -> ManifestArtifact:
        manifest_id = str(uuid.uuid4())
        payload = (
            f"{manifest_id}"
            f"{period_start.isoformat()}"
            f"{period_end.isoformat()}"
            f"{receipt_chain_head}"
            f"{receipt_count}"
            f"{permit_count}"
            f"{reject_count}"
            f"{review_count}"
        )
        public_proof_hash = _blake3_hex(payload)

        return cls(
            id=manifest_id,
            period_start=period_start,
            period_end=period_end,
            heartbeat_status=HeartbeatStatus.ALIVE,
            receipt_chain_head=receipt_chain_head,
            receipt_count=receipt_count,
            permit_count=permit_count,
            reject_count=reject_count,
            review_count=review_count,
            system_health=system_health,
            deployment_id=deployment_id,
            public_proof_hash=public_proof_hash,
        )

    @property
    def is_healthy(self) -> bool:
        return self.heartbeat_status == HeartbeatStatus.ALIVE

    def verify_proof(self) -> bool:
        payload = (
            f"{self.id}"
            f"{self.period_start.isoformat()}"
            f"{self.period_end.isoformat()}"
            f"{self.receipt_chain_head}"
            f"{self.receipt_count}"
            f"{self.permit_count}"
            f"{self.reject_count}"
            f"{self.review_count}"
        )
        expected = _blake3_hex(payload)
        return expected == self.public_proof_hash


# ── Genesis Seal ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GenesisSeal:
    """Trust anchor for the entire receipt chain. Set once at deployment."""

    genesis_hash: str
    authority_public_key: str
    deployment_id: str
    created_at: datetime

    @classmethod
    def new(cls, authority_public_key: str, deployment_id: str) -> GenesisSeal:
        created_at = datetime.now(timezone.utc)
        genesis_payload = f"bizra-genesis:{deployment_id}:{created_at.isoformat()}"
        genesis_hash = _blake3_hex(genesis_payload)
        return cls(
            genesis_hash=genesis_hash,
            authority_public_key=authority_public_key,
            deployment_id=deployment_id,
            created_at=created_at,
        )
