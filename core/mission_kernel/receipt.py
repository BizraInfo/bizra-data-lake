"""Receipt v1 core types and deterministic verification.

This module intentionally keeps the v0.1 Mission Kernel small: Python may
propose, but this receipt contract is deterministic, identity-bound, and
replay-verifiable.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

RECEIPT_SCHEMA_VERSION = "receipt.v1"
IHSAN_THRESHOLD_BP = 9900


class Decision(str, Enum):
    COMMIT = "COMMIT"
    REJECT = "REJECT"


@dataclass(frozen=True)
class MissionState:
    mission_id: str
    frozen_ethical_anchors_hash: str
    constitution_hash: str
    ihsan_threshold_bp: int = IHSAN_THRESHOLD_BP

    def __post_init__(self) -> None:
        _require_hex32(self.frozen_ethical_anchors_hash, "frozen_ethical_anchors_hash")
        _require_hex32(self.constitution_hash, "constitution_hash")
        if not self.mission_id.strip():
            raise ValueError("mission_id must be non-empty")
        if not 0 <= self.ihsan_threshold_bp <= 10_000:
            raise ValueError("ihsan_threshold_bp must be 0..10000")


@dataclass(frozen=True)
class Proposal:
    mission_id: str
    proposer_id: str
    payload_hash: str
    claims: tuple[str, ...] = field(default_factory=tuple)
    nonce: str = ""
    counter: int = 0

    def __post_init__(self) -> None:
        if not self.mission_id.strip():
            raise ValueError("mission_id must be non-empty")
        if not self.proposer_id.strip():
            raise ValueError("proposer_id must be non-empty")
        _require_hex32(self.payload_hash, "payload_hash")
        if self.counter < 0:
            raise ValueError("counter must be non-negative")
        object.__setattr__(self, "claims", tuple(str(claim) for claim in self.claims))


@dataclass(frozen=True)
class FateVerdict:
    admissible: bool
    proof_class: str = "threshold-only"
    counterexample: str | None = None

    def __post_init__(self) -> None:
        if not self.proof_class.strip():
            raise ValueError("proof_class must be non-empty")


@dataclass(frozen=True)
class SatConsensus:
    passed: bool
    votes_for: int
    votes_against: int
    veto: bool = False

    def __post_init__(self) -> None:
        if self.votes_for < 0 or self.votes_against < 0:
            raise ValueError("vote counts must be non-negative")


@dataclass(frozen=True)
class ReceiptV1:
    receipt_id: str
    mission_id: str
    proposal_hash: str
    proposer_id: str
    signer_id: str
    signer_public_key: str
    constitution_hash: str
    ihsan_score_bp: int
    fate: FateVerdict
    sat: SatConsensus
    decision: Decision
    prev_hash: str | None
    current_hash: str
    signature: str
    schema_version: str = RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RECEIPT_SCHEMA_VERSION:
            raise ValueError("unsupported receipt schema version")
        for field_name in ("receipt_id", "mission_id", "proposer_id", "signer_id"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must be non-empty")
        _require_hex32(self.proposal_hash, "proposal_hash")
        _require_hex32(self.constitution_hash, "constitution_hash")
        _require_hex32(self.current_hash, "current_hash")
        if self.prev_hash is not None:
            _require_hex32(self.prev_hash, "prev_hash")
        if not 0 <= self.ihsan_score_bp <= 10_000:
            raise ValueError("ihsan_score_bp must be 0..10000")


def create_receipt(
    *,
    mission: MissionState,
    proposal: Proposal,
    fate: FateVerdict,
    sat: SatConsensus,
    ihsan_score_bp: int,
    signer_id: str,
    signing_key: Ed25519PrivateKey,
    prev_hash: str | None = None,
) -> ReceiptV1:
    """Create an identity-bound signed receipt.

    Commit is allowed only if FATE admits, SAT passes, no veto exists, and
    Ihsan meets mission threshold. Otherwise the receipt is a signed rejection.
    """
    if mission.mission_id != proposal.mission_id:
        raise ValueError("proposal mission_id does not match mission state")
    if not signer_id.strip():
        raise ValueError("signer_id must be non-empty")
    decision = _decision(mission, fate, sat, ihsan_score_bp)
    public_key = signing_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    signer_public_key = base64.b64encode(public_key).decode("ascii")
    receipt_id = _sha256_hex(
        _canonical_bytes(
            {
                "mission_id": mission.mission_id,
                "proposal_hash": _proposal_hash(proposal),
                "signer_id": signer_id,
                "prev_hash": prev_hash,
            }
        )
    )[:32]
    unsigned = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "receipt_id": receipt_id,
        "mission_id": mission.mission_id,
        "proposal_hash": _proposal_hash(proposal),
        "proposer_id": proposal.proposer_id,
        "signer_id": signer_id,
        "signer_public_key": signer_public_key,
        "constitution_hash": mission.constitution_hash,
        "ihsan_score_bp": ihsan_score_bp,
        "fate": _dataclass_dict(fate),
        "sat": _dataclass_dict(sat),
        "decision": decision.value,
        "prev_hash": prev_hash,
    }
    current_hash = _sha256_hex(_canonical_bytes(unsigned))
    signature = base64.b64encode(signing_key.sign(bytes.fromhex(current_hash))).decode("ascii")
    return ReceiptV1(
        receipt_id=receipt_id,
        mission_id=mission.mission_id,
        proposal_hash=unsigned["proposal_hash"],
        proposer_id=proposal.proposer_id,
        signer_id=signer_id,
        signer_public_key=signer_public_key,
        constitution_hash=mission.constitution_hash,
        ihsan_score_bp=ihsan_score_bp,
        fate=fate,
        sat=sat,
        decision=decision,
        prev_hash=prev_hash,
        current_hash=current_hash,
        signature=signature,
    )


def verify_receipt(receipt: ReceiptV1, *, expected_public_key: str | None = None) -> bool:
    if expected_public_key is not None and receipt.signer_public_key != expected_public_key:
        return False
    if receipt.current_hash != _receipt_unsigned_hash(receipt):
        return False
    try:
        public_key_bytes = base64.b64decode(receipt.signer_public_key)
        signature = base64.b64decode(receipt.signature)
        Ed25519PublicKey.from_public_bytes(public_key_bytes).verify(
            signature, bytes.fromhex(receipt.current_hash)
        )
        return True
    except (InvalidSignature, ValueError):
        return False


def receipt_to_dict(receipt: ReceiptV1) -> dict[str, Any]:
    return {
        "schema_version": receipt.schema_version,
        "receipt_id": receipt.receipt_id,
        "mission_id": receipt.mission_id,
        "proposal_hash": receipt.proposal_hash,
        "proposer_id": receipt.proposer_id,
        "signer_id": receipt.signer_id,
        "signer_public_key": receipt.signer_public_key,
        "constitution_hash": receipt.constitution_hash,
        "ihsan_score_bp": receipt.ihsan_score_bp,
        "fate": _dataclass_dict(receipt.fate),
        "sat": _dataclass_dict(receipt.sat),
        "decision": receipt.decision.value,
        "prev_hash": receipt.prev_hash,
        "current_hash": receipt.current_hash,
        "signature": receipt.signature,
    }


def receipt_from_dict(data: dict[str, Any]) -> ReceiptV1:
    return ReceiptV1(
        receipt_id=str(data["receipt_id"]),
        mission_id=str(data["mission_id"]),
        proposal_hash=str(data["proposal_hash"]),
        proposer_id=str(data["proposer_id"]),
        signer_id=str(data["signer_id"]),
        signer_public_key=str(data["signer_public_key"]),
        constitution_hash=str(data["constitution_hash"]),
        ihsan_score_bp=int(data["ihsan_score_bp"]),
        fate=FateVerdict(**data["fate"]),
        sat=SatConsensus(**data["sat"]),
        decision=Decision(str(data["decision"])),
        prev_hash=data.get("prev_hash"),
        current_hash=str(data["current_hash"]),
        signature=str(data["signature"]),
        schema_version=str(data.get("schema_version", RECEIPT_SCHEMA_VERSION)),
    )


def proposal_hash(proposal: Proposal) -> str:
    return _proposal_hash(proposal)


def _decision(
    mission: MissionState, fate: FateVerdict, sat: SatConsensus, ihsan_score_bp: int
) -> Decision:
    if (
        fate.admissible
        and sat.passed
        and not sat.veto
        and ihsan_score_bp >= mission.ihsan_threshold_bp
    ):
        return Decision.COMMIT
    return Decision.REJECT


def _proposal_hash(proposal: Proposal) -> str:
    return _sha256_hex(
        _canonical_bytes(
            {
                "mission_id": proposal.mission_id,
                "proposer_id": proposal.proposer_id,
                "payload_hash": proposal.payload_hash,
                "claims": list(proposal.claims),
                "nonce": proposal.nonce,
                "counter": proposal.counter,
            }
        )
    )


def _receipt_unsigned_hash(receipt: ReceiptV1) -> str:
    data = receipt_to_dict(receipt)
    data.pop("current_hash")
    data.pop("signature")
    return _sha256_hex(_canonical_bytes(data))


def _dataclass_dict(value: Any) -> dict[str, Any]:
    raw = value.__dict__.copy()
    return {k: (v.value if isinstance(v, Enum) else v) for k, v in raw.items()}


def _canonical_bytes(data: dict[str, Any]) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_hex32(value: str, field_name: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{field_name} must be 32-byte hex")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be hex") from exc
