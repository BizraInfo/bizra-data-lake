"""
Canonical Receipt Adapter — Python mirror of Rust CanonicalReceipt v1.

Maps between the Python evidence ledger and the Rust canonical schema.
Both sides must produce byte-identical hashes for cross-language verification.

This is CANONICALIZATION_SPRINT_01 deliverable #6: edge hardening.

Standing on Giants:
  - Golden Vector protocol: Rust/Python digest parity
  - Nakamoto (2008): hash-chained immutable records
  - Lamport (1978): happens-before ordering
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class ReceiptState(IntEnum):
    HYPOTHESIS = 0
    VERIFIED = 1
    EXECUTABLE = 2
    COMMITTED = 3
    REPLAYABLE = 4
    MARKETABLE = 5


class ExecutionRoute(IntEnum):
    REFLEX = 0
    DELIBERATE = 1
    DEGRADED = 2
    REJECTED = 3


class VerdictStatus(IntEnum):
    ADMITTED = 0
    REJECTED = 1
    DEFERRED = 2


DOMAIN_CANONICAL_RECEIPT = b"bizra-canonical-receipt-v1"
IHSAN_FEDERATION_THRESHOLD = 0.95

GENESIS_SEED = bytes(
    [
        0xB1,
        0x2A,
        0xF3,
        0x7E,
        0xD4,
        0x91,
        0xC8,
        0x56,
        0x2F,
        0x0E,
        0x8B,
        0xD7,
        0x43,
        0x9A,
        0x5C,
        0x11,
        0xE7,
        0x2D,
        0x60,
        0xF8,
        0x1B,
        0x37,
        0xA4,
        0xCE,
        0x95,
        0x4F,
        0x0D,
        0x82,
        0x76,
        0x3C,
        0xB9,
        0x0A,
    ]
)


@dataclass
class CanonicalReceipt:
    """Python mirror of Rust CanonicalReceipt v1."""

    receipt_id: bytes  # 32 bytes
    mission_id: str
    genesis_hash: bytes  # 32 bytes
    policy_version: str
    verdict: VerdictStatus
    primary_reject: Optional[str]
    ihsan_score: float
    snr_score: float
    route: ExecutionRoute
    received_at: int  # Unix ms
    sealed_at: int  # Unix ms
    input_hash: bytes  # 32 bytes
    output_hash: bytes  # 32 bytes
    previous_receipt: bytes  # 32 bytes
    state: ReceiptState
    federation_admissible: bool
    signature: bytes  # 64 bytes

    def canonical_bytes(self) -> bytes:
        """Produce byte-identical output to Rust canonical_bytes().

        Must match bizra-core/src/canonical_receipt.rs exactly.
        """
        buf = bytearray()

        # mission_id (length-prefixed LE u32)
        mid = self.mission_id.encode("utf-8")
        buf.extend(struct.pack("<I", len(mid)))
        buf.extend(mid)

        # genesis_hash
        buf.extend(self.genesis_hash)

        # policy_version (length-prefixed LE u32)
        pv = self.policy_version.encode("utf-8")
        buf.extend(struct.pack("<I", len(pv)))
        buf.extend(pv)

        # verdict (1 byte)
        buf.append(int(self.verdict))

        # ihsan + snr as fixed-point u64
        buf.extend(struct.pack("<Q", round(self.ihsan_score * 1_000_000.0)))
        buf.extend(struct.pack("<Q", round(self.snr_score * 1_000_000.0)))

        # route (1 byte)
        buf.append(int(self.route))

        # timestamps
        buf.extend(struct.pack("<Q", self.received_at))
        buf.extend(struct.pack("<Q", self.sealed_at))

        # evidence hashes
        buf.extend(self.input_hash)
        buf.extend(self.output_hash)
        buf.extend(self.previous_receipt)

        # state (1 byte)
        buf.append(int(self.state))

        # federation flag
        buf.append(1 if self.federation_admissible else 0)

        return bytes(buf)

    def compute_id(self) -> bytes:
        """BLAKE3 receipt ID — must match Rust compute_id()."""
        import blake3

        canonical = self.canonical_bytes()
        h = blake3.blake3(DOMAIN_CANONICAL_RECEIPT + b":" + canonical)
        return h.digest()

    def sign(self, private_key_bytes: bytes) -> None:
        """Sign with Ed25519 and set receipt_id."""
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        key = Ed25519PrivateKey.from_private_bytes(private_key_bytes)
        canonical = self.canonical_bytes()
        self.signature = key.sign(canonical)
        self.receipt_id = self.compute_id()

    def verify(self, public_key_bytes: bytes) -> bool:
        """Verify Ed25519 signature."""
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        key = Ed25519PublicKey.from_public_bytes(public_key_bytes)
        try:
            key.verify(self.signature, self.canonical_bytes())
            return True
        except Exception:
            return False

    def to_dict(self) -> dict:
        return {
            "receipt_id": self.receipt_id.hex(),
            "mission_id": self.mission_id,
            "genesis_hash": self.genesis_hash.hex(),
            "policy_version": self.policy_version,
            "verdict": self.verdict.name,
            "primary_reject": self.primary_reject,
            "ihsan_score": self.ihsan_score,
            "snr_score": self.snr_score,
            "route": self.route.name,
            "received_at": self.received_at,
            "sealed_at": self.sealed_at,
            "input_hash": self.input_hash.hex(),
            "output_hash": self.output_hash.hex(),
            "previous_receipt": self.previous_receipt.hex(),
            "state": self.state.name,
            "federation_admissible": self.federation_admissible,
            "signature": self.signature.hex(),
        }


def from_mission_result(
    mission_id: str,
    genesis_hash: bytes,
    policy_version: str,
    verdict: VerdictStatus,
    ihsan_score: float,
    snr_score: float,
    route: ExecutionRoute,
    input_text: str,
    output_text: str,
    previous_receipt: bytes,
    received_at: int,
    primary_reject: Optional[str] = None,
) -> CanonicalReceipt:
    """Build a CanonicalReceipt from mission execution results."""
    import blake3

    now_ms = int(time.time() * 1000)
    input_hash = blake3.blake3(input_text.encode()).digest()
    output_hash = blake3.blake3(output_text.encode()).digest()

    state = {
        VerdictStatus.ADMITTED: ReceiptState.COMMITTED,
        VerdictStatus.REJECTED: ReceiptState.VERIFIED,
        VerdictStatus.DEFERRED: ReceiptState.HYPOTHESIS,
    }[verdict]

    federation = (
        verdict == VerdictStatus.ADMITTED and ihsan_score >= IHSAN_FEDERATION_THRESHOLD
    )

    return CanonicalReceipt(
        receipt_id=b"\x00" * 32,
        mission_id=mission_id,
        genesis_hash=genesis_hash,
        policy_version=policy_version,
        verdict=verdict,
        primary_reject=primary_reject,
        ihsan_score=ihsan_score,
        snr_score=snr_score,
        route=route,
        received_at=received_at,
        sealed_at=now_ms,
        input_hash=input_hash,
        output_hash=output_hash,
        previous_receipt=previous_receipt,
        state=state,
        federation_admissible=federation,
        signature=b"\x00" * 64,
    )
