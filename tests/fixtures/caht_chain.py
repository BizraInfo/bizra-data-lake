"""
CAHT — Content-Addressable Hash Trail builder and verifier.

Builds a deterministic hash chain over a sequence of proof trace events.
Each event's hash incorporates the previous event's hash, creating a
tamper-evident, append-only chain identical in semantics to the evidence
ledger but specialized for multi-phase proof traces.

Standing on Giants:
- Lamport (1978): Logical clocks and event ordering
- Merkle (1979): Hash chains for tamper detection
- Shannon (1948): SNR as information quality

Usage:
    chain = CAHTChain()
    for event in events:
        entry = chain.append(event)
    assert chain.verify()  # Full chain integrity check
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

# Import BLAKE3 from proof engine (deterministic, cross-language compatible)
from core.proof_engine.canonical import blake3_digest, canonical_bytes

# Genesis hash — same sentinel as evidence_ledger.py
GENESIS_HASH = "0" * 64


@dataclass
class CAHTEntry:
    """Single entry in the Content-Addressable Hash Trail."""

    seq: int
    phase: str
    event_type: str
    actor: str
    payload_hash: str
    prev_hash: str
    entry_hash: str
    ihsan_score: float
    snr_score: float
    receipt_status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seq": self.seq,
            "phase": self.phase,
            "event_type": self.event_type,
            "actor": self.actor,
            "payload_hash": self.payload_hash,
            "prev_hash": self.prev_hash,
            "entry_hash": self.entry_hash,
            "ihsan_score": self.ihsan_score,
            "snr_score": self.snr_score,
            "receipt_status": self.receipt_status,
        }


def _compute_caht_hash(
    seq: int,
    phase: str,
    event_type: str,
    actor: str,
    payload_hash: str,
    prev_hash: str,
    ihsan_score: float,
    snr_score: float,
) -> str:
    """Compute BLAKE3 hash for a CAHT entry (deterministic).

    Uses canonical_bytes for RFC 8785 JCS compliance.
    """
    obj = {
        "seq": seq,
        "phase": phase,
        "event_type": event_type,
        "actor": actor,
        "payload_hash": payload_hash,
        "prev_hash": prev_hash,
        "ihsan_score": ihsan_score,
        "snr_score": snr_score,
    }
    digest = blake3_digest(canonical_bytes(obj))
    if isinstance(digest, bytes):
        return digest.hex()
    return str(digest)


class CAHTChain:
    """Content-Addressable Hash Trail — append-only, tamper-evident event chain."""

    def __init__(self) -> None:
        self.entries: List[CAHTEntry] = []
        self._last_hash: str = GENESIS_HASH

    def append(self, event: Dict[str, Any]) -> CAHTEntry:
        """Append an event to the chain and return the CAHT entry."""
        seq = event["seq"]
        phase = event["phase"]
        event_type = event["event_type"]
        actor = event["actor"]
        ihsan = event.get("ihsan_score", 0.0)
        snr = event.get("snr_score", 0.0)
        receipt_status = event.get("receipt_status", "pending")

        # Hash the payload deterministically
        payload = event.get("payload", {})
        payload_digest = blake3_digest(canonical_bytes(payload))
        payload_hash = (
            payload_digest.hex()
            if isinstance(payload_digest, bytes)
            else str(payload_digest)
        )

        # Compute entry hash incorporating prev_hash
        entry_hash = _compute_caht_hash(
            seq=seq,
            phase=phase,
            event_type=event_type,
            actor=actor,
            payload_hash=payload_hash,
            prev_hash=self._last_hash,
            ihsan_score=ihsan,
            snr_score=snr,
        )

        entry = CAHTEntry(
            seq=seq,
            phase=phase,
            event_type=event_type,
            actor=actor,
            payload_hash=payload_hash,
            prev_hash=self._last_hash,
            entry_hash=entry_hash,
            ihsan_score=ihsan,
            snr_score=snr,
            receipt_status=receipt_status,
        )

        self.entries.append(entry)
        self._last_hash = entry_hash
        return entry

    def verify(self) -> bool:
        """Verify the entire chain is tamper-free.

        Replays all hashes from genesis and checks each link.
        Returns True if chain is intact, False if any link broken.
        """
        if not self.entries:
            return True

        prev = GENESIS_HASH
        for entry in self.entries:
            if entry.prev_hash != prev:
                return False

            expected = _compute_caht_hash(
                seq=entry.seq,
                phase=entry.phase,
                event_type=entry.event_type,
                actor=entry.actor,
                payload_hash=entry.payload_hash,
                prev_hash=entry.prev_hash,
                ihsan_score=entry.ihsan_score,
                snr_score=entry.snr_score,
            )
            if entry.entry_hash != expected:
                return False

            prev = entry.entry_hash

        return True

    @property
    def head_hash(self) -> str:
        """Hash of the most recent entry (chain tip)."""
        return self._last_hash

    def __len__(self) -> int:
        return len(self.entries)
