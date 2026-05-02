"""BIZRA Mission Kernel v0.1.

Deterministic trust spine for Node0:
Intent -> Constitution -> Proposal -> FATE -> Ihsan -> SAT -> Receipt -> Replay.

Python remains a proposal/contractor surface. This package owns the typed
receipt contract and local verification CLI for the first Node0 proof slice.
"""

from __future__ import annotations

from core.mission_kernel.chain import ChainReport, JsonlReceiptStore
from core.mission_kernel.identity import IdentityRecord, IdentityRegistry
from core.mission_kernel.receipt import (
    RECEIPT_SCHEMA_VERSION,
    Decision,
    FateVerdict,
    MissionState,
    Proposal,
    ReceiptV1,
    SatConsensus,
    create_receipt,
    verify_receipt,
)

__all__ = [
    "RECEIPT_SCHEMA_VERSION",
    "ChainReport",
    "Decision",
    "FateVerdict",
    "IdentityRecord",
    "IdentityRegistry",
    "JsonlReceiptStore",
    "MissionState",
    "Proposal",
    "ReceiptV1",
    "SatConsensus",
    "create_receipt",
    "verify_receipt",
]
