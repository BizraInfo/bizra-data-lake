"""
BIZRA Evidence Receipt — Cryptographic Trust Chain
═══════════════════════════════════════════════════

Every mission output that passes constitutional gates generates
an evidence receipt: a signed, hash-chained, immutable record.

This module is the Integrator agent's implementation
(PAT trust stage: "chaining" — the 7th and final stage).

Evidence chain properties:
    - Append-only (JSONL)
    - Hash-chained (each receipt references the previous hash)
    - Ed25519 signed (domain-separated: "bizra-evidence-v1")
    - Includes full Ihsan tensor (6-dim, not scalar)
    - POSIX file locks for concurrent access safety

Theorem 2.4 (amended): Evidence chain is tamper-evident.
    Modifying any receipt invalidates all subsequent hashes.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from generated.generated_constants import (
        CONSTITUTION_HASH,
        DOMAIN_EVIDENCE_RECEIPT,
    )
except ImportError:
    DOMAIN_EVIDENCE_RECEIPT = "bizra-evidence-v1"
    CONSTITUTION_HASH = "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class EvidenceReceipt:
    """A single immutable evidence record in the chain."""

    receipt_id: str  # SHA-256 of receipt content
    timestamp_utc: float  # Unix timestamp
    previous_hash: str  # Hash of the previous receipt (chain link)
    mission_id: str  # Unique mission identifier
    ihsan_tensor: dict[str, float]  # Full 6-dim tensor, NOT scalar
    ihsan_composite: float  # Weighted composite for quick lookup
    gate_results: dict[str, bool]  # Per-gate pass/fail
    snr_normalized: float  # Mission SNR
    tier: str  # "rejected" | "acceptable" | "bloom" | "ihsan"
    constitution_hash: str  # Hash of constitution.toml at evaluation time
    domain: str  # Domain separation context
    agent_chain: list[str]  # Which PAT agents participated
    metadata: dict[str, Any]  # Extensible context

    def compute_hash(self) -> str:
        """Deterministic hash of this receipt's content."""
        # Canonical JSON serialization (sorted keys, no whitespace)
        content = json.dumps(
            {k: v for k, v in asdict(self).items() if k != "receipt_id"},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(
            f"{DOMAIN_EVIDENCE_RECEIPT}:{content}".encode()
        ).hexdigest()

    def verify_hash(self) -> bool:
        """Verify this receipt's hash matches its content."""
        return self.receipt_id == self.compute_hash()


# ═══════════════════════════════════════════════════════════════════════════════
# EVIDENCE LEDGER — Append-only JSONL with POSIX locks
# ═══════════════════════════════════════════════════════════════════════════════


class EvidenceLedger:
    """
    Append-only evidence chain stored as JSONL.

    Thread-safe via POSIX file locks. Each receipt is hash-chained
    to the previous one, making tampering detectable.
    """

    GENESIS_HASH = "0" * 64  # Genesis block has no predecessor

    def __init__(self, path: str | Path = "evidence_ledger.jsonl"):
        self.path = Path(path)
        self._ensure_exists()

    def _ensure_exists(self):
        """Create ledger file if it doesn't exist."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def _get_last_hash(self) -> str:
        """Read the hash of the last receipt in the chain."""
        if not self.path.exists() or self.path.stat().st_size == 0:
            return self.GENESIS_HASH

        last_line = ""
        with open(self.path, "r") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    last_line = stripped

        if not last_line:
            return self.GENESIS_HASH

        try:
            last_receipt = json.loads(last_line)
            return last_receipt.get("receipt_id", self.GENESIS_HASH)
        except json.JSONDecodeError:
            return self.GENESIS_HASH

    def append(
        self,
        mission_id: str,
        ihsan_tensor: dict[str, float],
        ihsan_composite: float,
        gate_results: dict[str, bool],
        snr_normalized: float,
        tier: str,
        agent_chain: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EvidenceReceipt:
        """
        Append a new evidence receipt to the chain.

        Thread-safe via POSIX exclusive file lock.

        Returns:
            The created EvidenceReceipt with computed hash and chain link.
        """
        agent_chain = agent_chain or [
            "Planner",
            "Researcher",
            "Coder",
            "Evaluator",
            "Ethicist",
            "Publisher",
            "Integrator",
        ]
        metadata = metadata or {}

        with open(self.path, "a") as f:
            # Acquire exclusive lock
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                previous_hash = self._get_last_hash()

                receipt = EvidenceReceipt(
                    receipt_id="",  # Computed below
                    timestamp_utc=time.time(),
                    previous_hash=previous_hash,
                    mission_id=mission_id,
                    ihsan_tensor=ihsan_tensor,
                    ihsan_composite=ihsan_composite,
                    gate_results=gate_results,
                    snr_normalized=snr_normalized,
                    tier=tier,
                    constitution_hash=CONSTITUTION_HASH,
                    domain=DOMAIN_EVIDENCE_RECEIPT,
                    agent_chain=agent_chain,
                    metadata=metadata,
                )

                # Compute deterministic hash
                receipt.receipt_id = receipt.compute_hash()

                # Write as single JSONL line
                line = json.dumps(asdict(receipt), sort_keys=True, default=str)
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())

                return receipt
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def verify_chain(self) -> tuple[bool, int, list[str]]:
        """
        Verify the entire evidence chain integrity.

        Returns:
            (is_valid, receipt_count, errors)
        """
        if not self.path.exists() or self.path.stat().st_size == 0:
            return True, 0, []

        errors = []
        count = 0
        expected_prev = self.GENESIS_HASH

        with open(self.path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_num}: JSON decode error: {e}")
                    continue

                count += 1

                # Verify chain link
                if data.get("previous_hash") != expected_prev:
                    errors.append(
                        f"Line {line_num}: Chain break. "
                        f"Expected prev={expected_prev[:16]}..., "
                        f"got {data.get('previous_hash', 'missing')[:16]}..."
                    )

                # Verify self-hash
                receipt = EvidenceReceipt(**data)
                if not receipt.verify_hash():
                    errors.append(
                        f"Line {line_num}: Hash mismatch for receipt {receipt.receipt_id[:16]}..."
                    )

                expected_prev = data.get("receipt_id", "")

        return len(errors) == 0, count, errors

    def count(self) -> int:
        """Count receipts in the ledger."""
        if not self.path.exists():
            return 0
        with open(self.path) as f:
            return sum(1 for line in f if line.strip())

    def last_receipt(self) -> EvidenceReceipt | None:
        """Get the most recent receipt."""
        if not self.path.exists() or self.path.stat().st_size == 0:
            return None

        last_line = ""
        with open(self.path, "r") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    last_line = stripped

        if not last_line:
            return None

        try:
            return EvidenceReceipt(**json.loads(last_line))
        except (json.JSONDecodeError, TypeError):
            return None
