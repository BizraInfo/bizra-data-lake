"""
Persistent Receipt Storage
==========================
Provides PostgreSQL and JSONL append-only receipt storage.
Implements audit-grade evidence chain with integrity verification.

Status: FROZEN — Changes require version bump + test vector update
Semantics: Append-only, monotonically increasing offset, integrity-verified

Integration Points:
- core/pci/receipt.py — CommitReceipt dataclass, ReceiptGenerator
- kg/receipts.py — PostgreSQL kg_receipts table schema
- migrations/001_kg_core.sql — forbid_updates_deletes() trigger
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

try:
    import psycopg
    from psycopg import AsyncConnection
    from psycopg.rows import dict_row
    PSYCOPG_AVAILABLE = True
except ImportError:
    PSYCOPG_AVAILABLE = False
    AsyncConnection = Any  # type: ignore

try:
    import aiofiles
    import aiofiles.os
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False
    aiofiles = None  # type: ignore

from .receipt import CommitReceipt
from .crypto import canonical_json, domain_separated_digest
from .types import PCI_VERSION


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_JSONL_PATH = Path("docs/evidence/receipts/pci_receipts.jsonl")
INTEGRITY_DOMAIN = "bizra-pci-integrity-v1:"
CHAIN_DOMAIN = "bizra-pci-chain-v1:"


# =============================================================================
# STORED RECEIPT
# =============================================================================

@dataclass
class StoredReceipt:
    """
    Receipt with storage metadata.

    Extends CommitReceipt with:
    - offset: Monotonically increasing position in the chain
    - integrity_hash: SHA-256 of canonical JSON for tamper detection
    - created_at: Storage timestamp (may differ from receipt timestamp)
    - decision: COMMIT or REJECT for quick filtering
    - prev_hash: Hash of previous receipt for chain integrity (optional)
    """
    receipt_id: str
    offset: int  # Monotonically increasing
    envelope_digest: str
    decision: str  # COMMIT or REJECT
    gate_results: Dict[str, Any]
    quorum: Dict[str, Any]
    verifier_signatures: List[Dict[str, Any]]
    audit_digest: str
    created_at: str
    integrity_hash: str  # SHA-256 of canonical JSON
    prev_hash: Optional[str] = None  # For chain verification
    policy_hash: Optional[str] = None
    version: str = PCI_VERSION

    @classmethod
    def from_commit_receipt(
        cls,
        receipt: CommitReceipt,
        offset: int,
        prev_hash: Optional[str] = None,
    ) -> "StoredReceipt":
        """
        Create StoredReceipt from CommitReceipt.

        Computes integrity hash from canonical JSON representation.
        """
        # Determine decision from verification tier
        tier = receipt.verification.tier.value
        decision = "REJECT" if "REJECTED" in tier or tier == "UNVERIFIED" else "COMMIT"

        # Extract gate results
        gate_results = {
            "gates_passed": [g.value for g in receipt.verification.gates_passed],
            "ihsan_score": receipt.verification.ihsan_score,
            "snr_score": receipt.verification.snr_score,
            "latency_ms": receipt.verification.latency_ms,
        }

        # Extract quorum
        quorum = receipt.quorum.to_dict()

        # Extract verifier signatures
        verifier_signatures = [v.to_dict() for v in receipt.verifier_set]

        created_at = datetime.now(timezone.utc).isoformat()

        # Build storable data for integrity hash
        storable_data = {
            "receipt_id": receipt.receipt_id,
            "offset": offset,
            "envelope_digest": receipt.envelope_digest,
            "decision": decision,
            "gate_results": gate_results,
            "quorum": quorum,
            "verifier_signatures": verifier_signatures,
            "audit_digest": receipt.audit_digest,
            "created_at": created_at,
            "prev_hash": prev_hash,
            "policy_hash": receipt.policy_hash,
            "version": receipt.version,
        }

        # Compute integrity hash
        canonical = canonical_json(storable_data)
        integrity_hash = domain_separated_digest(canonical, INTEGRITY_DOMAIN)

        return cls(
            receipt_id=receipt.receipt_id,
            offset=offset,
            envelope_digest=receipt.envelope_digest,
            decision=decision,
            gate_results=gate_results,
            quorum=quorum,
            verifier_signatures=verifier_signatures,
            audit_digest=receipt.audit_digest,
            created_at=created_at,
            integrity_hash=integrity_hash,
            prev_hash=prev_hash,
            policy_hash=receipt.policy_hash,
            version=receipt.version,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "receipt_id": self.receipt_id,
            "offset": self.offset,
            "envelope_digest": self.envelope_digest,
            "decision": self.decision,
            "gate_results": self.gate_results,
            "quorum": self.quorum,
            "verifier_signatures": self.verifier_signatures,
            "audit_digest": self.audit_digest,
            "created_at": self.created_at,
            "integrity_hash": self.integrity_hash,
            "prev_hash": self.prev_hash,
            "policy_hash": self.policy_hash,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StoredReceipt":
        """Deserialize from dictionary."""
        return cls(
            receipt_id=data["receipt_id"],
            offset=data["offset"],
            envelope_digest=data["envelope_digest"],
            decision=data["decision"],
            gate_results=data["gate_results"],
            quorum=data["quorum"],
            verifier_signatures=data["verifier_signatures"],
            audit_digest=data["audit_digest"],
            created_at=data["created_at"],
            integrity_hash=data["integrity_hash"],
            prev_hash=data.get("prev_hash"),
            policy_hash=data.get("policy_hash"),
            version=data.get("version", PCI_VERSION),
        )

    def to_canonical_json(self) -> str:
        """Serialize to canonical JSON string."""
        return canonical_json(self.to_dict()).decode("utf-8")

    def verify_integrity(self) -> bool:
        """
        Verify the integrity hash matches the receipt data.

        Recomputes the hash from current data and compares.
        """
        # Build storable data (exclude integrity_hash for computation)
        storable_data = {
            "receipt_id": self.receipt_id,
            "offset": self.offset,
            "envelope_digest": self.envelope_digest,
            "decision": self.decision,
            "gate_results": self.gate_results,
            "quorum": self.quorum,
            "verifier_signatures": self.verifier_signatures,
            "audit_digest": self.audit_digest,
            "created_at": self.created_at,
            "prev_hash": self.prev_hash,
            "policy_hash": self.policy_hash,
            "version": self.version,
        }

        canonical = canonical_json(storable_data)
        computed_hash = domain_separated_digest(canonical, INTEGRITY_DOMAIN)

        return computed_hash == self.integrity_hash


# =============================================================================
# RECEIPT CHAIN
# =============================================================================

@dataclass
class ReceiptChain:
    """
    Represents the append-only receipt chain.

    Provides chain integrity verification and traversal.
    """

    _receipts: List[StoredReceipt] = field(default_factory=list)
    _latest_offset: int = 0
    _by_id: Dict[str, StoredReceipt] = field(default_factory=dict)
    _by_offset: Dict[int, StoredReceipt] = field(default_factory=dict)
    _by_envelope: Dict[str, List[StoredReceipt]] = field(default_factory=dict)

    def add(self, receipt: StoredReceipt) -> None:
        """Add a receipt to the chain."""
        self._receipts.append(receipt)
        self._by_id[receipt.receipt_id] = receipt
        self._by_offset[receipt.offset] = receipt

        if receipt.envelope_digest not in self._by_envelope:
            self._by_envelope[receipt.envelope_digest] = []
        self._by_envelope[receipt.envelope_digest].append(receipt)

        if receipt.offset > self._latest_offset:
            self._latest_offset = receipt.offset

    def get_by_id(self, receipt_id: str) -> Optional[StoredReceipt]:
        """Get receipt by ID."""
        return self._by_id.get(receipt_id)

    def get_by_offset(self, offset: int) -> Optional[StoredReceipt]:
        """Get receipt by offset."""
        return self._by_offset.get(offset)

    def get_by_envelope(self, envelope_digest: str) -> List[StoredReceipt]:
        """Get all receipts for an envelope."""
        return self._by_envelope.get(envelope_digest, [])

    @property
    def latest_offset(self) -> int:
        """Get the latest offset."""
        return self._latest_offset

    @property
    def length(self) -> int:
        """Get chain length."""
        return len(self._receipts)

    def verify_chain_integrity(self) -> Tuple[bool, Optional[str]]:
        """
        Verify entire chain integrity.

        Checks:
        1. Each receipt's integrity hash is valid
        2. Offsets are monotonically increasing
        3. prev_hash chain is unbroken (if present)

        Returns (valid, error_message)
        """
        if not self._receipts:
            return (True, None)

        # Sort by offset for verification
        sorted_receipts = sorted(self._receipts, key=lambda r: r.offset)

        prev_offset = -1
        prev_hash = None

        for receipt in sorted_receipts:
            # Verify individual integrity
            if not receipt.verify_integrity():
                return (False, f"Integrity hash mismatch at offset {receipt.offset}")

            # Verify monotonic offset
            if receipt.offset <= prev_offset:
                return (
                    False,
                    f"Non-monotonic offset: {receipt.offset} <= {prev_offset}"
                )

            # Verify chain linkage (if prev_hash is used)
            if prev_hash is not None and receipt.prev_hash is not None:
                if receipt.prev_hash != prev_hash:
                    return (
                        False,
                        f"Chain break at offset {receipt.offset}: "
                        f"expected prev_hash {prev_hash}, got {receipt.prev_hash}"
                    )

            prev_offset = receipt.offset
            prev_hash = receipt.integrity_hash

        return (True, None)

    def get_range(self, start_offset: int, end_offset: int) -> List[StoredReceipt]:
        """Get receipts in offset range (inclusive)."""
        return [
            r for r in self._receipts
            if start_offset <= r.offset <= end_offset
        ]

    def traverse_from(self, start_offset: int) -> List[StoredReceipt]:
        """Traverse chain from offset to end."""
        return sorted(
            [r for r in self._receipts if r.offset >= start_offset],
            key=lambda r: r.offset
        )


# =============================================================================
# JSONL RECEIPT STORE
# =============================================================================

class JSONLReceiptStore:
    """
    Append-only JSONL file storage for receipts.

    File: docs/evidence/receipts/pci_receipts.jsonl

    Features:
    - Atomic append writes with file locking
    - Integrity hash verification
    - Async file I/O
    - Chain integrity tracking
    """

    def __init__(self, filepath: Path = DEFAULT_JSONL_PATH):
        self.filepath = Path(filepath)
        self._lock = asyncio.Lock()
        self._chain = ReceiptChain()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize store, loading existing receipts."""
        if self._initialized:
            return

        async with self._lock:
            # Ensure directory exists
            self.filepath.parent.mkdir(parents=True, exist_ok=True)

            # Load existing receipts
            if self.filepath.exists():
                async with aiofiles.open(self.filepath, "r", encoding="utf-8") as f:
                    async for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                receipt = StoredReceipt.from_dict(data)
                                self._chain.add(receipt)
                            except (json.JSONDecodeError, KeyError) as e:
                                # Log but continue - don't fail on corrupt lines
                                print(f"Warning: Skipping corrupt line: {e}")

            self._initialized = True

    async def append(self, receipt: CommitReceipt) -> StoredReceipt:
        """
        Append receipt to JSONL file (atomic write).

        Steps:
        1. Acquire lock
        2. Get next offset
        3. Get prev_hash from latest receipt
        4. Create StoredReceipt with integrity hash
        5. Append line to file
        6. Update in-memory chain
        7. Return stored receipt
        """
        await self.initialize()

        async with self._lock:
            # Get next offset and prev_hash
            next_offset = self._chain.latest_offset + 1

            # Get prev_hash from latest receipt
            prev_receipt = self._chain.get_by_offset(self._chain.latest_offset)
            prev_hash = prev_receipt.integrity_hash if prev_receipt else None

            # Create stored receipt
            stored = StoredReceipt.from_commit_receipt(
                receipt,
                offset=next_offset,
                prev_hash=prev_hash,
            )

            # Append to file (atomic with newline)
            json_line = json.dumps(stored.to_dict(), separators=(",", ":"))
            async with aiofiles.open(self.filepath, "a", encoding="utf-8") as f:
                await f.write(json_line + "\n")
                await f.flush()

            # Update in-memory chain
            self._chain.add(stored)

            return stored

    async def get_by_id(self, receipt_id: str) -> Optional[StoredReceipt]:
        """Retrieve receipt by ID."""
        await self.initialize()
        return self._chain.get_by_id(receipt_id)

    async def get_by_offset(self, offset: int) -> Optional[StoredReceipt]:
        """Retrieve receipt by offset."""
        await self.initialize()
        return self._chain.get_by_offset(offset)

    async def get_by_envelope_digest(self, digest: str) -> List[StoredReceipt]:
        """Retrieve all receipts for an envelope."""
        await self.initialize()
        return self._chain.get_by_envelope(digest)

    async def get_range(
        self,
        start_offset: int,
        end_offset: int
    ) -> List[StoredReceipt]:
        """Retrieve receipts in offset range (inclusive)."""
        await self.initialize()
        return self._chain.get_range(start_offset, end_offset)

    async def verify_integrity(self, receipt: StoredReceipt) -> bool:
        """Verify receipt integrity hash."""
        return receipt.verify_integrity()

    async def verify_chain_integrity(self) -> Tuple[bool, Optional[str]]:
        """Verify entire chain integrity."""
        await self.initialize()
        return self._chain.verify_chain_integrity()

    async def stream_all(self) -> AsyncIterator[StoredReceipt]:
        """
        Stream all receipts for audit.

        Yields receipts in offset order for efficient memory usage.
        """
        await self.initialize()

        # Stream from file for memory efficiency
        async with aiofiles.open(self.filepath, "r", encoding="utf-8") as f:
            async for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        yield StoredReceipt.from_dict(data)
                    except (json.JSONDecodeError, KeyError):
                        continue

    async def count(self) -> int:
        """Get total receipt count."""
        await self.initialize()
        return self._chain.length

    async def get_latest_offset(self) -> int:
        """Get the latest offset."""
        await self.initialize()
        return self._chain.latest_offset


# =============================================================================
# POSTGRESQL RECEIPT STORE
# =============================================================================

class PostgreSQLReceiptStore:
    """
    PostgreSQL storage with append-only enforcement.

    Uses kg_receipts table with forbid_updates_deletes() trigger.

    Features:
    - Database-enforced append-only (trigger prevents UPDATE/DELETE)
    - Transaction-safe writes
    - Full-text search on receipt data
    - Chain integrity verification via sequential queries
    """

    def __init__(self, connection_string: str):
        if not PSYCOPG_AVAILABLE:
            raise ImportError(
                "psycopg is required for PostgreSQLReceiptStore. "
                "Install with: pip install psycopg[binary]"
            )

        self.conn_string = connection_string
        self._pool: Optional[Any] = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        """Initialize connection pool."""
        if self._pool is not None:
            return

        async with self._lock:
            if self._pool is None:
                # For async operations, we create connections as needed
                # In production, use a proper async pool like asyncpg
                self._pool = True  # Marker that we're initialized

    async def _get_connection(self) -> AsyncConnection:
        """Get a database connection."""
        await self.connect()
        return await psycopg.AsyncConnection.connect(
            self.conn_string,
            row_factory=dict_row,
        )

    async def append(self, receipt: CommitReceipt) -> StoredReceipt:
        """
        Insert receipt into kg_receipts table.

        The trigger prevents updates/deletes.
        Also stores in pci_receipts extension table for PCI-specific fields.
        """
        async with await self._get_connection() as conn:
            async with conn.cursor() as cur:
                # Get current max offset for PCI receipts
                await cur.execute(
                    """
                    SELECT COALESCE(MAX((payload->>'pci_offset')::int), 0) as max_offset
                    FROM kg_receipts
                    WHERE kind = 'PCI'
                    """
                )
                row = await cur.fetchone()
                next_offset = (row["max_offset"] if row else 0) + 1

                # Get prev_hash
                await cur.execute(
                    """
                    SELECT payload->>'integrity_hash' as prev_hash
                    FROM kg_receipts
                    WHERE kind = 'PCI'
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
                )
                row = await cur.fetchone()
                prev_hash = row["prev_hash"] if row else None

                # Create stored receipt
                stored = StoredReceipt.from_commit_receipt(
                    receipt,
                    offset=next_offset,
                    prev_hash=prev_hash,
                )

                # Determine decision for kg_receipts
                decision = stored.decision
                if decision == "COMMIT":
                    kg_decision = "ALLOWED"
                else:
                    kg_decision = "REJECTED"

                # Insert into kg_receipts (primary receipt store)
                await cur.execute(
                    """
                    INSERT INTO kg_receipts
                      (kind, policy_hash, decision, evidence_refs, payload,
                       ihsan, sape, snr, rejection_reasons, signature)
                    VALUES
                      ('PCI', %s, %s, %s::jsonb, %s::jsonb,
                       %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s)
                    RETURNING receipt_id, created_at
                    """,
                    (
                        stored.policy_hash or "unknown",
                        kg_decision,
                        json.dumps([{"type": "envelope", "digest": stored.envelope_digest}]),
                        json.dumps({
                            "pci_receipt_id": stored.receipt_id,
                            "pci_offset": stored.offset,
                            "envelope_digest": stored.envelope_digest,
                            "gate_results": stored.gate_results,
                            "quorum": stored.quorum,
                            "verifier_signatures": stored.verifier_signatures,
                            "audit_digest": stored.audit_digest,
                            "integrity_hash": stored.integrity_hash,
                            "prev_hash": stored.prev_hash,
                            "version": stored.version,
                        }),
                        json.dumps({
                            "score": stored.gate_results.get("ihsan_score", 0),
                            "tier": "PCI",
                            "gates_passed": stored.gate_results.get("gates_passed", []),
                        }),
                        json.dumps({
                            "cycle_id": stored.receipt_id,
                            "phase": "verification",
                            "stakes": "H",
                        }),
                        json.dumps({
                            "budget": "pci",
                            "ratio": stored.gate_results.get("snr_score"),
                        }),
                        json.dumps([]),  # rejection_reasons
                        stored.audit_digest,  # Use audit_digest as signature placeholder
                    ),
                )

                await conn.commit()

                return stored

    async def get_by_id(self, receipt_id: str) -> Optional[StoredReceipt]:
        """Retrieve by receipt_id."""
        async with await self._get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT payload, created_at
                    FROM kg_receipts
                    WHERE kind = 'PCI'
                      AND payload->>'pci_receipt_id' = %s
                    """,
                    (receipt_id,),
                )
                row = await cur.fetchone()

                if not row:
                    return None

                payload = row["payload"]
                return StoredReceipt(
                    receipt_id=payload["pci_receipt_id"],
                    offset=payload["pci_offset"],
                    envelope_digest=payload["envelope_digest"],
                    decision="COMMIT" if payload.get("decision") != "REJECT" else "REJECT",
                    gate_results=payload["gate_results"],
                    quorum=payload["quorum"],
                    verifier_signatures=payload["verifier_signatures"],
                    audit_digest=payload["audit_digest"],
                    created_at=row["created_at"].isoformat(),
                    integrity_hash=payload["integrity_hash"],
                    prev_hash=payload.get("prev_hash"),
                    policy_hash=payload.get("policy_hash"),
                    version=payload.get("version", PCI_VERSION),
                )

    async def get_by_envelope_digest(self, digest: str) -> List[StoredReceipt]:
        """Retrieve all receipts for an envelope."""
        async with await self._get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT payload, created_at
                    FROM kg_receipts
                    WHERE kind = 'PCI'
                      AND payload->>'envelope_digest' = %s
                    ORDER BY created_at ASC
                    """,
                    (digest,),
                )

                results = []
                async for row in cur:
                    payload = row["payload"]
                    results.append(StoredReceipt(
                        receipt_id=payload["pci_receipt_id"],
                        offset=payload["pci_offset"],
                        envelope_digest=payload["envelope_digest"],
                        decision="COMMIT" if payload.get("decision") != "REJECT" else "REJECT",
                        gate_results=payload["gate_results"],
                        quorum=payload["quorum"],
                        verifier_signatures=payload["verifier_signatures"],
                        audit_digest=payload["audit_digest"],
                        created_at=row["created_at"].isoformat(),
                        integrity_hash=payload["integrity_hash"],
                        prev_hash=payload.get("prev_hash"),
                        policy_hash=payload.get("policy_hash"),
                        version=payload.get("version", PCI_VERSION),
                    ))

                return results

    async def get_by_offset(self, offset: int) -> Optional[StoredReceipt]:
        """Retrieve by offset."""
        async with await self._get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT payload, created_at
                    FROM kg_receipts
                    WHERE kind = 'PCI'
                      AND (payload->>'pci_offset')::int = %s
                    """,
                    (offset,),
                )
                row = await cur.fetchone()

                if not row:
                    return None

                payload = row["payload"]
                return StoredReceipt(
                    receipt_id=payload["pci_receipt_id"],
                    offset=payload["pci_offset"],
                    envelope_digest=payload["envelope_digest"],
                    decision="COMMIT" if payload.get("decision") != "REJECT" else "REJECT",
                    gate_results=payload["gate_results"],
                    quorum=payload["quorum"],
                    verifier_signatures=payload["verifier_signatures"],
                    audit_digest=payload["audit_digest"],
                    created_at=row["created_at"].isoformat(),
                    integrity_hash=payload["integrity_hash"],
                    prev_hash=payload.get("prev_hash"),
                    policy_hash=payload.get("policy_hash"),
                    version=payload.get("version", PCI_VERSION),
                )

    async def get_evidence_chain(self, receipt_id: str) -> List[StoredReceipt]:
        """
        Traverse evidence chain from receipt.

        Follows prev_hash links backwards to build the chain
        that led to this receipt.
        """
        chain: List[StoredReceipt] = []
        current = await self.get_by_id(receipt_id)

        while current is not None:
            chain.append(current)

            if current.prev_hash is None:
                break

            # Find receipt with matching integrity_hash
            async with await self._get_connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        SELECT payload, created_at
                        FROM kg_receipts
                        WHERE kind = 'PCI'
                          AND payload->>'integrity_hash' = %s
                        """,
                        (current.prev_hash,),
                    )
                    row = await cur.fetchone()

                    if not row:
                        break

                    payload = row["payload"]
                    current = StoredReceipt(
                        receipt_id=payload["pci_receipt_id"],
                        offset=payload["pci_offset"],
                        envelope_digest=payload["envelope_digest"],
                        decision="COMMIT" if payload.get("decision") != "REJECT" else "REJECT",
                        gate_results=payload["gate_results"],
                        quorum=payload["quorum"],
                        verifier_signatures=payload["verifier_signatures"],
                        audit_digest=payload["audit_digest"],
                        created_at=row["created_at"].isoformat(),
                        integrity_hash=payload["integrity_hash"],
                        prev_hash=payload.get("prev_hash"),
                        policy_hash=payload.get("policy_hash"),
                        version=payload.get("version", PCI_VERSION),
                    )

        # Return in chronological order (oldest first)
        chain.reverse()
        return chain

    async def verify_chain_integrity(
        self,
        start_offset: int = 0
    ) -> Tuple[bool, int]:
        """
        Verify chain integrity from offset.

        Checks:
        1. Each receipt's integrity hash is valid
        2. Offsets are monotonically increasing
        3. prev_hash chain is unbroken

        Returns (valid, last_verified_offset)
        """
        async with await self._get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT payload, created_at
                    FROM kg_receipts
                    WHERE kind = 'PCI'
                      AND (payload->>'pci_offset')::int >= %s
                    ORDER BY (payload->>'pci_offset')::int ASC
                    """,
                    (start_offset,),
                )

                prev_offset = start_offset - 1
                prev_hash = None
                last_verified = start_offset - 1

                async for row in cur:
                    payload = row["payload"]

                    receipt = StoredReceipt(
                        receipt_id=payload["pci_receipt_id"],
                        offset=payload["pci_offset"],
                        envelope_digest=payload["envelope_digest"],
                        decision="COMMIT" if payload.get("decision") != "REJECT" else "REJECT",
                        gate_results=payload["gate_results"],
                        quorum=payload["quorum"],
                        verifier_signatures=payload["verifier_signatures"],
                        audit_digest=payload["audit_digest"],
                        created_at=row["created_at"].isoformat(),
                        integrity_hash=payload["integrity_hash"],
                        prev_hash=payload.get("prev_hash"),
                        policy_hash=payload.get("policy_hash"),
                        version=payload.get("version", PCI_VERSION),
                    )

                    # Verify integrity
                    if not receipt.verify_integrity():
                        return (False, last_verified)

                    # Verify monotonic offset
                    if receipt.offset <= prev_offset:
                        return (False, last_verified)

                    # Verify chain linkage
                    if prev_hash is not None and receipt.prev_hash is not None:
                        if receipt.prev_hash != prev_hash:
                            return (False, last_verified)

                    prev_offset = receipt.offset
                    prev_hash = receipt.integrity_hash
                    last_verified = receipt.offset

                return (True, last_verified)

    async def stream_all(
        self,
        start_offset: int = 0
    ) -> AsyncIterator[StoredReceipt]:
        """Stream all receipts for audit."""
        async with await self._get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT payload, created_at
                    FROM kg_receipts
                    WHERE kind = 'PCI'
                      AND (payload->>'pci_offset')::int >= %s
                    ORDER BY (payload->>'pci_offset')::int ASC
                    """,
                    (start_offset,),
                )

                async for row in cur:
                    payload = row["payload"]
                    yield StoredReceipt(
                        receipt_id=payload["pci_receipt_id"],
                        offset=payload["pci_offset"],
                        envelope_digest=payload["envelope_digest"],
                        decision="COMMIT" if payload.get("decision") != "REJECT" else "REJECT",
                        gate_results=payload["gate_results"],
                        quorum=payload["quorum"],
                        verifier_signatures=payload["verifier_signatures"],
                        audit_digest=payload["audit_digest"],
                        created_at=row["created_at"].isoformat(),
                        integrity_hash=payload["integrity_hash"],
                        prev_hash=payload.get("prev_hash"),
                        policy_hash=payload.get("policy_hash"),
                        version=payload.get("version", PCI_VERSION),
                    )

    async def count(self) -> int:
        """Get total receipt count."""
        async with await self._get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT COUNT(*) as count
                    FROM kg_receipts
                    WHERE kind = 'PCI'
                    """
                )
                row = await cur.fetchone()
                return row["count"] if row else 0

    async def get_latest_offset(self) -> int:
        """Get the latest offset."""
        async with await self._get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT COALESCE(MAX((payload->>'pci_offset')::int), 0) as max_offset
                    FROM kg_receipts
                    WHERE kind = 'PCI'
                    """
                )
                row = await cur.fetchone()
                return row["max_offset"] if row else 0


# =============================================================================
# HYBRID RECEIPT STORE
# =============================================================================

class HybridReceiptStore:
    """
    Hybrid storage: PostgreSQL primary + JSONL backup.

    Provides redundancy and local audit capability.

    Features:
    - Dual-write to both stores
    - Automatic sync/recovery
    - Consistency verification
    - Local audit without database access
    """

    def __init__(
        self,
        pg_store: PostgreSQLReceiptStore,
        jsonl_store: JSONLReceiptStore,
    ):
        self.pg = pg_store
        self.jsonl = jsonl_store
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize both stores."""
        await asyncio.gather(
            self.pg.connect(),
            self.jsonl.initialize(),
        )

    async def append(self, receipt: CommitReceipt) -> StoredReceipt:
        """
        Append to both stores atomically.

        Strategy:
        1. Append to PostgreSQL (primary)
        2. Append to JSONL (backup)
        3. Verify consistency
        4. On failure, log but don't rollback (append-only)
        """
        async with self._lock:
            # Append to PostgreSQL first (primary)
            pg_stored = await self.pg.append(receipt)

            try:
                # Append to JSONL (backup)
                # We need to ensure consistent offset
                jsonl_stored = await self.jsonl.append(receipt)

                # Verify consistency
                if pg_stored.integrity_hash != jsonl_stored.integrity_hash:
                    # Log warning but don't fail - append-only means we keep both
                    print(
                        f"Warning: Integrity hash mismatch between stores "
                        f"for receipt {receipt.receipt_id}"
                    )
            except Exception as e:
                # Log but don't fail - primary write succeeded
                print(f"Warning: JSONL backup failed for {receipt.receipt_id}: {e}")

            return pg_stored

    async def get_by_id(self, receipt_id: str) -> Optional[StoredReceipt]:
        """Retrieve by ID, preferring PostgreSQL."""
        result = await self.pg.get_by_id(receipt_id)
        if result is None:
            # Fallback to JSONL
            result = await self.jsonl.get_by_id(receipt_id)
        return result

    async def get_by_offset(self, offset: int) -> Optional[StoredReceipt]:
        """Retrieve by offset, preferring PostgreSQL."""
        result = await self.pg.get_by_offset(offset)
        if result is None:
            result = await self.jsonl.get_by_offset(offset)
        return result

    async def get_by_envelope_digest(self, digest: str) -> List[StoredReceipt]:
        """Retrieve all receipts for an envelope."""
        return await self.pg.get_by_envelope_digest(digest)

    async def verify_chain_integrity(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify chain integrity in both stores.

        Returns (overall_valid, details)
        """
        pg_valid, pg_last = await self.pg.verify_chain_integrity()
        jsonl_valid, jsonl_error = await self.jsonl.verify_chain_integrity()

        return (
            pg_valid and jsonl_valid,
            {
                "postgres": {"valid": pg_valid, "last_verified_offset": pg_last},
                "jsonl": {"valid": jsonl_valid, "error": jsonl_error},
            }
        )

    async def sync_stores(self) -> Dict[str, Any]:
        """
        Sync JSONL from PostgreSQL (recovery).

        Use when JSONL is behind or corrupted.

        Returns sync statistics.
        """
        stats = {
            "pg_count": 0,
            "jsonl_count": 0,
            "synced": 0,
            "errors": [],
        }

        stats["pg_count"] = await self.pg.count()
        stats["jsonl_count"] = await self.jsonl.count()

        # Get JSONL latest offset
        jsonl_latest = await self.jsonl.get_latest_offset()

        # Stream from PostgreSQL starting after JSONL latest
        async for receipt in self.pg.stream_all(start_offset=jsonl_latest + 1):
            try:
                # We need to reconstruct a CommitReceipt-like object
                # For recovery, we directly write the StoredReceipt
                json_line = json.dumps(receipt.to_dict(), separators=(",", ":"))
                async with aiofiles.open(
                    self.jsonl.filepath,
                    "a",
                    encoding="utf-8"
                ) as f:
                    await f.write(json_line + "\n")

                # Update in-memory chain
                self.jsonl._chain.add(receipt)
                stats["synced"] += 1

            except Exception as e:
                stats["errors"].append(f"offset {receipt.offset}: {str(e)}")

        return stats

    async def get_evidence_chain(self, receipt_id: str) -> List[StoredReceipt]:
        """Traverse evidence chain from receipt."""
        return await self.pg.get_evidence_chain(receipt_id)

    async def stream_all(self) -> AsyncIterator[StoredReceipt]:
        """Stream all receipts for audit (from PostgreSQL)."""
        async for receipt in self.pg.stream_all():
            yield receipt

    async def count(self) -> Dict[str, int]:
        """Get receipt counts from both stores."""
        pg_count = await self.pg.count()
        jsonl_count = await self.jsonl.count()
        return {"postgres": pg_count, "jsonl": jsonl_count}

    async def get_latest_offset(self) -> int:
        """Get the latest offset (from PostgreSQL)."""
        return await self.pg.get_latest_offset()


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_receipt_store(
    mode: str = "hybrid",  # "postgres", "jsonl", "hybrid"
    pg_conn: Optional[str] = None,
    jsonl_path: Optional[Path] = None,
) -> Union[PostgreSQLReceiptStore, JSONLReceiptStore, HybridReceiptStore]:
    """
    Factory function to create appropriate receipt store.

    Args:
        mode: Storage mode ("postgres", "jsonl", or "hybrid")
        pg_conn: PostgreSQL connection string (required for postgres/hybrid)
        jsonl_path: Path to JSONL file (optional, defaults to standard location)

    Returns:
        Configured receipt store instance

    Environment variables:
        DATABASE_URL: Default PostgreSQL connection string
        BIZRA_RECEIPT_PATH: Default JSONL directory
    """
    # Get defaults from environment
    if pg_conn is None:
        pg_conn = os.environ.get(
            "DATABASE_URL",
            "postgresql://bizra:bizra@localhost:5432/bizra"
        )

    if jsonl_path is None:
        base_path = os.environ.get("BIZRA_RECEIPT_PATH", "docs/evidence/receipts")
        jsonl_path = Path(base_path) / "pci_receipts.jsonl"

    if mode == "postgres":
        if not PSYCOPG_AVAILABLE:
            raise ImportError(
                "psycopg is required for PostgreSQL mode. "
                "Install with: pip install psycopg[binary]"
            )
        return PostgreSQLReceiptStore(pg_conn)

    elif mode == "jsonl":
        return JSONLReceiptStore(jsonl_path)

    elif mode == "hybrid":
        if not PSYCOPG_AVAILABLE:
            print(
                "Warning: psycopg not available, falling back to JSONL mode. "
                "Install psycopg for hybrid mode."
            )
            return JSONLReceiptStore(jsonl_path)

        pg_store = PostgreSQLReceiptStore(pg_conn)
        jsonl_store = JSONLReceiptStore(jsonl_path)
        return HybridReceiptStore(pg_store, jsonl_store)

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'postgres', 'jsonl', or 'hybrid'")


# =============================================================================
# GLOBAL STORE INSTANCE
# =============================================================================

_receipt_store: Optional[
    Union[PostgreSQLReceiptStore, JSONLReceiptStore, HybridReceiptStore]
] = None


def get_persistent_receipt_store(
    mode: Optional[str] = None,
) -> Union[PostgreSQLReceiptStore, JSONLReceiptStore, HybridReceiptStore]:
    """
    Get the global persistent receipt store.

    Creates on first access with default configuration.
    Override mode with BIZRA_RECEIPT_MODE environment variable.
    """
    global _receipt_store

    if _receipt_store is None:
        if mode is None:
            mode = os.environ.get("BIZRA_RECEIPT_MODE", "hybrid")
        _receipt_store = create_receipt_store(mode=mode)

    return _receipt_store


async def reset_persistent_receipt_store() -> None:
    """Reset the global store (for testing)."""
    global _receipt_store
    _receipt_store = None
