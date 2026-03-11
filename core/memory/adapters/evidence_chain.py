"""
Evidence-Aware Memory — Constitutional Audit Trail for AgentDB.

Every memory operation (store, search, forget, batch) emits a hash-chained
receipt into the Evidence Ledger. This makes the memory subsystem fully
compliant with DDAGI OS §7 (Evidence & Proof) and enables tamper-evident
audit trails for all AI agent memory operations.

Standing on Giants:
- Lamport (1978): Hash-chained event ordering
- Nakamoto (2008): Tamper-evident receipt chains
- Deming (1950): Measurement-driven quality (PDCA)
- Shannon (1948): SNR as information quality measure
- BIZRA Spine §7: Every action produces an ActionReceipt

Artifact: core/memory/adapters/evidence_chain.py
"""

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.memory.agent_db import AgentDB
from core.memory.types import MemoryRecord, QueryOptions, SearchResult
from core.proof_engine.canonical import canonical_bytes, hex_digest
from core.proof_engine.evidence_ledger import EvidenceLedger, LedgerEntry

logger = logging.getLogger(__name__)

# Receipt operation types
OP_STORE = "memory.store"
OP_STORE_RECORD = "memory.store_record"
OP_STORE_BATCH = "memory.store_batch"
OP_SEARCH = "memory.search"
OP_FORGET = "memory.forget"

# Default node ID when not specified
_DEFAULT_NODE_ID = "node0"


def _content_digest(content: str) -> str:
    """Compute BLAKE3-based content digest for receipt sealing."""
    return hex_digest(content.encode("utf-8"))


def _query_digest(options: QueryOptions) -> str:
    """Compute deterministic digest of a search query."""
    query_repr: Dict[str, Any] = {}
    if options.query_text:
        query_repr["query_text"] = options.query_text
    if options.source:
        query_repr["source"] = options.source
    if options.top_k:
        query_repr["top_k"] = options.top_k
    if options.min_score:
        query_repr["min_score"] = options.min_score
    if options.use_mmr:
        query_repr["use_mmr"] = True
        query_repr["mmr_lambda"] = options.mmr_lambda
    if options.metadata_filters:
        query_repr["metadata_filters"] = options.metadata_filters
    return hex_digest(canonical_bytes(query_repr))


def _receipt_id() -> str:
    """Generate a unique receipt ID."""
    return f"mem-{uuid.uuid4().hex[:12]}"


class EvidenceAwareMemory:
    """Constitutional memory wrapper — every operation produces a receipt.

    Wraps AgentDB with an EvidenceLedger that records hash-chained
    receipts for every store, search, and forget operation. The chain
    is tamper-evident: modifying any receipt invalidates all subsequent
    hashes, detectable via ``verify()``.

    Usage::

        from core.memory.agent_db import AgentDB
        from core.memory.adapters.evidence_chain import EvidenceAwareMemory
        from pathlib import Path

        db = AgentDB(data_dir=Path("/tmp/memory"))
        mem = EvidenceAwareMemory(db, ledger_dir=Path("/tmp/evidence"))

        # Store with receipt
        record_id, entry = mem.store("important fact", source="user")

        # Search with receipt
        results, entry = mem.search("important")

        # Verify chain integrity
        is_valid, errors = mem.verify()

    Standing on Giants:
    - Lamport (1978): Logical clocks and event ordering
    - Nakamoto (2008): Hash-linked evidence chains
    - BIZRA Spine §7: Every action produces ActionReceipt
    """

    def __init__(
        self,
        db: AgentDB,
        ledger_dir: Optional[Path] = None,
        node_id: str = _DEFAULT_NODE_ID,
        validate_receipts: bool = False,
    ):
        """Initialize evidence-aware memory.

        Args:
            db: AgentDB instance for memory operations.
            ledger_dir: Directory for the evidence ledger JSONL file.
                If None, uses db's data_dir / "evidence".
            node_id: Node identifier for receipt attribution.
            validate_receipts: Whether to run schema validation on receipts.
                Disabled by default for performance — enable for strict auditing.
        """
        self._db = db
        self._node_id = node_id

        # Resolve ledger directory
        if ledger_dir is None:
            ledger_dir = db._config.data_dir / "evidence"
        ledger_dir.mkdir(parents=True, exist_ok=True)

        self._ledger = EvidenceLedger(
            ledger_dir / "memory_evidence.jsonl",
            validate_on_append=validate_receipts,
        )

        # Operation counters for stats
        self._op_counts: Dict[str, int] = {
            OP_STORE: 0,
            OP_STORE_RECORD: 0,
            OP_STORE_BATCH: 0,
            OP_SEARCH: 0,
            OP_FORGET: 0,
        }
        self._total_latency_ms: float = 0.0

    def store(
        self,
        content: str,
        source: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
    ) -> Tuple[str, LedgerEntry]:
        """Store content with evidence receipt.

        Returns:
            (record_id, ledger_entry) — the stored record ID and the
            evidence receipt entry.
        """
        t0 = time.monotonic()
        record = self._db.store(
            content=content,
            source=source,
            metadata=metadata,
            importance=importance,
        )
        duration_ms = (time.monotonic() - t0) * 1000
        record_id = record.id

        entry = self._emit_receipt(
            operation=OP_STORE,
            status="accepted",
            seal_digest=_content_digest(content),
            details={
                "record_id": record_id,
                "source": source,
                "importance": importance,
                "content_length": len(content),
            },
            duration_ms=duration_ms,
        )

        self._op_counts[OP_STORE] += 1
        self._total_latency_ms += duration_ms
        return record_id, entry

    def store_record(self, record: MemoryRecord) -> Tuple[str, LedgerEntry]:
        """Store a MemoryRecord with evidence receipt.

        Returns:
            (record_id, ledger_entry)
        """
        t0 = time.monotonic()
        self._db.store_record(record)
        duration_ms = (time.monotonic() - t0) * 1000

        entry = self._emit_receipt(
            operation=OP_STORE_RECORD,
            status="accepted",
            seal_digest=_content_digest(record.content),
            details={
                "record_id": record.id,
                "source": record.source,
                "importance": record.importance,
                "content_length": len(record.content),
            },
            duration_ms=duration_ms,
        )

        self._op_counts[OP_STORE_RECORD] += 1
        self._total_latency_ms += duration_ms
        return record.id, entry

    def store_batch(self, records: List[MemoryRecord]) -> Tuple[int, LedgerEntry]:
        """Batch store with single evidence receipt.

        Returns:
            (count_stored, ledger_entry)
        """
        t0 = time.monotonic()
        count = self._db.store_batch(records)
        duration_ms = (time.monotonic() - t0) * 1000

        # Compute aggregate digest over all record contents
        batch_content = "|".join(r.content for r in records)
        batch_digest = _content_digest(batch_content)

        sources = list({r.source for r in records})

        entry = self._emit_receipt(
            operation=OP_STORE_BATCH,
            status="accepted",
            seal_digest=batch_digest,
            details={
                "records_submitted": len(records),
                "records_stored": count,
                "sources": sources,
                "total_content_length": sum(len(r.content) for r in records),
            },
            duration_ms=duration_ms,
        )

        self._op_counts[OP_STORE_BATCH] += 1
        self._total_latency_ms += duration_ms
        return count, entry

    def search(
        self,
        query_text: Optional[str] = None,
        source: Optional[str] = None,
        top_k: int = 10,
        min_score: float = 0.1,
        use_mmr: bool = False,
        mmr_lambda: float = 0.7,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[SearchResult], LedgerEntry]:
        """Search with evidence receipt.

        Returns:
            (results, ledger_entry)
        """
        options = QueryOptions(
            query_text=query_text,
            source=source,
            top_k=top_k,
            min_score=min_score,
            use_mmr=use_mmr,
            mmr_lambda=mmr_lambda,
            metadata_filters=metadata_filters,
        )

        t0 = time.monotonic()
        results = self._db.search(
            query=query_text,
            source=source,
            top_k=top_k,
            min_score=min_score,
        )
        duration_ms = (time.monotonic() - t0) * 1000

        entry = self._emit_receipt(
            operation=OP_SEARCH,
            status="accepted",
            seal_digest=_query_digest(options),
            details={
                "query_text": query_text,
                "source": source,
                "top_k": top_k,
                "result_count": len(results),
                "top_score": results[0].score if results else 0.0,
                "use_mmr": use_mmr,
            },
            duration_ms=duration_ms,
        )

        self._op_counts[OP_SEARCH] += 1
        self._total_latency_ms += duration_ms
        return results, entry

    def forget(self, record_id: str) -> Tuple[bool, LedgerEntry]:
        """Forget with evidence receipt (audit trail for deletion).

        Returns:
            (was_deleted, ledger_entry)
        """
        t0 = time.monotonic()
        deleted = self._db.forget(record_id)
        duration_ms = (time.monotonic() - t0) * 1000

        entry = self._emit_receipt(
            operation=OP_FORGET,
            status="accepted" if deleted else "rejected",
            seal_digest=hex_digest(record_id.encode("utf-8")),
            details={
                "record_id": record_id,
                "deleted": deleted,
            },
            duration_ms=duration_ms,
        )

        self._op_counts[OP_FORGET] += 1
        self._total_latency_ms += duration_ms
        return deleted, entry

    def find(
        self,
        source: Optional[str] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ) -> List[MemoryRecord]:
        """Find records by metadata (pass-through, no receipt for reads-only).

        find() is a lightweight metadata query — no receipt emitted to avoid
        noise in the evidence chain. Use search() for audited queries.
        """
        return self._db.find(source=source, limit=limit)

    def verify(self) -> Tuple[bool, List[str]]:
        """Verify the integrity of the memory evidence chain.

        Returns:
            (is_valid, errors) — errors is empty list on success.
        """
        return self._ledger.verify_chain()

    def stats(self) -> Dict[str, Any]:
        """Aggregate stats: memory + evidence chain."""
        db_stats = self._db.stats()

        avg_latency = 0.0
        total_ops = sum(self._op_counts.values())
        if total_ops > 0:
            avg_latency = self._total_latency_ms / total_ops

        return {
            **db_stats,
            "evidence": {
                "ledger_entries": self._ledger.sequence,
                "chain_valid": self._ledger.verify_chain()[0],
                "operations": dict(self._op_counts),
                "total_operations": total_ops,
                "avg_latency_ms": round(avg_latency, 2),
                "node_id": self._node_id,
            },
        }

    @property
    def db(self) -> AgentDB:
        """Access underlying AgentDB (for advanced operations)."""
        return self._db

    @property
    def ledger(self) -> EvidenceLedger:
        """Access underlying EvidenceLedger (for chain inspection)."""
        return self._ledger

    def _emit_receipt(
        self,
        operation: str,
        status: str,
        seal_digest: str,
        details: Dict[str, Any],
        duration_ms: float,
    ) -> LedgerEntry:
        """Emit a receipt into the evidence ledger.

        Receipt format is lightweight (no full schema validation) to keep
        memory operations fast. For strict auditing, initialize with
        validate_receipts=True.
        """
        receipt: Dict[str, Any] = {
            "receipt_id": _receipt_id(),
            "operation": operation,
            "status": status,
            "node_id": self._node_id,
            "seal": {
                "algorithm": "blake3",
                "digest": seal_digest,
            },
            "details": details,
            "metrics": {
                "duration_ms": round(duration_ms, 3),
            },
        }

        try:
            return self._ledger.append(receipt)
        except Exception as exc:
            # Evidence chain failure must NOT block memory operations
            # Log and return a sentinel entry
            logger.error("Evidence chain append failed for %s: %s", operation, exc)
            from core.proof_engine.evidence_ledger import GENESIS_HASH

            return LedgerEntry(
                sequence=-1,
                receipt=receipt,
                prev_hash=GENESIS_HASH,
                entry_hash="error",
                timestamp="error",
            )


__all__ = ["EvidenceAwareMemory"]
