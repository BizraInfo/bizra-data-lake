"""
Tests for Evidence-Aware Memory — Constitutional Audit Trail.

Validates that every memory operation (store, search, forget, batch)
produces a hash-chained receipt in the Evidence Ledger, and that
the chain integrity can be verified.

Standing on Giants:
- Lamport (1978): Hash chain verification
- Nakamoto (2008): Tamper-evident receipts
- Deming (1950): PDCA quality cycle
"""

from pathlib import Path
from typing import List

import pytest

from core.memory.agent_db import AgentDB
from core.memory.config import MemoryConfig
from core.memory.types import MemoryRecord


@pytest.fixture
def evidence_mem(tmp_path: Path):
    """Create an EvidenceAwareMemory instance for testing."""
    from core.memory.adapters.evidence_chain import EvidenceAwareMemory

    config = MemoryConfig(data_dir=tmp_path / "db", auto_embed=False)
    db = AgentDB(config)
    db.initialize()
    ledger_dir = tmp_path / "evidence"
    mem = EvidenceAwareMemory(db, ledger_dir=ledger_dir, node_id="test-node")
    return mem


class TestStoreReceipts:
    """Store operations produce correct receipts."""

    def test_store_returns_record_id_and_entry(self, evidence_mem):
        record_id, entry = evidence_mem.store("Test content", source="unit_test")
        assert isinstance(record_id, str)
        assert len(record_id) > 0
        assert entry.sequence == 1
        assert len(entry.entry_hash) > 0

    def test_store_receipt_contains_operation(self, evidence_mem):
        _, entry = evidence_mem.store("Hello", source="test")
        assert entry.receipt["operation"] == "memory.store"
        assert entry.receipt["status"] == "accepted"

    def test_store_receipt_contains_details(self, evidence_mem):
        _, entry = evidence_mem.store("Important fact", source="user", importance=0.9)
        details = entry.receipt["details"]
        assert details["source"] == "user"
        assert details["importance"] == 0.9
        assert details["content_length"] == len("Important fact")
        assert "record_id" in details

    def test_store_receipt_has_seal_digest(self, evidence_mem):
        _, entry = evidence_mem.store("Sealed content", source="test")
        seal = entry.receipt["seal"]
        assert seal["algorithm"] == "blake3"
        assert len(seal["digest"]) > 0

    def test_store_receipt_has_metrics(self, evidence_mem):
        _, entry = evidence_mem.store("Timed content", source="test")
        metrics = entry.receipt["metrics"]
        assert "duration_ms" in metrics
        assert metrics["duration_ms"] >= 0.0

    def test_store_receipt_has_node_id(self, evidence_mem):
        _, entry = evidence_mem.store("Node content", source="test")
        assert entry.receipt["node_id"] == "test-node"

    def test_store_record_produces_receipt(self, evidence_mem):
        record = MemoryRecord(
            id="test-rec-1",
            content="Record content",
            source="record_test",
            importance=0.7,
        )
        record_id, entry = evidence_mem.store_record(record)
        assert entry.receipt["operation"] == "memory.store_record"
        assert entry.receipt["details"]["source"] == "record_test"

    def test_sequential_stores_chain_hashes(self, evidence_mem):
        """Each receipt links to the previous via prev_hash."""
        _, e1 = evidence_mem.store("First", source="test")
        _, e2 = evidence_mem.store("Second", source="test")
        _, e3 = evidence_mem.store("Third", source="test")

        assert e1.sequence == 1
        assert e2.sequence == 2
        assert e3.sequence == 3
        # Hash chain: each entry's prev_hash == previous entry's entry_hash
        assert e2.prev_hash == e1.entry_hash
        assert e3.prev_hash == e2.entry_hash

    def test_store_content_actually_persists(self, evidence_mem):
        """The underlying AgentDB actually stores the data."""
        record_id, _ = evidence_mem.store("Persistent data", source="persist_test")
        results = evidence_mem.db.search(query="Persistent data", top_k=5)
        assert len(results) >= 1
        assert any(r.record.content == "Persistent data" for r in results)


class TestBatchReceipts:
    """Batch store produces aggregate receipt."""

    def test_batch_store_returns_count_and_entry(self, evidence_mem):
        records = [
            MemoryRecord(
                id=f"batch-{i}",
                content=f"Batch item {i}",
                source="batch_test",
            )
            for i in range(5)
        ]
        count, entry = evidence_mem.store_batch(records)
        assert count == 5
        assert entry.receipt["operation"] == "memory.store_batch"

    def test_batch_receipt_contains_aggregate_details(self, evidence_mem):
        records = [
            MemoryRecord(
                id=f"agg-{i}",
                content=f"Aggregate content {i}",
                source="agg_test",
            )
            for i in range(3)
        ]
        count, entry = evidence_mem.store_batch(records)
        details = entry.receipt["details"]
        assert details["records_submitted"] == 3
        assert details["records_stored"] == 3
        assert "agg_test" in details["sources"]
        assert details["total_content_length"] > 0

    def test_batch_receipt_has_aggregate_seal(self, evidence_mem):
        records = [
            MemoryRecord(id="s1", content="A", source="test"),
            MemoryRecord(id="s2", content="B", source="test"),
        ]
        _, entry = evidence_mem.store_batch(records)
        assert entry.receipt["seal"]["algorithm"] == "blake3"
        assert len(entry.receipt["seal"]["digest"]) > 0


class TestSearchReceipts:
    """Search operations produce correct receipts."""

    def test_search_returns_results_and_entry(self, evidence_mem):
        evidence_mem.store("Searchable fact about physics", source="science")
        results, entry = evidence_mem.search(query_text="physics")
        assert isinstance(results, list)
        assert entry.receipt["operation"] == "memory.search"

    def test_search_receipt_contains_query_details(self, evidence_mem):
        evidence_mem.store("AI knowledge base entry", source="ai")
        _, entry = evidence_mem.search(query_text="AI knowledge", source="ai", top_k=5)
        details = entry.receipt["details"]
        assert details["query_text"] == "AI knowledge"
        assert details["source"] == "ai"
        assert details["top_k"] == 5
        assert "result_count" in details

    def test_search_receipt_has_query_digest(self, evidence_mem):
        _, entry = evidence_mem.search(query_text="test query")
        seal = entry.receipt["seal"]
        assert seal["algorithm"] == "blake3"
        assert len(seal["digest"]) > 0

    def test_search_receipt_includes_top_score(self, evidence_mem):
        evidence_mem.store("Exact match content", source="test")
        results, entry = evidence_mem.search(query_text="Exact match content")
        details = entry.receipt["details"]
        if results:
            assert details["top_score"] > 0.0
        else:
            assert details["top_score"] == 0.0

    def test_empty_search_produces_receipt(self, evidence_mem):
        """Even empty results produce a receipt for auditability."""
        results, entry = evidence_mem.search(query_text="nonexistent xyz")
        assert entry.receipt["operation"] == "memory.search"
        assert entry.receipt["details"]["result_count"] == 0


class TestForgetReceipts:
    """Forget operations produce audit trail receipts."""

    def test_forget_existing_record(self, evidence_mem):
        record_id, _ = evidence_mem.store("To be forgotten", source="test")
        deleted, entry = evidence_mem.forget(record_id)
        assert deleted is True
        assert entry.receipt["operation"] == "memory.forget"
        assert entry.receipt["status"] == "accepted"
        assert entry.receipt["details"]["deleted"] is True

    def test_forget_nonexistent_record(self, evidence_mem):
        deleted, entry = evidence_mem.forget("nonexistent-id")
        # AgentDB soft-delete always succeeds (marks as deleted)
        assert entry.receipt["operation"] == "memory.forget"
        assert entry.receipt["details"]["record_id"] == "nonexistent-id"

    def test_forget_receipt_has_record_id_digest(self, evidence_mem):
        record_id, _ = evidence_mem.store("Delete me", source="test")
        _, entry = evidence_mem.forget(record_id)
        assert entry.receipt["seal"]["algorithm"] == "blake3"
        assert len(entry.receipt["seal"]["digest"]) > 0


class TestChainIntegrity:
    """Evidence chain is tamper-evident and verifiable."""

    def test_empty_chain_is_valid(self, evidence_mem):
        is_valid, errors = evidence_mem.verify()
        assert is_valid is True
        assert errors == []

    def test_chain_after_operations_is_valid(self, evidence_mem):
        evidence_mem.store("Item 1", source="test")
        evidence_mem.store("Item 2", source="test")
        evidence_mem.search(query_text="Item")
        evidence_mem.forget("nonexistent")

        is_valid, errors = evidence_mem.verify()
        assert is_valid is True
        assert errors == []

    def test_chain_sequence_is_monotonic(self, evidence_mem):
        entries: List = []
        for i in range(5):
            _, entry = evidence_mem.store(f"Monotonic {i}", source="test")
            entries.append(entry)

        for i in range(1, len(entries)):
            assert entries[i].sequence == entries[i - 1].sequence + 1

    def test_chain_hashes_link(self, evidence_mem):
        _, e1 = evidence_mem.store("Chain A", source="test")
        _, e2 = evidence_mem.search(query_text="Chain")
        _, e3 = evidence_mem.store("Chain B", source="test")

        assert e2.prev_hash == e1.entry_hash
        assert e3.prev_hash == e2.entry_hash

    def test_mixed_operations_all_receipted(self, evidence_mem):
        """Mixed store/search/forget all appear in the chain."""
        evidence_mem.store("Alpha", source="test")
        evidence_mem.store("Beta", source="test")
        evidence_mem.search(query_text="Alpha")
        records = [
            MemoryRecord(id="b1", content="Gamma", source="batch"),
        ]
        evidence_mem.store_batch(records)
        evidence_mem.forget("nonexistent")

        # 5 operations = 5 ledger entries
        assert evidence_mem.ledger.sequence == 5

        is_valid, errors = evidence_mem.verify()
        assert is_valid is True


class TestStats:
    """Stats aggregate memory + evidence metrics."""

    def test_stats_includes_evidence_section(self, evidence_mem):
        stats = evidence_mem.stats()
        assert "evidence" in stats

    def test_stats_tracks_operation_counts(self, evidence_mem):
        evidence_mem.store("Count me", source="test")
        evidence_mem.store("Count me too", source="test")
        evidence_mem.search(query_text="Count")

        stats = evidence_mem.stats()
        ev = stats["evidence"]
        assert ev["operations"]["memory.store"] == 2
        assert ev["operations"]["memory.search"] == 1
        assert ev["total_operations"] == 3
        assert ev["ledger_entries"] == 3

    def test_stats_tracks_avg_latency(self, evidence_mem):
        evidence_mem.store("Latency test", source="test")
        stats = evidence_mem.stats()
        assert stats["evidence"]["avg_latency_ms"] >= 0.0

    def test_stats_reports_chain_validity(self, evidence_mem):
        evidence_mem.store("Valid chain", source="test")
        stats = evidence_mem.stats()
        assert stats["evidence"]["chain_valid"] is True

    def test_stats_includes_node_id(self, evidence_mem):
        stats = evidence_mem.stats()
        assert stats["evidence"]["node_id"] == "test-node"

    def test_stats_includes_db_stats(self, evidence_mem):
        """Evidence stats are merged with underlying AgentDB stats."""
        evidence_mem.store("DB stat test", source="test")
        stats = evidence_mem.stats()
        assert "total_records" in stats


class TestFindPassthrough:
    """find() is a lightweight passthrough (no receipt)."""

    def test_find_returns_records(self, evidence_mem):
        evidence_mem.store("Findable A", source="findme")
        evidence_mem.store("Findable B", source="findme")
        records = evidence_mem.find(source="findme")
        assert len(records) >= 2

    def test_find_does_not_emit_receipt(self, evidence_mem):
        evidence_mem.store("No receipt find", source="quiet")
        evidence_mem.find(source="quiet")
        # Only the store should have a receipt, not the find
        assert evidence_mem.ledger.sequence == 1


class TestEdgeCases:
    """Edge cases and error resilience."""

    def test_default_ledger_dir(self, tmp_path: Path):
        """When no ledger_dir given, uses db's data_dir/evidence."""
        from core.memory.adapters.evidence_chain import EvidenceAwareMemory

        config = MemoryConfig(data_dir=tmp_path / "auto", auto_embed=False)
        db = AgentDB(config)
        db.initialize()
        mem = EvidenceAwareMemory(db)
        mem.store("Auto dir test", source="test")
        assert (tmp_path / "auto" / "evidence" / "memory_evidence.jsonl").exists()

    def test_receipt_ids_are_unique(self, evidence_mem):
        ids = set()
        for i in range(10):
            _, entry = evidence_mem.store(f"Unique {i}", source="test")
            ids.add(entry.receipt["receipt_id"])
        assert len(ids) == 10

    def test_seal_digest_differs_by_content(self, evidence_mem):
        _, e1 = evidence_mem.store("Content A", source="test")
        _, e2 = evidence_mem.store("Content B", source="test")
        assert e1.receipt["seal"]["digest"] != e2.receipt["seal"]["digest"]

    def test_query_digest_differs_by_query(self, evidence_mem):
        _, e1 = evidence_mem.search(query_text="alpha")
        _, e2 = evidence_mem.search(query_text="beta")
        assert e1.receipt["seal"]["digest"] != e2.receipt["seal"]["digest"]

    def test_large_batch_produces_single_receipt(self, evidence_mem):
        records = [
            MemoryRecord(
                id=f"large-{i}",
                content=f"Large batch item number {i} with details",
                source="large_batch",
            )
            for i in range(50)
        ]
        count, entry = evidence_mem.store_batch(records)
        assert count == 50
        assert entry.receipt["details"]["records_stored"] == 50
        # Only 1 receipt for the whole batch
        assert evidence_mem.ledger.sequence == 1

    def test_ledger_persists_across_reopen(self, tmp_path: Path):
        """Evidence chain survives process restart."""
        from core.memory.adapters.evidence_chain import EvidenceAwareMemory

        config = MemoryConfig(data_dir=tmp_path / "persist", auto_embed=False)
        ledger_dir = tmp_path / "evidence"

        # First session
        db1 = AgentDB(config)
        db1.initialize()
        mem1 = EvidenceAwareMemory(db1, ledger_dir=ledger_dir)
        mem1.store("Session 1 data", source="test")
        mem1.store("More session 1", source="test")
        seq_after_s1 = mem1.ledger.sequence

        # Second session (new objects, same dirs)
        db2 = AgentDB(config)
        db2.initialize()
        mem2 = EvidenceAwareMemory(db2, ledger_dir=ledger_dir)
        mem2.store("Session 2 data", source="test")

        assert mem2.ledger.sequence == seq_after_s1 + 1
        is_valid, errors = mem2.verify()
        assert is_valid is True
