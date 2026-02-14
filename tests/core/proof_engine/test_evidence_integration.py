"""
Evidence Ledger Integration Tests — Full lifecycle verification.

Exercises the complete evidence ledger cycle: creation, append, hash-chain
integrity, receipt emission, verifier response, schema validation, resume
from disk, concurrency safety, and tamper detection.

Standing on Giants:
- Lamport (1978): Logical clocks, event ordering
- Merkle (1979): Hash chains for tamper detection
- Shannon (1948): SNR as information quality metric
- BIZRA Spearpoint PRD SP-002: "every verification call emits a receipt"
"""

import copy
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict

import pytest

from core.proof_engine.evidence_ledger import (
    GENESIS_HASH,
    EvidenceLedger,
    LedgerEntry,
    VerifierResponse,
    _compute_entry_hash,
    emit_receipt,
)
from core.proof_engine.schema_validator import validate_receipt


# =============================================================================
# FIXTURES & HELPERS
# =============================================================================

@pytest.fixture
def ledger_file(tmp_path: Path) -> Path:
    """Temporary ledger file path (does not yet exist on disk)."""
    return tmp_path / "test_evidence.jsonl"


@pytest.fixture
def ledger(ledger_file: Path) -> EvidenceLedger:
    """Ledger with schema validation disabled (fast unit tests)."""
    return EvidenceLedger(ledger_file, validate_on_append=False)


@pytest.fixture
def validated_ledger(ledger_file: Path) -> EvidenceLedger:
    """Ledger with schema validation enabled."""
    return EvidenceLedger(ledger_file, validate_on_append=True)


def _make_receipt(
    receipt_id: str = "a1b2c3d4e5f6a1b2",
    decision: str = "APPROVED",
    status: str = "accepted",
    snr_score: float = 0.95,
    ihsan_score: float = 0.96,
    reason_codes: list | None = None,
) -> Dict[str, Any]:
    """Build a minimal schema-compliant receipt dict."""
    return {
        "receipt_id": receipt_id,
        "timestamp": "2026-02-11T00:00:00Z",
        "node_id": "node-integration-test",
        "policy_version": "1.0.0",
        "status": status,
        "decision": decision,
        "reason_codes": reason_codes or [],
        "snr": {"score": snr_score},
        "ihsan": {
            "score": ihsan_score,
            "threshold": 0.95,
            "decision": "APPROVED" if ihsan_score >= 0.95 else "REJECTED",
        },
        "seal": {"algorithm": "blake3", "digest": "a" * 64},
    }


# =============================================================================
# 1. TestLedgerLifecycle — Creation, append, file persistence, resume
# =============================================================================

class TestLedgerLifecycle:
    """Verify the full create-append-persist-resume lifecycle."""

    def test_new_ledger_empty(self, ledger: EvidenceLedger) -> None:
        """A fresh ledger reports sequence=0, last_hash=GENESIS_HASH, zero entries."""
        assert ledger.sequence == 0
        assert ledger.last_hash == GENESIS_HASH
        assert ledger.count() == 0
        assert ledger.entries() == []

    def test_append_increments_sequence(self, ledger: EvidenceLedger) -> None:
        """Each append call increments the sequence number by exactly one."""
        for expected_seq in range(1, 6):
            entry = ledger.append(_make_receipt(receipt_id=f"a1b2c3d4e5f6a{expected_seq:03d}"))
            assert entry.sequence == expected_seq
            assert ledger.sequence == expected_seq

    def test_append_creates_file(
        self, ledger: EvidenceLedger, ledger_file: Path
    ) -> None:
        """The first append materializes the JSONL file on disk."""
        assert not ledger_file.exists()
        ledger.append(_make_receipt())
        assert ledger_file.exists()
        assert ledger_file.stat().st_size > 0

    def test_multiple_appends_chain(self, ledger: EvidenceLedger) -> None:
        """Appending N entries yields a contiguous chain of length N."""
        entries = []
        for i in range(10):
            entries.append(
                ledger.append(_make_receipt(receipt_id=f"a1b2c3d4e5f6b{i:03d}"))
            )
        assert ledger.count() == 10
        assert ledger.sequence == 10
        # Verify sequential linkage
        for idx in range(1, len(entries)):
            assert entries[idx].prev_hash == entries[idx - 1].entry_hash

    def test_resume_from_file(self, ledger_file: Path) -> None:
        """Close a ledger, reopen it, and confirm sequence and hash are preserved."""
        # Phase 1: populate
        ledger_a = EvidenceLedger(ledger_file, validate_on_append=False)
        ledger_a.append(_make_receipt(receipt_id="a1b2c3d4e5f6c001"))
        ledger_a.append(_make_receipt(receipt_id="a1b2c3d4e5f6c002"))
        ledger_a.append(_make_receipt(receipt_id="a1b2c3d4e5f6c003"))
        saved_seq = ledger_a.sequence
        saved_hash = ledger_a.last_hash

        # Phase 2: reopen (simulates process restart)
        ledger_b = EvidenceLedger(ledger_file, validate_on_append=False)
        assert ledger_b.sequence == saved_seq
        assert ledger_b.last_hash == saved_hash

        # Phase 3: continue appending and verify chain integrity
        e4 = ledger_b.append(_make_receipt(receipt_id="a1b2c3d4e5f6c004"))
        assert e4.sequence == saved_seq + 1
        assert e4.prev_hash == saved_hash

        # Full chain should still verify
        is_valid, errors = ledger_b.verify_chain()
        assert is_valid is True, f"Chain broken after resume: {errors}"


# =============================================================================
# 2. TestHashChain — Cryptographic chain link integrity
# =============================================================================

class TestHashChain:
    """Verify the SHA-256 hash chain that makes tampering detectable."""

    def test_first_entry_prev_is_genesis(self, ledger: EvidenceLedger) -> None:
        """The first entry's prev_hash must be the GENESIS_HASH sentinel."""
        entry = ledger.append(_make_receipt(receipt_id="a1b2c3d4e5f6d001"))
        assert entry.prev_hash == GENESIS_HASH
        assert entry.prev_hash == "0" * 64

    def test_chain_links_correctly(self, ledger: EvidenceLedger) -> None:
        """Entry N's prev_hash equals entry N-1's entry_hash for all N > 1."""
        prev_entry = ledger.append(_make_receipt(receipt_id="a1b2c3d4e5f6e001"))
        for i in range(2, 8):
            cur_entry = ledger.append(
                _make_receipt(receipt_id=f"a1b2c3d4e5f6e{i:03d}")
            )
            assert cur_entry.prev_hash == prev_entry.entry_hash, (
                f"Chain break at seq {i}: prev_hash={cur_entry.prev_hash[:16]}... "
                f"!= expected={prev_entry.entry_hash[:16]}..."
            )
            prev_entry = cur_entry

    def test_verify_chain_valid(self, ledger: EvidenceLedger) -> None:
        """An untampered chain passes verify_chain with zero errors."""
        for i in range(5):
            ledger.append(_make_receipt(receipt_id=f"a1b2c3d4e5f6f{i:03d}"))
        is_valid, errors = ledger.verify_chain()
        assert is_valid is True
        assert errors == []

    def test_verify_chain_detects_tampering(
        self, ledger: EvidenceLedger, ledger_file: Path
    ) -> None:
        """Modifying a receipt field in the JSONL file causes verify_chain to fail."""
        for i in range(4):
            ledger.append(_make_receipt(receipt_id=f"a1b2c3d4e5f6g{i:03d}"))

        # Tamper: overwrite the receipt_id in line 2 (0-indexed)
        lines = ledger_file.read_text(encoding="utf-8").strip().split("\n")
        tampered = json.loads(lines[1])
        tampered["receipt"]["receipt_id"] = "deadbeefdeadbeef"
        lines[1] = json.dumps(tampered, separators=(",", ":"), sort_keys=True)
        ledger_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # Re-open and verify
        ledger_reopened = EvidenceLedger(ledger_file, validate_on_append=False)
        is_valid, errors = ledger_reopened.verify_chain()
        assert is_valid is False
        assert len(errors) >= 1
        assert any("entry_hash mismatch" in err for err in errors)

    def test_hash_deterministic(self) -> None:
        """Identical inputs to _compute_entry_hash always produce the same digest."""
        receipt = _make_receipt(receipt_id="a1b2c3d4e5f6h001")
        h1 = _compute_entry_hash(1, receipt, GENESIS_HASH)
        h2 = _compute_entry_hash(1, receipt, GENESIS_HASH)
        h3 = _compute_entry_hash(1, receipt, GENESIS_HASH)
        assert h1 == h2 == h3
        assert len(h1) == 64  # SHA-256 hex digest


# =============================================================================
# 3. TestReceiptEmission — emit_receipt() bridge function
# =============================================================================

class TestReceiptEmission:
    """Verify emit_receipt produces schema-compliant, correctly populated entries."""

    def test_emit_receipt_creates_entry(self, ledger: EvidenceLedger) -> None:
        """emit_receipt returns a LedgerEntry and increments ledger state."""
        entry = emit_receipt(
            ledger,
            receipt_id="a1b2c3d4e5f60001",
            node_id="node-integration-01",
            snr_score=0.92,
            ihsan_score=0.96,
            seal_digest="b" * 64,
        )
        assert isinstance(entry, LedgerEntry)
        assert entry.sequence == 1
        assert ledger.count() == 1
        assert entry.receipt["receipt_id"] == "a1b2c3d4e5f60001"

    def test_emit_receipt_accepted(self, ledger: EvidenceLedger) -> None:
        """An accepted emit produces decision=APPROVED with empty reason_codes."""
        entry = emit_receipt(
            ledger,
            receipt_id="a1b2c3d4e5f60002",
            node_id="node-integration-01",
            status="accepted",
            decision="APPROVED",
            snr_score=0.95,
            ihsan_score=0.97,
            ihsan_threshold=0.95,
            seal_digest="c" * 64,
        )
        r = entry.receipt
        assert r["status"] == "accepted"
        assert r["decision"] == "APPROVED"
        assert r["reason_codes"] == []
        assert r["ihsan"]["decision"] == "APPROVED"
        assert r["ihsan"]["score"] == 0.97
        assert r["ihsan"]["threshold"] == 0.95

    def test_emit_receipt_rejected(self, ledger: EvidenceLedger) -> None:
        """A rejected emit includes reason codes and ihsan REJECTED decision."""
        entry = emit_receipt(
            ledger,
            receipt_id="a1b2c3d4e5f60003",
            node_id="node-integration-01",
            status="rejected",
            decision="REJECTED",
            reason_codes=["SNR_BELOW_THRESHOLD", "IHSAN_BELOW_THRESHOLD"],
            snr_score=0.40,
            ihsan_score=0.50,
            ihsan_threshold=0.95,
            seal_digest="d" * 64,
        )
        r = entry.receipt
        assert r["decision"] == "REJECTED"
        assert "SNR_BELOW_THRESHOLD" in r["reason_codes"]
        assert "IHSAN_BELOW_THRESHOLD" in r["reason_codes"]
        assert r["ihsan"]["decision"] == "REJECTED"

    def test_emit_receipt_with_snr_trace(self, ledger: EvidenceLedger) -> None:
        """emit_receipt propagates SNR trace components into the receipt."""
        trace = {
            "signal_mass": 0.85,
            "noise_mass": 0.05,
            "signal_components": {"grounding": 0.9, "diversity": 0.8},
            "noise_components": {"filler": 0.03, "repetition": 0.02},
            "trace_id": "trace-abc-123",
        }
        entry = emit_receipt(
            ledger,
            receipt_id="a1b2c3d4e5f60004",
            node_id="node-integration-01",
            snr_score=0.94,
            ihsan_score=0.96,
            seal_digest="e" * 64,
            snr_trace=trace,
        )
        snr = entry.receipt["snr"]
        assert snr["score"] == 0.94
        assert snr["signal_mass"] == 0.85
        assert snr["noise_mass"] == 0.05
        assert snr["signal_components"]["grounding"] == 0.9
        assert snr["noise_components"]["filler"] == 0.03
        assert snr["trace_id"] == "trace-abc-123"

    def test_emit_receipt_with_graph_hash(self, ledger: EvidenceLedger) -> None:
        """emit_receipt places graph_hash under the outputs section."""
        graph_hash_val = "f" * 64
        entry = emit_receipt(
            ledger,
            receipt_id="a1b2c3d4e5f60005",
            node_id="node-integration-01",
            snr_score=0.95,
            ihsan_score=0.97,
            seal_digest="a" * 64,
            graph_hash=graph_hash_val,
        )
        assert "outputs" in entry.receipt
        assert entry.receipt["outputs"]["graph_hash"] == graph_hash_val


# =============================================================================
# 4. TestVerifierResponse — Uniform response envelope
# =============================================================================

class TestVerifierResponse:
    """Verify VerifierResponse factory methods and serialization contract."""

    def test_approved_response(self) -> None:
        """approved() produces a well-formed APPROVED VerifierResponse."""
        resp = VerifierResponse.approved(
            receipt_id="rcpt_integration_001",
            receipt_signature="deadbeef" * 16,
            artifacts={"chain_valid": True, "entries": 42},
        )
        assert resp.decision == "APPROVED"
        assert resp.reason_codes == []
        assert resp.receipt_id == "rcpt_integration_001"
        d = resp.to_dict()
        assert d["artifacts"]["chain_valid"] is True
        assert d["artifacts"]["entries"] == 42

    def test_rejected_response_requires_reasons(self) -> None:
        """rejected() with empty reason_codes raises ValueError."""
        with pytest.raises(ValueError, match="at least one reason code"):
            VerifierResponse.rejected(reason_codes=[])

    def test_quarantined_response(self) -> None:
        """quarantined() produces a well-formed QUARANTINED VerifierResponse."""
        resp = VerifierResponse.quarantined(
            reason_codes=["EVIDENCE_EXPIRED", "EVIDENCE_TAMPERED"],
            receipt_id="rcpt_integration_002",
        )
        assert resp.decision == "QUARANTINED"
        assert len(resp.reason_codes) == 2
        assert "EVIDENCE_EXPIRED" in resp.reason_codes
        assert "EVIDENCE_TAMPERED" in resp.reason_codes

    def test_to_dict_complete(self) -> None:
        """to_dict() returns the five canonical keys with correct values."""
        resp = VerifierResponse.rejected(
            reason_codes=["SIGNATURE_INVALID"],
            receipt_id="rcpt_integration_003",
            receipt_signature="abcd" * 32,
            artifacts={"detail": "bad signature bytes"},
        )
        d = resp.to_dict()
        expected_keys = {
            "decision",
            "reason_codes",
            "receipt_id",
            "receipt_signature",
            "artifacts",
        }
        assert set(d.keys()) == expected_keys
        assert d["decision"] == "REJECTED"
        assert d["reason_codes"] == ["SIGNATURE_INVALID"]
        assert d["receipt_id"] == "rcpt_integration_003"
        assert d["receipt_signature"] == "abcd" * 32
        assert d["artifacts"]["detail"] == "bad signature bytes"

    def test_rejected_empty_reasons_raises(self) -> None:
        """Both rejected() and quarantined() enforce non-empty reason_codes."""
        with pytest.raises(ValueError, match="at least one reason code"):
            VerifierResponse.rejected(reason_codes=[])
        with pytest.raises(ValueError, match="at least one reason code"):
            VerifierResponse.quarantined(reason_codes=[])


# =============================================================================
# 5. TestLedgerConcurrency — Thread safety and scale
# =============================================================================

class TestLedgerConcurrency:
    """Verify thread safety and correctness at scale."""

    def test_concurrent_appends_preserve_chain(self, ledger_file: Path) -> None:
        """Multiple threads appending concurrently produce a valid chain."""
        ledger = EvidenceLedger(ledger_file, validate_on_append=False)
        num_threads = 8
        appends_per_thread = 10
        errors_from_threads: list = []

        def _worker(thread_id: int) -> None:
            try:
                for i in range(appends_per_thread):
                    rid = f"a1b2c3d4{thread_id:04x}{i:04x}"
                    ledger.append(_make_receipt(receipt_id=rid))
            except Exception as exc:
                errors_from_threads.append(f"Thread {thread_id}: {exc}")

        threads = [
            threading.Thread(target=_worker, args=(tid,))
            for tid in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        # No thread raised an exception
        assert errors_from_threads == [], f"Thread errors: {errors_from_threads}"

        # Total count matches
        total_expected = num_threads * appends_per_thread
        assert ledger.count() == total_expected

        # Chain is valid despite concurrent writes
        is_valid, chain_errors = ledger.verify_chain()
        assert is_valid is True, f"Chain errors after concurrent append: {chain_errors}"

        # All entries present and sequential
        all_entries = ledger.entries()
        assert len(all_entries) == total_expected
        for idx, entry in enumerate(all_entries, start=1):
            assert entry.sequence == idx

    def test_large_ledger_verification(self, ledger_file: Path) -> None:
        """A ledger with 100 entries verifies correctly and performs acceptably."""
        ledger = EvidenceLedger(ledger_file, validate_on_append=False)
        entry_count = 100

        start_time = time.monotonic()
        for i in range(entry_count):
            ledger.append(_make_receipt(receipt_id=f"a1b2c3d4e5{i:06x}"))
        append_duration = time.monotonic() - start_time

        assert ledger.count() == entry_count

        start_time = time.monotonic()
        is_valid, errors = ledger.verify_chain()
        verify_duration = time.monotonic() - start_time

        assert is_valid is True, f"Chain errors: {errors}"
        assert errors == []

        # Sanity: both operations should complete in well under 10 seconds
        assert append_duration < 10.0, f"Append took {append_duration:.2f}s"
        assert verify_duration < 10.0, f"Verify took {verify_duration:.2f}s"


# =============================================================================
# 6. TestSchemaValidation — Receipt schema enforcement
# =============================================================================

class TestSchemaValidation:
    """Verify schema validation gates on the ledger's append path."""

    def test_valid_receipt_passes(self, validated_ledger: EvidenceLedger) -> None:
        """A fully schema-compliant receipt is accepted by the validated ledger."""
        receipt = _make_receipt()
        entry = validated_ledger.append(receipt)
        assert entry.sequence == 1
        assert entry.receipt["receipt_id"] == receipt["receipt_id"]

    def test_invalid_receipt_raises(self, validated_ledger: EvidenceLedger) -> None:
        """A receipt missing required fields is rejected with ValueError."""
        incomplete = {"receipt_id": "a1b2c3d4e5f60099", "timestamp": "2026-02-11T00:00:00Z"}
        with pytest.raises(ValueError, match="schema validation"):
            validated_ledger.append(incomplete)
        # Ledger state must not have advanced
        assert validated_ledger.sequence == 0
        assert validated_ledger.count() == 0

    def test_emit_receipt_always_valid(self, ledger_file: Path) -> None:
        """Receipts produced by emit_receipt always pass schema validation."""
        # Use a validated ledger to prove emit_receipt builds compliant receipts
        val_ledger = EvidenceLedger(ledger_file, validate_on_append=True)
        entry = emit_receipt(
            val_ledger,
            receipt_id="a1b2c3d4e5f6ff01",
            node_id="node-schema-test",
            policy_version="1.0.0",
            status="accepted",
            decision="APPROVED",
            snr_score=0.95,
            ihsan_score=0.97,
            ihsan_threshold=0.95,
            seal_digest="b" * 64,
        )
        assert entry.sequence == 1

        # Double-check: independently validate the receipt dict
        is_valid, schema_errors = validate_receipt(entry.receipt)
        assert is_valid is True, f"Schema errors: {schema_errors}"


# =============================================================================
# 7. BONUS — Edge cases and roundtrip fidelity
# =============================================================================

class TestEdgeCases:
    """Additional edge-case and roundtrip tests for robustness."""

    def test_entry_jsonl_roundtrip(self) -> None:
        """LedgerEntry survives to_jsonl -> from_jsonl without data loss."""
        original = LedgerEntry(
            sequence=99,
            receipt=_make_receipt(receipt_id="a1b2c3d4e5f6ff99"),
            prev_hash="ab" * 32,
            entry_hash="cd" * 32,
            timestamp="2026-02-11T12:00:00+00:00",
        )
        serialized = original.to_jsonl()
        restored = LedgerEntry.from_jsonl(serialized)

        assert restored.sequence == original.sequence
        assert restored.receipt == original.receipt
        assert restored.prev_hash == original.prev_hash
        assert restored.entry_hash == original.entry_hash
        assert restored.timestamp == original.timestamp

    def test_jsonl_is_single_line(self, ledger: EvidenceLedger) -> None:
        """Each serialized entry is exactly one line (no embedded newlines)."""
        entry = ledger.append(_make_receipt())
        line = entry.to_jsonl()
        assert "\n" not in line
        assert "\r" not in line

    def test_hash_differs_for_different_receipts(self) -> None:
        """Different receipt contents produce different entry hashes."""
        r1 = _make_receipt(receipt_id="a1b2c3d4e5f6aa01")
        r2 = _make_receipt(receipt_id="a1b2c3d4e5f6aa02")
        h1 = _compute_entry_hash(1, r1, GENESIS_HASH)
        h2 = _compute_entry_hash(1, r2, GENESIS_HASH)
        assert h1 != h2

    def test_hash_differs_for_different_prev_hash(self) -> None:
        """Same receipt but different prev_hash produces different entry hash."""
        receipt = _make_receipt()
        h1 = _compute_entry_hash(1, receipt, GENESIS_HASH)
        h2 = _compute_entry_hash(1, receipt, "f" * 64)
        assert h1 != h2

    def test_deleted_middle_entry_detected(
        self, ledger: EvidenceLedger, ledger_file: Path
    ) -> None:
        """Deleting a middle entry breaks both sequence and chain link checks."""
        for i in range(5):
            ledger.append(_make_receipt(receipt_id=f"a1b2c3d4e5f6bb{i:02d}"))

        # Remove the third line (index 2)
        lines = ledger_file.read_text(encoding="utf-8").strip().split("\n")
        del lines[2]
        ledger_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        reopened = EvidenceLedger(ledger_file, validate_on_append=False)
        is_valid, errors = reopened.verify_chain()
        assert is_valid is False
        assert len(errors) >= 1

    def test_emit_receipt_with_all_optional_fields(
        self, ledger: EvidenceLedger
    ) -> None:
        """emit_receipt with every optional parameter populates the receipt fully."""
        entry = emit_receipt(
            ledger,
            receipt_id="a1b2c3d4e5f6cc01",
            node_id="node-full-test",
            policy_version="2.1.0",
            status="accepted",
            decision="APPROVED",
            reason_codes=[],
            snr_score=0.98,
            ihsan_score=0.99,
            ihsan_threshold=0.95,
            seal_digest="ab" * 32,
            seal_algorithm="blake3",
            query_digest="cd" * 32,
            policy_digest="ef" * 32,
            payload_digest="01" * 32,
            graph_hash="23" * 32,
            gate_passed="ihsan_gate",
            duration_ms=123.45,
            claim_tags={"measured": 10, "design": 2, "implemented": 5, "target": 1},
            snr_trace={
                "signal_mass": 0.90,
                "noise_mass": 0.02,
                "signal_components": {"grounding": 0.95},
                "noise_components": {"filler": 0.01},
                "claim_tags": {"measured": 10},
                "trace_id": "trace-full-001",
            },
        )
        r = entry.receipt
        assert r["policy_version"] == "2.1.0"
        assert r["gate_passed"] == "ihsan_gate"
        assert r["metrics"]["duration_ms"] == 123.45
        assert r["claim_tags_summary"]["measured"] == 10
        assert r["inputs"]["query_digest"] == "cd" * 32
        assert r["inputs"]["policy_digest"] == "ef" * 32
        assert r["outputs"]["payload_digest"] == "01" * 32
        assert r["outputs"]["graph_hash"] == "23" * 32
        assert r["snr"]["trace_id"] == "trace-full-001"
        assert r["snr"]["claim_tags"]["measured"] == 10
