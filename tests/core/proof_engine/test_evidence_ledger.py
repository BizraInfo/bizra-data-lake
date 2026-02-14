"""
Evidence Ledger Tests — SP-002 Completion.

Proves that the evidence ledger is append-only, hash-chained,
tamper-detectable, and emits schema-compliant receipts.

Standing on Giants:
- Lamport (1978): Event ordering
- Merkle (1979): Hash chains
- BIZRA Spearpoint PRD SP-002: "every verification call emits a receipt"
"""

import copy
import json
import os
import pytest
import tempfile
from pathlib import Path

from core.proof_engine.evidence_ledger import (
    EvidenceLedger,
    LedgerEntry,
    VerifierResponse,
    emit_receipt,
    GENESIS_HASH,
    _compute_entry_hash,
)
from core.proof_engine.reason_codes import ReasonCode
from core.pci.crypto import generate_keypair, verify_signature


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def ledger_path(tmp_path):
    """Temporary ledger file path."""
    return tmp_path / "test_ledger.jsonl"


@pytest.fixture
def ledger(ledger_path):
    """Evidence ledger instance (no schema validation for speed)."""
    return EvidenceLedger(ledger_path, validate_on_append=False)


@pytest.fixture
def validated_ledger(ledger_path):
    """Evidence ledger with schema validation enabled."""
    return EvidenceLedger(ledger_path, validate_on_append=True)


@pytest.fixture(autouse=True)
def _node_role_env(monkeypatch):
    monkeypatch.setenv("BIZRA_NODE_ROLE", "node")


def _minimal_receipt(receipt_id="a1b2c3d4e5f6a1b2", decision="APPROVED"):
    """Build a minimal schema-compliant receipt."""
    return {
        "receipt_id": receipt_id,
        "timestamp": "2026-02-10T19:00:00Z",
        "node_id": "node-test-01",
        "policy_version": "1.0.0",
        "status": "accepted",
        "decision": decision,
        "reason_codes": [],
        "snr": {"score": 0.95},
        "ihsan": {"score": 0.96, "threshold": 0.95, "decision": "APPROVED"},
        "seal": {"algorithm": "blake3", "digest": "a" * 64},
    }


# =============================================================================
# LEDGER BASICS
# =============================================================================

class TestLedgerBasics:
    """Tests for basic ledger operations."""

    def test_empty_ledger(self, ledger):
        """New ledger starts empty."""
        assert ledger.sequence == 0
        assert ledger.last_hash == GENESIS_HASH
        assert ledger.count() == 0

    def test_append_increments_sequence(self, ledger):
        """Appending increments sequence number."""
        ledger.append(_minimal_receipt("r1"))
        assert ledger.sequence == 1
        ledger.append(_minimal_receipt("r2"))
        assert ledger.sequence == 2

    def test_append_returns_entry(self, ledger):
        """Append returns a LedgerEntry."""
        entry = ledger.append(_minimal_receipt("r1"))
        assert isinstance(entry, LedgerEntry)
        assert entry.sequence == 1
        assert entry.prev_hash == GENESIS_HASH
        assert len(entry.entry_hash) == 64
        assert entry.receipt["receipt_id"] == "r1"

    def test_append_chains_hashes(self, ledger):
        """Each entry's prev_hash is the previous entry's hash."""
        e1 = ledger.append(_minimal_receipt("r1"))
        e2 = ledger.append(_minimal_receipt("r2"))
        e3 = ledger.append(_minimal_receipt("r3"))

        assert e1.prev_hash == GENESIS_HASH
        assert e2.prev_hash == e1.entry_hash
        assert e3.prev_hash == e2.entry_hash

    def test_entry_hash_deterministic(self):
        """Same inputs produce same hash."""
        receipt = _minimal_receipt("r1")
        h1 = _compute_entry_hash(1, receipt, GENESIS_HASH)
        h2 = _compute_entry_hash(1, receipt, GENESIS_HASH)
        assert h1 == h2

    def test_entry_hash_changes_with_sequence(self):
        """Different sequence numbers produce different hashes."""
        receipt = _minimal_receipt("r1")
        h1 = _compute_entry_hash(1, receipt, GENESIS_HASH)
        h2 = _compute_entry_hash(2, receipt, GENESIS_HASH)
        assert h1 != h2

    def test_count_tracks_entries(self, ledger):
        """count() returns number of entries."""
        for i in range(5):
            ledger.append(_minimal_receipt(f"r{i}"))
        assert ledger.count() == 5

    def test_entries_returns_all(self, ledger):
        """entries() reads all appended entries."""
        for i in range(3):
            ledger.append(_minimal_receipt(f"r{i}"))
        entries = ledger.entries()
        assert len(entries) == 3
        assert entries[0].sequence == 1
        assert entries[2].sequence == 3


# =============================================================================
# CHAIN INTEGRITY
# =============================================================================

class TestChainIntegrity:
    """Tests for hash-chain tamper detection."""

    def test_valid_chain_passes(self, ledger):
        """Untampered chain passes verification."""
        for i in range(5):
            ledger.append(_minimal_receipt(f"r{i}"))
        is_valid, errors = ledger.verify_chain()
        assert is_valid is True
        assert errors == []

    def test_empty_chain_passes(self, ledger):
        """Empty ledger passes verification."""
        is_valid, errors = ledger.verify_chain()
        assert is_valid is True

    def test_tampered_receipt_detected(self, ledger, ledger_path):
        """Modifying a receipt breaks the chain."""
        for i in range(3):
            ledger.append(_minimal_receipt(f"r{i}"))

        # Tamper: modify the second entry's receipt
        lines = ledger_path.read_text(encoding="utf-8").strip().split("\n")
        entry = json.loads(lines[1])
        entry["receipt"]["receipt_id"] = "TAMPERED"
        lines[1] = json.dumps(entry, separators=(",", ":"), sort_keys=True)
        ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # Re-open and verify
        ledger2 = EvidenceLedger(ledger_path, validate_on_append=False)
        is_valid, errors = ledger2.verify_chain()
        assert is_valid is False
        assert any("entry_hash mismatch" in e for e in errors)

    def test_broken_chain_link_detected(self, ledger, ledger_path):
        """Modifying prev_hash breaks the chain."""
        for i in range(3):
            ledger.append(_minimal_receipt(f"r{i}"))

        # Tamper: modify the third entry's prev_hash
        lines = ledger_path.read_text(encoding="utf-8").strip().split("\n")
        entry = json.loads(lines[2])
        entry["prev_hash"] = "f" * 64
        lines[2] = json.dumps(entry, separators=(",", ":"), sort_keys=True)
        ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        ledger2 = EvidenceLedger(ledger_path, validate_on_append=False)
        is_valid, errors = ledger2.verify_chain()
        assert is_valid is False
        assert any("prev_hash mismatch" in e for e in errors)

    def test_deleted_entry_detected(self, ledger, ledger_path):
        """Deleting an entry breaks sequence monotonicity."""
        for i in range(3):
            ledger.append(_minimal_receipt(f"r{i}"))

        # Tamper: delete the second entry
        lines = ledger_path.read_text(encoding="utf-8").strip().split("\n")
        del lines[1]
        ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        ledger2 = EvidenceLedger(ledger_path, validate_on_append=False)
        is_valid, errors = ledger2.verify_chain()
        assert is_valid is False


# =============================================================================
# RESUME FROM EXISTING LEDGER
# =============================================================================

class TestLedgerResume:
    """Tests for resuming from an existing ledger file."""

    def test_resume_continues_sequence(self, ledger_path):
        """Resuming from existing file continues the sequence."""
        ledger1 = EvidenceLedger(ledger_path, validate_on_append=False)
        ledger1.append(_minimal_receipt("r1"))
        ledger1.append(_minimal_receipt("r2"))
        last_hash = ledger1.last_hash

        # Re-open
        ledger2 = EvidenceLedger(ledger_path, validate_on_append=False)
        assert ledger2.sequence == 2
        assert ledger2.last_hash == last_hash

        # Append more
        e3 = ledger2.append(_minimal_receipt("r3"))
        assert e3.sequence == 3
        assert e3.prev_hash == last_hash

    def test_resume_chain_stays_valid(self, ledger_path):
        """Chain remains valid across resume."""
        ledger1 = EvidenceLedger(ledger_path, validate_on_append=False)
        for i in range(3):
            ledger1.append(_minimal_receipt(f"r{i}"))

        ledger2 = EvidenceLedger(ledger_path, validate_on_append=False)
        for i in range(3, 6):
            ledger2.append(_minimal_receipt(f"r{i}"))

        is_valid, errors = ledger2.verify_chain()
        assert is_valid is True
        assert ledger2.count() == 6


# =============================================================================
# SCHEMA VALIDATION ON APPEND
# =============================================================================

class TestSchemaValidation:
    """Tests for schema validation on append."""

    def test_valid_receipt_accepted(self, validated_ledger):
        """Schema-compliant receipt is accepted."""
        entry = validated_ledger.append(_minimal_receipt())
        assert entry.sequence == 1

    def test_invalid_receipt_rejected(self, validated_ledger):
        """Invalid receipt raises ValueError."""
        bad = {"not": "a receipt"}
        with pytest.raises(ValueError, match="schema validation"):
            validated_ledger.append(bad)

    def test_missing_required_field_rejected(self, validated_ledger):
        """Receipt missing required field is rejected."""
        receipt = _minimal_receipt()
        del receipt["seal"]
        with pytest.raises(ValueError, match="schema validation"):
            validated_ledger.append(receipt)


# =============================================================================
# RECEIPT EMISSION
# =============================================================================

class TestReceiptEmission:
    """Tests for the emit_receipt() bridge function."""

    def test_emit_basic_receipt(self, ledger):
        """emit_receipt creates a schema-compliant entry."""
        entry = emit_receipt(
            ledger,
            receipt_id="abc123def456abc1",
            node_id="node-genesis-01",
            snr_score=0.95,
            ihsan_score=0.97,
            seal_digest="a" * 64,
        )
        assert entry.sequence == 1
        assert entry.receipt["decision"] == "APPROVED"
        assert entry.receipt["snr"]["score"] == 0.95
        assert entry.receipt["ihsan"]["score"] == 0.97

    def test_emit_rejected_receipt(self, ledger):
        """emit_receipt with rejection includes reason codes."""
        entry = emit_receipt(
            ledger,
            receipt_id="abc123def456abc2",
            node_id="node-genesis-01",
            status="rejected",
            decision="REJECTED",
            reason_codes=["SNR_BELOW_THRESHOLD"],
            snr_score=0.5,
            ihsan_score=0.4,
            ihsan_threshold=0.95,
            seal_digest="b" * 64,
        )
        assert entry.receipt["decision"] == "REJECTED"
        assert "SNR_BELOW_THRESHOLD" in entry.receipt["reason_codes"]

    def test_emit_with_graph_hash(self, ledger):
        """emit_receipt includes graph_hash in outputs."""
        entry = emit_receipt(
            ledger,
            receipt_id="abc123def456abc3",
            node_id="node-genesis-01",
            snr_score=0.95,
            ihsan_score=0.97,
            seal_digest="c" * 64,
            graph_hash="d" * 64,
        )
        assert entry.receipt["outputs"]["graph_hash"] == "d" * 64

    def test_emit_with_claim_tags(self, ledger):
        """emit_receipt includes claim_tags_summary."""
        entry = emit_receipt(
            ledger,
            receipt_id="abc123def456abc4",
            node_id="node-genesis-01",
            snr_score=0.95,
            ihsan_score=0.97,
            seal_digest="e" * 64,
            claim_tags={"measured": 5, "design": 0, "implemented": 1, "target": 0},
        )
        assert entry.receipt["claim_tags_summary"]["measured"] == 5

    def test_emit_with_metrics(self, ledger):
        """emit_receipt includes duration metrics."""
        entry = emit_receipt(
            ledger,
            receipt_id="abc123def456abc5",
            node_id="node-genesis-01",
            snr_score=0.95,
            ihsan_score=0.97,
            seal_digest="f" * 64,
            duration_ms=42.5,
        )
        assert entry.receipt["metrics"]["duration_ms"] == 42.5

    def test_emit_ihsan_decision_auto(self, ledger):
        """emit_receipt auto-computes ihsan decision from score vs threshold."""
        # Above threshold
        entry = emit_receipt(
            ledger,
            receipt_id="abc123def456abc6",
            node_id="node-genesis-01",
            snr_score=0.95,
            ihsan_score=0.97,
            ihsan_threshold=0.95,
            seal_digest="a" * 64,
        )
        assert entry.receipt["ihsan"]["decision"] == "APPROVED"

        # Below threshold
        entry = emit_receipt(
            ledger,
            receipt_id="abc123def456abc7",
            node_id="node-genesis-01",
            snr_score=0.95,
            ihsan_score=0.90,
            ihsan_threshold=0.95,
            seal_digest="b" * 64,
        )
        assert entry.receipt["ihsan"]["decision"] == "REJECTED"

    def test_emit_receipt_attaches_ed25519_signature(
        self, validated_ledger
    ):
        """Configured signer key attaches schema-compliant Ed25519 signature."""
        private_key_hex, public_key_hex = generate_keypair()
        entry = emit_receipt(
            validated_ledger,
            receipt_id="abc123def456abc8",
            node_id="node-genesis-01",
            snr_score=0.95,
            ihsan_score=0.97,
            seal_digest="c" * 64,
            signer_private_key_hex=private_key_hex,
            signer_public_key_hex=public_key_hex,
        )

        sig = entry.receipt.get("signature", {})
        assert sig["algorithm"] == "ed25519"
        assert len(sig["value"]) == 128
        assert sig["public_key"] == public_key_hex
        assert verify_signature(
            entry.receipt["seal"]["digest"],
            sig["value"],
            sig["public_key"],
        )

    def test_emit_receipt_rejects_mismatched_public_key(self, ledger):
        """Mismatched signer keys fail-closed with ValueError."""
        private_key_hex, _ = generate_keypair()
        _, wrong_public_key_hex = generate_keypair()

        with pytest.raises(ValueError, match="does not match"):
            emit_receipt(
                ledger,
                receipt_id="abc123def456abc9",
                node_id="node-genesis-01",
                snr_score=0.95,
                ihsan_score=0.97,
                seal_digest="d" * 64,
                signer_private_key_hex=private_key_hex,
                signer_public_key_hex=wrong_public_key_hex,
            )

    def test_emit_critical_receipt_includes_origin_proof(self, ledger, tmp_path):
        """Critical receipts include origin + origin_digest in the signed body."""
        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        origin = {
            "designation": "node0",
            "genesis_node": True,
            "genesis_block": True,
            "block_id": "block0",
            "home_base_device": True,
            "node_id": "node0_fixture",
            "node_name": "Node0 Fixture",
            "authority_source": "genesis_files",
            "hash_validated": True,
        }
        entry = emit_receipt(
            ledger,
            receipt_id="abcd1234abcd1234",
            node_id="node0_fixture",
            snr_score=0.96,
            ihsan_score=0.97,
            seal_digest="f" * 64,
            critical_decision=True,
            node_role="node0",
            origin=origin,
            signer_private_key_hex="11" * 32,
            state_dir=state_dir,
        )
        assert entry.receipt["origin"]["designation"] == "node0"
        assert len(entry.receipt["origin_digest"]) == 64
        assert entry.receipt["signature"]["algorithm"] == "ed25519"

    def test_emit_critical_node0_requires_signature_key(self, ledger, tmp_path, monkeypatch):
        """Node0 critical receipts fail if signer key is unavailable."""
        monkeypatch.delenv("BIZRA_RECEIPT_PRIVATE_KEY_HEX", raising=False)
        with pytest.raises(RuntimeError, match="Unsigned critical receipt forbidden"):
            emit_receipt(
                ledger,
                receipt_id="abcd1234abcd1235",
                node_id="node0_fixture",
                snr_score=0.96,
                ihsan_score=0.97,
                seal_digest="e" * 64,
                critical_decision=True,
                node_role="node0",
                signer_private_key_hex="",
                state_dir=tmp_path / "state",
            )


# =============================================================================
# VERIFIER RESPONSE
# =============================================================================

class TestVerifierResponse:
    """Tests for the uniform verifier response shape."""

    def test_approved_response(self):
        """APPROVED response has correct shape."""
        resp = VerifierResponse.approved(
            receipt_id="rcpt_001",
            receipt_signature="sig_hex",
            artifacts={"genesis_valid": True},
        )
        d = resp.to_dict()
        assert d["decision"] == "APPROVED"
        assert d["reason_codes"] == []
        assert d["receipt_id"] == "rcpt_001"
        assert d["receipt_signature"] == "sig_hex"
        assert d["artifacts"]["genesis_valid"] is True

    def test_rejected_response(self):
        """REJECTED response requires reason codes."""
        resp = VerifierResponse.rejected(
            reason_codes=["SIGNATURE_INVALID", "GENESIS_MISMATCH"],
            receipt_id="rcpt_002",
        )
        d = resp.to_dict()
        assert d["decision"] == "REJECTED"
        assert len(d["reason_codes"]) == 2
        assert "SIGNATURE_INVALID" in d["reason_codes"]

    def test_rejected_without_reasons_raises(self):
        """REJECTED with empty reason_codes raises ValueError."""
        with pytest.raises(ValueError, match="at least one reason code"):
            VerifierResponse.rejected(reason_codes=[])

    def test_quarantined_response(self):
        """QUARANTINED response requires reason codes."""
        resp = VerifierResponse.quarantined(
            reason_codes=["EVIDENCE_EXPIRED"],
            receipt_id="rcpt_003",
        )
        d = resp.to_dict()
        assert d["decision"] == "QUARANTINED"
        assert "EVIDENCE_EXPIRED" in d["reason_codes"]

    def test_quarantined_without_reasons_raises(self):
        """QUARANTINED with empty reason_codes raises ValueError."""
        with pytest.raises(ValueError, match="at least one reason code"):
            VerifierResponse.quarantined(reason_codes=[])

    def test_response_shape_uniform(self):
        """All response types have exactly the same keys."""
        approved = VerifierResponse.approved("r1").to_dict()
        rejected = VerifierResponse.rejected(["CODE"], "r2").to_dict()
        quarantined = VerifierResponse.quarantined(["CODE"], "r3").to_dict()
        assert set(approved.keys()) == set(rejected.keys()) == set(quarantined.keys())

    def test_response_keys_match_spec(self):
        """Response keys match the ITP specification."""
        resp = VerifierResponse.approved("r1").to_dict()
        expected_keys = {"decision", "reason_codes", "receipt_id", "receipt_signature", "artifacts"}
        assert set(resp.keys()) == expected_keys


# =============================================================================
# JSONL SERIALIZATION ROUNDTRIP
# =============================================================================

class TestSerialization:
    """Tests for JSONL serialization roundtrip."""

    def test_entry_roundtrip(self):
        """LedgerEntry survives serialize → deserialize."""
        entry = LedgerEntry(
            sequence=42,
            receipt=_minimal_receipt("r42"),
            prev_hash="a" * 64,
            entry_hash="b" * 64,
            timestamp="2026-02-10T19:00:00Z",
        )
        line = entry.to_jsonl()
        restored = LedgerEntry.from_jsonl(line)
        assert restored.sequence == 42
        assert restored.receipt["receipt_id"] == "r42"
        assert restored.prev_hash == "a" * 64
        assert restored.entry_hash == "b" * 64

    def test_jsonl_single_line(self):
        """Each entry serializes to exactly one line."""
        entry = LedgerEntry(
            sequence=1,
            receipt=_minimal_receipt("r1"),
            prev_hash=GENESIS_HASH,
            entry_hash="c" * 64,
            timestamp="2026-02-10T19:00:00Z",
        )
        line = entry.to_jsonl()
        assert "\n" not in line
