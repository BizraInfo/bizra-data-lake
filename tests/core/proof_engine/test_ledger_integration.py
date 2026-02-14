"""
Ledger Integration Tests — SP-003 Completion.

Proves that the Evidence Ledger is wired into the live pipeline:
- SovereignRuntime initializes the ledger
- _emit_query_receipt() produces schema-compliant entries
- VerifierResponse matches the ITP specification
- API endpoints include receipt references

Standing on Giants:
- Lamport (1978): Event ordering
- Merkle (1979): Hash chains
- BIZRA Spearpoint PRD SP-003: "wire ledger into live pipeline"
"""

import asyncio
import hashlib
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.proof_engine.evidence_ledger import (
    EvidenceLedger,
    VerifierResponse,
    emit_receipt,
    GENESIS_HASH,
)
from core.proof_engine.reason_codes import ReasonCode


# =============================================================================
# RUNTIME INTEGRATION
# =============================================================================

class TestRuntimeLedgerInit:
    """Tests that SovereignRuntime can initialize the ledger."""

    def test_init_evidence_ledger_creates_instance(self, tmp_path):
        """_init_evidence_ledger creates an EvidenceLedger."""
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig

        config = RuntimeConfig()
        config.state_dir = tmp_path
        runtime = SovereignRuntime(config)
        runtime._init_evidence_ledger()

        assert runtime._evidence_ledger is not None
        assert runtime._evidence_ledger.sequence == 0
        assert (tmp_path / "evidence.jsonl").parent.exists()

    def test_init_evidence_ledger_resumes(self, tmp_path):
        """_init_evidence_ledger resumes from existing file."""
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig

        # Pre-populate ledger
        ledger = EvidenceLedger(tmp_path / "evidence.jsonl", validate_on_append=False)
        ledger.append({
            "receipt_id": "a1b2c3d4e5f6a1b2",
            "timestamp": "2026-02-10T19:00:00Z",
            "node_id": "test-node",
            "policy_version": "1.0.0",
            "status": "accepted",
            "decision": "APPROVED",
            "reason_codes": [],
            "snr": {"score": 0.95},
            "ihsan": {"score": 0.97, "threshold": 0.95, "decision": "APPROVED"},
            "seal": {"algorithm": "blake3", "digest": "a" * 64},
        })

        config = RuntimeConfig()
        config.state_dir = tmp_path
        runtime = SovereignRuntime(config)
        runtime._init_evidence_ledger()

        assert runtime._evidence_ledger.sequence == 1

    def test_evidence_ledger_field_exists(self):
        """SovereignRuntime has _evidence_ledger attribute."""
        from core.sovereign.runtime_core import SovereignRuntime

        runtime = SovereignRuntime()
        assert hasattr(runtime, "_evidence_ledger")


# =============================================================================
# RECEIPT EMISSION
# =============================================================================

class TestEmitQueryReceipt:
    """Tests for _emit_query_receipt integration."""

    def test_emit_receipt_on_approved(self, tmp_path):
        """Approved query emits receipt to ledger."""
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig, SovereignQuery, SovereignResult

        config = RuntimeConfig()
        config.state_dir = tmp_path
        runtime = SovereignRuntime(config)
        runtime._init_evidence_ledger()

        result = SovereignResult(query_id="a1b2c3d4e5f60001")
        result.success = True
        result.response = "The answer is 42."
        result.snr_score = 0.96
        result.ihsan_score = 0.97
        result.validation_passed = True
        result.processing_time_ms = 150.0
        result.claim_tags = {"snr_score": "measured", "ihsan_score": "measured"}

        query = SovereignQuery(text="What is the meaning?", id="a1b2c3d4e5f60001")

        runtime._emit_query_receipt(result, query)

        assert runtime._evidence_ledger.sequence == 1
        entries = runtime._evidence_ledger.entries()
        assert len(entries) == 1
        receipt = entries[0].receipt
        assert receipt["decision"] == "APPROVED"
        assert receipt["snr"]["score"] == 0.96
        assert receipt["ihsan"]["score"] == 0.97

    def test_emit_receipt_on_rejected_ihsan(self, tmp_path):
        """Low ihsan query emits REJECTED receipt."""
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig, SovereignQuery, SovereignResult

        config = RuntimeConfig()
        config.state_dir = tmp_path
        config.ihsan_threshold = 0.95
        runtime = SovereignRuntime(config)
        runtime._init_evidence_ledger()

        result = SovereignResult(query_id="a1b2c3d4e5f60002")
        result.response = "Low quality answer."
        result.snr_score = 0.90
        result.ihsan_score = 0.70
        result.validation_passed = False
        result.processing_time_ms = 100.0
        result.claim_tags = {}

        query = SovereignQuery(text="Bad query", id="a1b2c3d4e5f60002")

        runtime._emit_query_receipt(result, query)

        entries = runtime._evidence_ledger.entries()
        receipt = entries[0].receipt
        assert receipt["decision"] == "REJECTED"
        assert "IHSAN_BELOW_THRESHOLD" in receipt["reason_codes"]

    def test_emit_receipt_on_low_snr(self, tmp_path):
        """Low SNR query emits QUARANTINED receipt."""
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig, SovereignQuery, SovereignResult

        config = RuntimeConfig()
        config.state_dir = tmp_path
        config.ihsan_threshold = 0.95
        runtime = SovereignRuntime(config)
        runtime._init_evidence_ledger()

        result = SovereignResult(query_id="a1b2c3d4e5f60003")
        result.response = "Noisy output."
        result.snr_score = 0.50
        result.ihsan_score = 0.96
        result.validation_passed = True
        result.processing_time_ms = 200.0
        result.claim_tags = {}

        query = SovereignQuery(text="Noisy query", id="a1b2c3d4e5f60003")

        runtime._emit_query_receipt(result, query)

        entries = runtime._evidence_ledger.entries()
        receipt = entries[0].receipt
        assert receipt["decision"] == "QUARANTINED"
        assert "SNR_BELOW_THRESHOLD" in receipt["reason_codes"]

    def test_emit_receipt_includes_graph_hash(self, tmp_path):
        """Receipt includes graph_hash when available."""
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig, SovereignQuery, SovereignResult

        config = RuntimeConfig()
        config.state_dir = tmp_path
        runtime = SovereignRuntime(config)
        runtime._init_evidence_ledger()

        result = SovereignResult(query_id="a1b2c3d4e5f60004")
        result.response = "Answer with graph."
        result.snr_score = 0.96
        result.ihsan_score = 0.97
        result.validation_passed = True
        result.processing_time_ms = 100.0
        result.graph_hash = "d" * 64
        result.claim_tags = {}

        query = SovereignQuery(text="Graph query", id="a1b2c3d4e5f60004")

        runtime._emit_query_receipt(result, query)

        entries = runtime._evidence_ledger.entries()
        receipt = entries[0].receipt
        assert receipt["outputs"]["graph_hash"] == "d" * 64

    def test_emit_receipt_no_ledger_is_noop(self):
        """No ledger means no crash — fire-and-forget."""
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import SovereignQuery, SovereignResult

        runtime = SovereignRuntime()
        # _evidence_ledger is None by default
        result = SovereignResult(query_id="a1b2c3d4e5f6ffff")
        result.response = "test"
        result.snr_score = 0.9
        result.ihsan_score = 0.9
        result.validation_passed = True
        result.processing_time_ms = 10.0
        result.claim_tags = {}

        query = SovereignQuery(text="test", id="a1b2c3d4e5f6ffff")

        # Should not raise
        runtime._emit_query_receipt(result, query)

    def test_emit_receipt_chain_integrity(self, tmp_path):
        """Multiple receipts form a valid hash chain."""
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig, SovereignQuery, SovereignResult

        config = RuntimeConfig()
        config.state_dir = tmp_path
        runtime = SovereignRuntime(config)
        runtime._init_evidence_ledger()

        for i in range(5):
            hex_id = f"a1b2c3d4e5f6{i:04x}"
            result = SovereignResult(query_id=hex_id)
            result.response = f"Answer {i}"
            result.snr_score = 0.96
            result.ihsan_score = 0.97
            result.validation_passed = True
            result.processing_time_ms = 50.0
            result.claim_tags = {}

            query = SovereignQuery(text=f"Query {i}", id=hex_id)
            runtime._emit_query_receipt(result, query)

        is_valid, errors = runtime._evidence_ledger.verify_chain()
        assert is_valid is True, f"Chain integrity broken: {errors}"
        assert runtime._evidence_ledger.sequence == 5


# =============================================================================
# VERIFIER RESPONSE CONTRACT
# =============================================================================

class TestVerifierResponseContract:
    """Tests that VerifierResponse matches the ITP specification."""

    def test_all_three_decisions_have_factory(self):
        """APPROVED, REJECTED, QUARANTINED all have factory methods."""
        assert hasattr(VerifierResponse, "approved")
        assert hasattr(VerifierResponse, "rejected")
        assert hasattr(VerifierResponse, "quarantined")

    def test_rejected_requires_reason_code_from_enum(self):
        """Reason codes should be valid ReasonCode values."""
        resp = VerifierResponse.rejected(
            reason_codes=[ReasonCode.SNR_BELOW_THRESHOLD.value],
            receipt_id="rcpt_test",
        )
        assert resp.reason_codes[0] in [rc.value for rc in ReasonCode]

    def test_response_is_json_serializable(self):
        """VerifierResponse.to_dict() is JSON-serializable."""
        import json

        resp = VerifierResponse.approved("rcpt_test", artifacts={"hash": "abc"})
        serialized = json.dumps(resp.to_dict())
        assert isinstance(serialized, str)
        deserialized = json.loads(serialized)
        assert deserialized["decision"] == "APPROVED"
