"""Tests for BIZRA Evidence Receipt chain."""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evidence_receipt import EvidenceLedger, EvidenceReceipt


@pytest.fixture
def ledger(tmp_path):
    return EvidenceLedger(tmp_path / "test_ledger.jsonl")


def _sample_receipt_args(**overrides):
    base = dict(
        mission_id="test-mission-001",
        ihsan_tensor={
            "moral_clarity": 0.92,
            "epistemic_humility": 0.88,
            "structural_integrity": 0.91,
            "verifiability": 0.90,
            "intent_alignment": 0.93,
            "resilience": 0.87,
        },
        ihsan_composite=0.902,
        gate_results={
            "alpha_4": True,
            "alpha_7": True,
            "alpha_8": True,
            "alpha_9": True,
            "alpha_10": True,
        },
        snr_normalized=0.85,
        tier="bloom",
    )
    base.update(overrides)
    return base


class TestEvidenceChain:
    """Evidence chain must be append-only and tamper-evident."""

    def test_first_receipt_links_to_genesis(self, ledger):
        receipt = ledger.append(**_sample_receipt_args())
        assert receipt.previous_hash == "0" * 64

    def test_second_receipt_links_to_first(self, ledger):
        r1 = ledger.append(**_sample_receipt_args(mission_id="m1"))
        r2 = ledger.append(**_sample_receipt_args(mission_id="m2"))
        assert r2.previous_hash == r1.receipt_id

    def test_chain_of_five(self, ledger):
        receipts = []
        for i in range(5):
            r = ledger.append(**_sample_receipt_args(mission_id=f"m{i}"))
            receipts.append(r)

        # Verify each links to predecessor
        assert receipts[0].previous_hash == "0" * 64
        for i in range(1, 5):
            assert receipts[i].previous_hash == receipts[i - 1].receipt_id

    def test_receipt_hash_is_deterministic(self, ledger):
        r = ledger.append(**_sample_receipt_args())
        assert r.verify_hash()
        assert r.receipt_id == r.compute_hash()

    def test_full_chain_verification(self, ledger):
        for i in range(10):
            ledger.append(**_sample_receipt_args(mission_id=f"mission-{i}"))

        valid, count, errors = ledger.verify_chain()
        assert valid
        assert count == 10
        assert errors == []

    def test_tamper_detection(self, ledger):
        """Modifying a receipt must break chain verification."""
        for i in range(3):
            ledger.append(**_sample_receipt_args(mission_id=f"m{i}"))

        # Tamper: change a field in the middle receipt
        lines = ledger.path.read_text().splitlines()
        middle = json.loads(lines[1])
        middle["ihsan_composite"] = 0.999  # Tampered!
        lines[1] = json.dumps(middle, sort_keys=True)
        ledger.path.write_text("\n".join(lines) + "\n")

        valid, count, errors = ledger.verify_chain()
        assert not valid
        assert len(errors) > 0

    def test_count(self, ledger):
        assert ledger.count() == 0
        ledger.append(**_sample_receipt_args())
        assert ledger.count() == 1
        ledger.append(**_sample_receipt_args(mission_id="m2"))
        assert ledger.count() == 2

    def test_last_receipt(self, ledger):
        assert ledger.last_receipt() is None
        ledger.append(**_sample_receipt_args(mission_id="first"))
        ledger.append(**_sample_receipt_args(mission_id="last"))
        last = ledger.last_receipt()
        assert last is not None
        assert last.mission_id == "last"

    def test_empty_chain_is_valid(self, ledger):
        valid, count, errors = ledger.verify_chain()
        assert valid
        assert count == 0


class TestEvidenceReceipt:
    """Individual receipt properties."""

    def test_ihsan_tensor_is_dict_not_scalar(self, ledger):
        """Fixes the poi.proto scalar problem: tensor MUST be a dict."""
        r = ledger.append(**_sample_receipt_args())
        assert isinstance(r.ihsan_tensor, dict)
        assert len(r.ihsan_tensor) == 6

    def test_domain_separation(self, ledger):
        r = ledger.append(**_sample_receipt_args())
        assert r.domain == "bizra-evidence-v1"

    def test_constitution_hash_recorded(self, ledger):
        r = ledger.append(**_sample_receipt_args())
        assert r.constitution_hash != ""

    def test_agent_chain_default(self, ledger):
        r = ledger.append(**_sample_receipt_args())
        assert len(r.agent_chain) == 7
        assert r.agent_chain[0] == "Planner"
        assert r.agent_chain[-1] == "Integrator"

    def test_gate_results_all_present(self, ledger):
        r = ledger.append(**_sample_receipt_args())
        assert "alpha_4" in r.gate_results
        assert "alpha_7" in r.gate_results
        assert "alpha_8" in r.gate_results
        assert "alpha_9" in r.gate_results
        assert "alpha_10" in r.gate_results

    def test_timestamp_is_recent(self, ledger):
        import time

        before = time.time()
        r = ledger.append(**_sample_receipt_args())
        after = time.time()
        assert before <= r.timestamp_utc <= after
