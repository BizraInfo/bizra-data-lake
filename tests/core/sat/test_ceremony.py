"""
Tests for the Genesis-100 Ceremony.

Standing on Giants:
- Nakamoto (2008): Genesis block ceremony
"""

import json

import pytest

from core.sat.ceremony import GenesisReceipt, genesis_100_ceremony
from core.sat.gate_result import CheckResult, CheckStatus, GateResult


class TestGenesisReceipt:
    def test_default_ceremony_name(self):
        r = GenesisReceipt()
        assert r.ceremony == "GENESIS_100"

    def test_to_dict(self):
        r = GenesisReceipt(
            timestamp="2026-03-08T00:00:00Z",
            all_passed=True,
            total_checks=68,
            passed_checks=68,
        )
        d = r.to_dict()
        assert d["ceremony"] == "GENESIS_100"
        assert d["all_passed"] is True
        assert d["total_checks"] == 68

    def test_sign(self):
        pytest.importorskip("nacl")
        from nacl.signing import SigningKey

        r = GenesisReceipt(
            timestamp="2026-03-08T00:00:00Z",
            all_passed=True,
            total_checks=68,
            passed_checks=68,
        )
        signer = SigningKey.generate()
        r.sign(signer)
        assert r.hash is not None
        assert r.signature is not None
        assert len(r.hash) == 128  # BLAKE2b hex
        assert len(r.signature) > 0

    def test_to_dict_serializable(self):
        r = GenesisReceipt(
            timestamp="2026-03-08T00:00:00Z",
            all_passed=False,
            total_checks=10,
            passed_checks=7,
            failed_checks=[("Sentinel", "tests_pass"), ("Ledger", "bloom_soulbound")],
        )
        d = r.to_dict()
        # Must be JSON-serializable
        serialized = json.dumps(d)
        assert "GENESIS_100" in serialized


@pytest.mark.slow
class TestGenesisCeremony:
    """Full ceremony integration tests — marked slow because they run real checks."""

    def test_ceremony_returns_5_agents(self):
        passed, receipt = genesis_100_ceremony(
            skip_manual=True,
            skip_slow=True,
            sign=False,
            store=False,
        )
        assert receipt.ceremony == "GENESIS_100"
        assert len(receipt.agents) == 5

    def test_ceremony_has_timestamp(self):
        _, receipt = genesis_100_ceremony(
            skip_manual=True,
            skip_slow=True,
            sign=False,
            store=False,
        )
        assert receipt.timestamp != ""
        assert "T" in receipt.timestamp

    def test_ceremony_total_checks_minimum(self):
        _, receipt = genesis_100_ceremony(
            skip_manual=True,
            skip_slow=True,
            sign=False,
            store=False,
        )
        # With skips we should still have a good number of checks
        assert receipt.total_checks >= 40

    def test_ceremony_receipt_signed(self):
        pytest.importorskip("nacl")
        _, receipt = genesis_100_ceremony(
            skip_manual=True,
            skip_slow=True,
            store=False,
        )
        assert receipt.hash is not None
        assert receipt.signature is not None
