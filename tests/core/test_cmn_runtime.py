"""
CMN Runtime Harness — Integration tests.

Tests that the CMN runtime boots, wires all 7 components,
and produces valid constitutional health reports.
"""

from __future__ import annotations

import json
from pathlib import Path


from core.sovereign.cmn_runtime import CMNRuntime


class TestCMNRuntimeBoot:
    """Test CMN runtime boot and component wiring."""

    def test_boot_all_components(self, tmp_path: Path) -> None:
        """Boot should initialize all 7 components."""
        cmn = CMNRuntime(data_dir=tmp_path, node_id="test-node")
        report = cmn.boot()

        assert report["sovereignty"] == "ok"
        assert report["membrane"] == "ok"
        assert report["zann_zero"] == "ok"
        assert report["invariant_checker"] == "ok"
        assert report["frozen_agents"] == "ok"
        assert report["irp_trust"] == "ok"
        assert cmn._booted is True

    def test_boot_freezes_constitutional_agents(self, tmp_path: Path) -> None:
        """P5-Ethicist and S2-Oracle must be frozen at boot."""
        cmn = CMNRuntime(data_dir=tmp_path)
        cmn.boot()

        assert cmn._frozen_registry.is_frozen("P5-Ethicist")
        assert cmn._frozen_registry.is_frozen("S2-Oracle")
        assert not cmn._frozen_registry.is_frozen("P1-Planner")

    def test_boot_registers_sat_validators(self, tmp_path: Path) -> None:
        """All 5 SAT agents registered in IRP trust model."""
        cmn = CMNRuntime(data_dir=tmp_path)
        cmn.boot()

        assert cmn._irp_model.narrator_count() == 5


class TestConstitutionalHealth:
    """Test the /v1/health/constitutional backing function."""

    def test_healthy_system(self, tmp_path: Path) -> None:
        """All invariants pass => status=constitutional, ihsan=1.0."""
        cmn = CMNRuntime(data_dir=tmp_path)
        cmn.boot()

        health = cmn.constitutional_health()
        assert health["status"] == "constitutional"
        assert health["ihsan_score"] == 1.0
        assert all(health["invariants"].values())
        assert health["chain_length"] == 1

    def test_chained_receipts(self, tmp_path: Path) -> None:
        """Consecutive checks produce different receipt hashes."""
        cmn = CMNRuntime(data_dir=tmp_path)
        cmn.boot()

        h1 = cmn.constitutional_health()
        h2 = cmn.constitutional_health()
        assert h1["receipt_hash"] != h2["receipt_hash"]
        assert h2["chain_length"] == 2

    def test_not_booted_returns_status(self, tmp_path: Path) -> None:
        """Before boot, health returns not_booted."""
        cmn = CMNRuntime(data_dir=tmp_path)
        health = cmn.constitutional_health()
        assert health["status"] == "not_booted"

    def test_riba_violation_detected(self, tmp_path: Path) -> None:
        """Float amounts in seed ledger => riba_zero violation."""
        ledger = tmp_path / "seed_ledger.jsonl"
        ledger.write_text(
            json.dumps({"tx_id": "bad", "amount": 1.5, "recipient": "n1"}) + "\n"
        )
        cmn = CMNRuntime(data_dir=tmp_path, seed_ledger_path=ledger)
        cmn.boot()

        health = cmn.constitutional_health()
        assert health["invariants"]["riba_zero"] is False
        assert health["status"] == "violation"
        assert health["ihsan_score"] < 1.0


class TestMissionVerification:
    """Test membrane property verification of mission results."""

    def test_valid_mission_passes(self, tmp_path: Path) -> None:
        """Mission with ihsan >= 0.95 and receipt => verified."""
        cmn = CMNRuntime(data_dir=tmp_path)
        cmn.boot()

        result = {"ihsan_score": 0.97, "evidence_receipt_id": "abc123"}
        verification = cmn.verify_mission_result(result)
        assert verification["verified"] is True

    def test_low_ihsan_fails(self, tmp_path: Path) -> None:
        """Mission with ihsan < 0.95 => not verified."""
        cmn = CMNRuntime(data_dir=tmp_path)
        cmn.boot()

        result = {"ihsan_score": 0.80, "evidence_receipt_id": "abc123"}
        verification = cmn.verify_mission_result(result)
        assert verification["verified"] is False
        assert verification["checks"]["constitutional_alignment"]["passed"] is False


class TestIRPTrust:
    """Test IRP trust chain evaluation."""

    def test_known_chain_trust(self, tmp_path: Path) -> None:
        """Chain of known SAT validators => min trust."""
        cmn = CMNRuntime(data_dir=tmp_path)
        cmn.boot()

        result = cmn.evaluate_trust_chain(["S1-Validator", "S3-Mediator"])
        assert result["trust"] == 0.90  # min(0.95, 0.90)
        assert result["poisoned"] is False

    def test_unknown_narrator_poisons(self, tmp_path: Path) -> None:
        """Unknown narrator in chain => trust = 0."""
        cmn = CMNRuntime(data_dir=tmp_path)
        cmn.boot()

        result = cmn.evaluate_trust_chain(["S1-Validator", "rogue_agent"])
        assert result["trust"] == 0.0
        assert result["poisoned"] is True


class TestFrozenAgentGuard:
    """Test Godel Escape enforcement."""

    def test_frozen_agent_blocked(self, tmp_path: Path) -> None:
        """Modifying P5-Ethicist must be blocked."""
        cmn = CMNRuntime(data_dir=tmp_path)
        cmn.boot()

        assert cmn.guard_frozen_agent("P5-Ethicist") is False
        assert cmn.guard_frozen_agent("S2-Oracle") is False

    def test_non_frozen_agent_allowed(self, tmp_path: Path) -> None:
        """Modifying P1-Planner is allowed."""
        cmn = CMNRuntime(data_dir=tmp_path)
        cmn.boot()

        assert cmn.guard_frozen_agent("P1-Planner") is True
