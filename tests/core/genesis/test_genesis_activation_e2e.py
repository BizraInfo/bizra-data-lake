"""
E2E Tests — Genesis Activation Pipeline
=========================================

Tests the full pipeline: ceremony → orchestrator → heartbeat → evidence.

Standing on Giants:
- Nakamoto (2008): Genesis block determinism
- Merkle (1979): Hash chain integrity
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.genesis.activation import GenesisActivation, GenesisActivationResult
from core.proof_engine.genesis_ceremony import verify_ceremony

# Deterministic seed for reproducible tests
TEST_SEED = b"bizra-test-seed-32-bytes-minimum!"
ALT_SEED = b"bizra-alt-seed---32-bytes-minimum!"


class TestGenesisActivationE2E:
    """Full pipeline: ceremony → orchestrator → heartbeat → evidence."""

    def test_activation_produces_all_receipts(self, tmp_path: Path) -> None:
        """Ceremony + orchestrator + boot all return valid results."""
        activation = GenesisActivation(
            node_seed=TEST_SEED,
            data_dir=tmp_path / "genesis",
            skip_breath=True,  # Breath requires full Helix3 wiring
        )
        result = activation.activate()

        assert isinstance(result, GenesisActivationResult)
        assert result.ceremony_result is not None
        assert result.ceremony_result.genesis_hash
        assert result.ceremony_result.pat_roster
        assert result.ceremony_result.sat_roster
        assert result.boot_receipt_dict
        assert result.boot_receipt_dict.get("node_id")
        assert result.node_id
        assert result.genesis_hash
        assert result.activation_hash
        assert result.duration_ms > 0

    def test_activation_hash_deterministic(self, tmp_path: Path) -> None:
        """Same seed → same ceremony genesis_hash (reproducible genesis)."""
        from core.proof_engine.genesis_ceremony import CeremonyConfig, run_ceremony

        # Fix timestamp so ceremony is fully deterministic
        config = CeremonyConfig(timestamp_ms=1700000000000)

        result1 = run_ceremony(TEST_SEED, config)
        result2 = run_ceremony(TEST_SEED, config)

        assert result1.genesis_hash == result2.genesis_hash
        assert result1.pat_roster == result2.pat_roster
        assert result1.sat_roster == result2.sat_roster

    def test_activation_ceremony_verified(self, tmp_path: Path) -> None:
        """verify_ceremony() passes on activation output."""
        activation = GenesisActivation(
            node_seed=TEST_SEED,
            data_dir=tmp_path / "genesis",
            skip_orchestrator=True,
            skip_breath=True,
        )
        activation.activate()

        genesis_path = tmp_path / "genesis" / "node0_genesis.json"
        assert genesis_path.exists()

        valid, reasons = verify_ceremony(genesis_path)
        assert valid, f"Ceremony verification failed: {reasons}"
        assert reasons == []

    def test_activation_evidence_chain_valid(self, tmp_path: Path) -> None:
        """Boot receipt chains correctly to ceremony hash."""
        activation = GenesisActivation(
            node_seed=TEST_SEED,
            data_dir=tmp_path / "genesis",
            skip_orchestrator=True,
            skip_breath=True,
        )
        result = activation.activate()

        assert result.evidence_chain_valid
        assert result.boot_receipt_dict.get("boot_hash")
        # Activation hash incorporates both genesis and boot hashes
        assert result.activation_hash != result.genesis_hash
        assert result.activation_hash != result.boot_receipt_dict["boot_hash"]

    def test_activation_degraded_mode(self, tmp_path: Path) -> None:
        """Works without orchestrator (Level 0 graceful degradation)."""
        activation = GenesisActivation(
            node_seed=TEST_SEED,
            data_dir=tmp_path / "genesis",
            skip_orchestrator=True,
            skip_breath=True,
        )
        result = activation.activate()

        assert isinstance(result, GenesisActivationResult)
        assert result.orchestrator_success  # True because skipped (not failed)
        assert result.node_id
        assert result.activation_hash

    def test_activation_rejects_empty_seed(self, tmp_path: Path) -> None:
        """Empty/None seed raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            GenesisActivation(node_seed=b"", data_dir=tmp_path / "genesis")

    def test_activation_rejects_non_bytes_seed(self, tmp_path: Path) -> None:
        """String seed raises TypeError."""
        with pytest.raises(TypeError, match="bytes"):
            GenesisActivation(node_seed="not-bytes", data_dir=tmp_path / "genesis")  # type: ignore[arg-type]

    def test_activation_writes_receipt_artifact(self, tmp_path: Path) -> None:
        """Activation receipt JSON is written to disk and is valid."""
        activation = GenesisActivation(
            node_seed=TEST_SEED,
            data_dir=tmp_path / "genesis",
            skip_orchestrator=True,
            skip_breath=True,
        )
        result = activation.activate()

        receipt_path = tmp_path / "genesis" / "activation_receipt.json"
        assert receipt_path.exists()

        receipt_data = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt_data["activation_hash"] == result.activation_hash
        assert receipt_data["node_id"] == result.node_id
        assert receipt_data["genesis_hash"] == result.genesis_hash
        assert receipt_data["evidence_chain_valid"] is True

    def test_activation_verify_passes(self, tmp_path: Path) -> None:
        """verify() confirms artifacts on disk are intact."""
        activation = GenesisActivation(
            node_seed=TEST_SEED,
            data_dir=tmp_path / "genesis",
            skip_orchestrator=True,
            skip_breath=True,
        )
        activation.activate()

        valid, reasons = activation.verify()
        assert valid, f"Verification failed: {reasons}"
        assert reasons == []

    def test_activation_verify_detects_missing_files(self, tmp_path: Path) -> None:
        """verify() fails when genesis files are missing."""
        activation = GenesisActivation(
            node_seed=TEST_SEED,
            data_dir=tmp_path / "empty",
        )
        valid, reasons = activation.verify()
        assert not valid
        assert any("MISSING" in r for r in reasons)

    def test_different_seeds_produce_different_hashes(self, tmp_path: Path) -> None:
        """Different seeds must produce different genesis identities."""
        a1 = GenesisActivation(
            node_seed=TEST_SEED,
            data_dir=tmp_path / "g1",
            skip_orchestrator=True,
            skip_breath=True,
        )
        a2 = GenesisActivation(
            node_seed=ALT_SEED,
            data_dir=tmp_path / "g2",
            skip_orchestrator=True,
            skip_breath=True,
        )
        r1 = a1.activate()
        r2 = a2.activate()

        assert r1.genesis_hash != r2.genesis_hash
        assert r1.node_id != r2.node_id
        assert r1.activation_hash != r2.activation_hash

    def test_ceremony_produces_7_pat_5_sat(self, tmp_path: Path) -> None:
        """Ceremony roster has exactly 7 PAT and 5 SAT agents."""
        activation = GenesisActivation(
            node_seed=TEST_SEED,
            data_dir=tmp_path / "genesis",
            skip_orchestrator=True,
            skip_breath=True,
        )
        result = activation.activate()

        pat_lines = [
            l
            for l in result.ceremony_result.pat_roster.strip().split("\n")
            if l.strip()
        ]
        sat_lines = [
            l
            for l in result.ceremony_result.sat_roster.strip().split("\n")
            if l.strip()
        ]

        assert len(pat_lines) == 7, f"Expected 7 PAT agents, got {len(pat_lines)}"
        assert len(sat_lines) == 5, f"Expected 5 SAT agents, got {len(sat_lines)}"

    def test_genesis_files_written_to_disk(self, tmp_path: Path) -> None:
        """All expected genesis files are produced."""
        activation = GenesisActivation(
            node_seed=TEST_SEED,
            data_dir=tmp_path / "genesis",
            skip_orchestrator=True,
            skip_breath=True,
        )
        activation.activate()

        genesis_dir = tmp_path / "genesis"
        assert (genesis_dir / "node0_genesis.json").exists()
        assert (genesis_dir / "genesis_hash.txt").exists()
        assert (genesis_dir / "pat_roster.txt").exists()
        assert (genesis_dir / "sat_roster.txt").exists()
        assert (genesis_dir / "activation_receipt.json").exists()
