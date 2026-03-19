"""Tests for Node0 MVSA proof Python wrapper (Wave 2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from core.sovereign.node0_mvsa import (
    REASON_BINARY_UNAVAILABLE,
    _resolve_binary,
    read_mvsa_proof,
    run_mvsa_proof,
)


def _make_proof(status: str = "ready") -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "generated_at": "2026-03-10T12:00:00Z",
        "node_id": "BIZRA-TEST",
        "genesis_hash": "a7f68f1f" * 8,
        "genesis_hash_valid": True,
        "network": {
            "mode": "loopback",
            "bind_addr": "127.0.0.1:9999",
            "bootstrap_ok": True,
            "peer_count": 0,
        },
        "consensus": {
            "proof_type": "local_self_validation",
            "proposal_ok": True,
            "self_validation_ok": True,
            "proof_id": "mvsa-proof-test123",
        },
        "status": status,
        "reason_code": "OK",
        "duration_ms": 42.0,
    }


class TestBinaryResolution:
    """Tests for Rust binary resolution order."""

    def test_env_var_takes_precedence(self, tmp_path: Path) -> None:
        # Create a fake binary
        fake_bin = tmp_path / "node0-mvsa"
        fake_bin.write_text("#!/bin/sh\necho test", encoding="utf-8")
        fake_bin.chmod(0o755)

        with patch.dict("os.environ", {"BIZRA_NODE0_MVSA_BIN": str(fake_bin)}):
            result = _resolve_binary(tmp_path)
            assert result == fake_bin

    def test_returns_none_when_no_binary(self, tmp_path: Path) -> None:
        with patch.dict("os.environ", {}, clear=True):
            result = _resolve_binary(tmp_path)
            assert result is None

    def test_prefers_release_over_debug(self, tmp_path: Path) -> None:
        omega = tmp_path / "bizra-omega"
        release = omega / "target" / "release" / "node0-mvsa"
        debug = omega / "target" / "debug" / "node0-mvsa"
        release.parent.mkdir(parents=True)
        debug.parent.mkdir(parents=True)
        release.write_text("#!/bin/sh", encoding="utf-8")
        release.chmod(0o755)
        debug.write_text("#!/bin/sh", encoding="utf-8")
        debug.chmod(0o755)

        with patch.dict("os.environ", {}, clear=True):
            result = _resolve_binary(tmp_path)
            assert result == release


class TestReadMvsaProof:
    """Tests for reading persisted proof artifacts."""

    def test_returns_none_when_absent(self, tmp_path: Path) -> None:
        assert read_mvsa_proof(tmp_path) is None

    def test_reads_valid_proof(self, tmp_path: Path) -> None:
        proof = _make_proof()
        (tmp_path / "node0_mvsa_proof.json").write_text(
            json.dumps(proof), encoding="utf-8"
        )
        result = read_mvsa_proof(tmp_path)
        assert result is not None
        assert result["status"] == "ready"
        assert result["consensus"]["self_validation_ok"] is True


class TestRunMvsaProof:
    """Tests for the full proof execution path."""

    def test_fails_when_no_binary_and_no_cargo(self, tmp_path: Path) -> None:
        state_dir = tmp_path / "sovereign_state"
        state_dir.mkdir()

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match=REASON_BINARY_UNAVAILABLE):
                run_mvsa_proof(state_dir, tmp_path)

    def test_success_with_mock_binary(self, tmp_path: Path) -> None:
        """Simulate a successful Rust binary run by writing proof directly."""
        state_dir = tmp_path / "sovereign_state"
        state_dir.mkdir()
        proof = _make_proof()

        import subprocess

        fake_result = subprocess.CompletedProcess(
            args=["fake"], returncode=0, stdout="", stderr="OK"
        )

        # Pre-write the proof file (as the binary would)
        (state_dir / "node0_mvsa_proof.json").write_text(
            json.dumps(proof), encoding="utf-8"
        )

        with (
            patch("core.sovereign.node0_mvsa._resolve_binary") as mock_resolve,
            patch("core.sovereign.node0_mvsa._run_binary", return_value=fake_result),
        ):
            mock_resolve.return_value = Path("/fake/node0-mvsa")
            result = run_mvsa_proof(state_dir, tmp_path)

        assert result["status"] == "ready"
        assert result["genesis_hash_valid"] is True
        assert result["consensus"]["self_validation_ok"] is True
