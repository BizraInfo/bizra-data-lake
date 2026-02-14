"""Unit tests for Node0/Block0 origin guard enforcement."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.sovereign.origin_guard import (
    enforce_node0_fail_closed,
    resolve_origin_snapshot,
    validate_genesis_chain,
)
from core.sovereign.runtime_core import SovereignRuntime
from core.sovereign.runtime_types import RuntimeConfig


def _write_valid_genesis_fixture(state_dir: Path) -> None:
    genesis_hash_hex = "ab" * 32
    payload = {
        "timestamp": 1770295290922,
        "identity": {
            "node_id": "node0_fixture_0001",
            "public_key": "11" * 32,
            "name": "Node0 Fixture",
            "location": "CI",
            "created_at": 1770295290922,
            "identity_hash": [1] * 32,
        },
        "pat_team": {
            "owner_node": "node0_fixture_0001",
            "agents": [],
            "team_hash": [2] * 32,
        },
        "sat_team": {
            "owner_node": "node0_fixture_0001",
            "agents": [],
            "team_hash": [3] * 32,
            "governance": {"quorum": 0.67, "voting_period_hours": 72, "upgrade_threshold": 0.8},
        },
        "partnership_hash": [4] * 32,
        "genesis_hash": list(bytes.fromhex(genesis_hash_hex)),
        "hardware": {},
        "knowledge": {},
    }
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "node0_genesis.json").write_text(json.dumps(payload), encoding="utf-8")
    (state_dir / "genesis_hash.txt").write_text(genesis_hash_hex, encoding="utf-8")


def test_validate_genesis_chain_passes_with_valid_fixture(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    _write_valid_genesis_fixture(state_dir)
    valid, reason = validate_genesis_chain(state_dir)
    assert valid is True
    assert reason == "ok"


def test_validate_genesis_chain_fails_when_missing(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    valid, reason = validate_genesis_chain(state_dir)
    assert valid is False
    assert "node0_genesis.json" in reason


def test_validate_genesis_chain_fails_on_hash_mismatch(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    _write_valid_genesis_fixture(state_dir)
    (state_dir / "genesis_hash.txt").write_text("ff" * 32, encoding="utf-8")
    valid, reason = validate_genesis_chain(state_dir)
    assert valid is False
    assert "validation failed" in reason


def test_resolve_origin_snapshot_defaults_to_ephemeral(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    snapshot = resolve_origin_snapshot(state_dir, "node")
    assert snapshot["designation"] == "ephemeral_node"
    assert snapshot["genesis_node"] is False
    assert snapshot["authority_source"] == "genesis_files"
    assert snapshot["hash_validated"] is False


def test_resolve_origin_snapshot_node0_returns_genesis_projection(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    _write_valid_genesis_fixture(state_dir)
    snapshot = resolve_origin_snapshot(state_dir, "node0")
    assert snapshot["designation"] == "node0"
    assert snapshot["genesis_node"] is True
    assert snapshot["genesis_block"] is True
    assert snapshot["block_id"] == "block0"
    assert snapshot["home_base_device"] is True
    assert snapshot["node_id"] == "node0_fixture_0001"
    assert snapshot["hash_validated"] is True


def test_enforce_node0_fail_closed_raises_on_missing_genesis(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Node0 genesis enforcement failed"):
        enforce_node0_fail_closed(tmp_path / "missing", "node0")


@pytest.mark.asyncio
async def test_runtime_initialize_fails_closed_for_node0_without_genesis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BIZRA_NODE_ROLE", "node0")
    cfg = RuntimeConfig(
        node_id="test-node-0001",
        enable_graph_reasoning=False,
        enable_snr_optimization=False,
        enable_guardian_validation=False,
        enable_autonomous_loop=False,
        enable_cache=False,
        enable_persistence=False,
        autonomous_enabled=False,
        enable_zpk_preflight=False,
        enable_proactive_kernel=False,
        state_dir=tmp_path / "missing_state",
    )
    runtime = SovereignRuntime(cfg)
    with pytest.raises(RuntimeError, match="Node0 genesis enforcement failed"):
        await runtime.initialize()

