"""Tests for Node0 authority resolution and migration (Wave 1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core.sovereign.node0_authority import (
    REASON_CANONICAL_VALID,
    REASON_LEGACY_CONFLICT,
    REASON_LEGACY_INSUFFICIENT,
    REASON_LEGACY_MIGRATED,
    REASON_NO_AUTHORITY,
    RESULT_BLOCKED,
    RESULT_CANONICAL,
    RESULT_MIGRATED,
    require_authority,
    resolve_authority,
)


def _make_genesis(
    node_id: str = "BIZRA-TEST",
    pat_count: int = 7,
    sat_count: int = 5,
) -> dict[str, Any]:
    """Build a ceremony-compatible genesis dict."""
    pat_agents = [
        {
            "agent_id": f"P{i+1}",
            "role": f"PAT-{i+1}",
            "public_key": f"pk-pat-{i+1}",
            "capabilities": [],
            "giants": [],
            "created_at": 1000,
            "agent_hash": list(range(32)),
        }
        for i in range(pat_count)
    ]
    sat_agents = [
        {
            "agent_id": f"S{i+1}",
            "role": f"SAT-{i+1}",
            "public_key": f"pk-sat-{i+1}",
            "capabilities": [],
            "giants": [],
            "created_at": 1000,
            "agent_hash": list(range(32)),
        }
        for i in range(sat_count)
    ]
    genesis_hash = list(range(32))
    return {
        "timestamp": 1000,
        "identity": {
            "node_id": node_id,
            "public_key": "pk-node0",
            "name": "Test",
            "location": "Test",
            "created_at": 1000,
            "identity_hash": list(range(32)),
        },
        "hardware": {},
        "knowledge": {},
        "pat_team": {"owner_node": node_id, "agents": pat_agents, "team_hash": list(range(32))},
        "sat_team": {
            "agents": sat_agents,
            "team_hash": list(range(32)),
            "governance": {"quorum": 0.67, "voting_period_hours": 72, "upgrade_threshold": 0.8},
        },
        "partnership_hash": list(range(32)),
        "genesis_hash": genesis_hash,
    }


def _write_canonical(state_dir: Path, genesis: dict[str, Any]) -> None:
    """Write canonical genesis + hash file."""
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "node0_genesis.json").write_text(
        json.dumps(genesis, indent=2), encoding="utf-8"
    )
    genesis_hash = genesis.get("genesis_hash", [])
    hash_hex = bytes(genesis_hash).hex()
    (state_dir / "genesis_hash.txt").write_text(hash_hex, encoding="utf-8")


class TestCanonicalAuthority:
    """Tests for canonical authority resolution (step 1)."""

    def test_canonical_valid(self, tmp_path: Path) -> None:
        state_dir = tmp_path / "sovereign_state"
        genesis = _make_genesis()
        _write_canonical(state_dir, genesis)

        result = resolve_authority(state_dir, tmp_path)
        assert result.is_valid
        assert result.result == RESULT_CANONICAL
        assert result.reason_code == REASON_CANONICAL_VALID
        assert result.genesis is not None
        assert result.genesis.node_id == "BIZRA-TEST"

    def test_canonical_missing_blocks_without_legacy(self, tmp_path: Path) -> None:
        state_dir = tmp_path / "sovereign_state"
        state_dir.mkdir(parents=True, exist_ok=True)

        result = resolve_authority(state_dir, tmp_path)
        assert not result.is_valid
        assert result.result == RESULT_BLOCKED
        assert result.reason_code == REASON_NO_AUTHORITY

    def test_pat_sat_from_authority(self, tmp_path: Path) -> None:
        state_dir = tmp_path / "sovereign_state"
        genesis = _make_genesis(pat_count=7, sat_count=5)
        _write_canonical(state_dir, genesis)

        result = resolve_authority(state_dir, tmp_path)
        assert result.genesis is not None
        assert len(result.genesis.pat_team) == 7
        assert len(result.genesis.sat_team) == 5
        assert result.genesis.pat_agent_ids == [f"P{i+1}" for i in range(7)]
        assert result.genesis.sat_agent_ids == [f"S{i+1}" for i in range(5)]


class TestLegacyMigration:
    """Tests for legacy source migration (step 2-3)."""

    def test_migrates_legacy_ceremony(self, tmp_path: Path) -> None:
        state_dir = tmp_path / "sovereign_state"
        state_dir.mkdir(parents=True, exist_ok=True)
        legacy_path = state_dir / "genesis.json"
        genesis = _make_genesis(node_id="BIZRA-LEGACY")
        legacy_path.write_text(json.dumps(genesis, indent=2), encoding="utf-8")

        result = resolve_authority(state_dir, tmp_path)
        assert result.is_valid
        assert result.result == RESULT_MIGRATED
        assert result.reason_code == REASON_LEGACY_MIGRATED
        assert result.genesis is not None
        assert result.genesis.node_id == "BIZRA-LEGACY"

        # Canonical files should now exist
        assert (state_dir / "node0_genesis.json").exists()
        assert (state_dir / "genesis_hash.txt").exists()
        assert (state_dir / "pat_roster.txt").exists()
        assert (state_dir / "sat_roster.txt").exists()
        assert (state_dir / "node0_authority_migration.json").exists()

    def test_rejects_reference_only(self, tmp_path: Path) -> None:
        state_dir = tmp_path / "sovereign_state"
        state_dir.mkdir(parents=True, exist_ok=True)

        # Put genesis in 04_GOLD (reference-only)
        gold_dir = tmp_path / "04_GOLD"
        gold_dir.mkdir(parents=True, exist_ok=True)
        genesis = _make_genesis(node_id="BIZRA-GOLD")
        (gold_dir / "genesis.json").write_text(
            json.dumps(genesis, indent=2), encoding="utf-8"
        )

        result = resolve_authority(state_dir, tmp_path)
        assert not result.is_valid
        assert result.result == RESULT_BLOCKED
        assert result.reason_code == REASON_LEGACY_INSUFFICIENT

    def test_rejects_conflicting_legacy_sources(self, tmp_path: Path) -> None:
        state_dir = tmp_path / "sovereign_state"
        state_dir.mkdir(parents=True, exist_ok=True)

        # Source 1 in sovereign_state/genesis.json
        g1 = _make_genesis(node_id="BIZRA-A")
        (state_dir / "genesis.json").write_text(
            json.dumps(g1, indent=2), encoding="utf-8"
        )

        # Source 2 in bizra-storage/genesis.json with different hash
        bs_dir = tmp_path / "bizra-storage"
        bs_dir.mkdir(parents=True, exist_ok=True)
        g2 = _make_genesis(node_id="BIZRA-B")
        g2["genesis_hash"] = list(range(32, 64))  # Different hash
        (bs_dir / "genesis.json").write_text(
            json.dumps(g2, indent=2), encoding="utf-8"
        )

        result = resolve_authority(state_dir, tmp_path)
        assert not result.is_valid
        assert result.result == RESULT_BLOCKED
        assert result.reason_code == REASON_LEGACY_CONFLICT


class TestRequireAuthority:
    """Tests for the fail-closed require_authority()."""

    def test_returns_genesis_when_valid(self, tmp_path: Path) -> None:
        state_dir = tmp_path / "sovereign_state"
        genesis_data = _make_genesis()
        _write_canonical(state_dir, genesis_data)

        genesis = require_authority(state_dir, tmp_path)
        assert genesis.node_id == "BIZRA-TEST"

    def test_raises_when_no_authority(self, tmp_path: Path) -> None:
        state_dir = tmp_path / "sovereign_state"
        state_dir.mkdir(parents=True, exist_ok=True)

        with pytest.raises(RuntimeError, match="authority resolution failed"):
            require_authority(state_dir, tmp_path)
