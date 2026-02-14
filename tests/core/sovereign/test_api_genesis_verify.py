"""Tests for genesis verification endpoints with normalized origin projection."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from starlette.testclient import TestClient

from core.sovereign.api import create_fastapi_app


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


def _runtime(state_dir: Path) -> MagicMock:
    runtime = MagicMock()
    runtime.metrics = MagicMock(to_prometheus=lambda include_help=False: "")
    runtime.status.return_value = {
        "health": {"status": "healthy"},
        "identity": {"version": "test"},
        "state": {"running": True},
        "autonomous": {"running": False},
    }
    runtime.config = SimpleNamespace(state_dir=state_dir)
    runtime._state_dir = state_dir
    return runtime


def test_verify_genesis_includes_origin_projection(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BIZRA_NODE_ROLE", "node0")
    state_dir = tmp_path / "state"
    _write_valid_genesis_fixture(state_dir)
    app = create_fastapi_app(_runtime(state_dir))
    client = TestClient(app)

    resp = client.post("/v1/verify/genesis")
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"] == "APPROVED"
    assert body["artifacts"]["hash_validated"] is True
    assert body["artifacts"]["origin"]["designation"] == "node0"

    header = client.get("/v1/verify/genesis/header")
    assert header.status_code == 200
    header_body = header.json()
    assert header_body["hash_validated"] is True
    assert header_body["origin"]["designation"] == "node0"


def test_verify_genesis_rejects_invalid_chain(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BIZRA_NODE_ROLE", "node0")
    state_dir = tmp_path / "missing"
    app = create_fastapi_app(_runtime(state_dir))
    client = TestClient(app)

    resp = client.post("/v1/verify/genesis")
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"] == "REJECTED"
    assert "GENESIS_CHAIN_INVALID" in body["reason_codes"]
    assert body["artifacts"]["hash_validated"] is False

    header = client.get("/v1/verify/genesis/header")
    assert header.status_code == 503
    header_body = header.json()
    assert header_body["hash_validated"] is False

