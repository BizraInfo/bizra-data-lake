from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from scripts.node0_standalone import Node0StandaloneManager, create_app


def _make_genesis_json(
    state_dir: Path, node_id: str = "node0-test"
) -> dict[str, object]:
    genesis_hash = list(range(32))
    genesis = {
        "timestamp": 1000,
        "identity": {
            "node_id": node_id,
            "public_key": "ab" * 32,
            "name": "Node0 Test",
            "location": "Dubai",
            "created_at": 1000,
            "identity_hash": list(range(32)),
        },
        "hardware": {},
        "knowledge": {},
        "pat_team": {
            "owner_node": node_id,
            "agents": [
                {
                    "agent_id": f"P{i}",
                    "role": f"PAT-{i}",
                    "public_key": "cd" * 32,
                    "capabilities": [],
                    "giants": [],
                    "created_at": 1000,
                    "agent_hash": list(range(32)),
                }
                for i in range(1, 8)
            ],
            "team_hash": list(range(32)),
        },
        "sat_team": {
            "agents": [
                {
                    "agent_id": f"S{i}",
                    "role": f"SAT-{i}",
                    "public_key": "ef" * 32,
                    "capabilities": [],
                    "giants": [],
                    "created_at": 1000,
                    "agent_hash": list(range(32)),
                }
                for i in range(1, 6)
            ],
            "team_hash": list(range(32)),
            "governance": {
                "quorum": 0.67,
                "voting_period_hours": 72,
                "upgrade_threshold": 0.8,
            },
        },
        "partnership_hash": list(range(32)),
        "genesis_hash": genesis_hash,
    }
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "node0_genesis.json").write_text(
        json.dumps(genesis, indent=2), encoding="utf-8"
    )
    (state_dir / "genesis_hash.txt").write_text(
        bytes(genesis_hash).hex(), encoding="utf-8"
    )
    return genesis


def _make_mvsa_proof(state_dir: Path, status: str = "ready") -> dict[str, object]:
    proof = {
        "schema_version": "1.0.0",
        "generated_at": "2026-03-10T12:00:00Z",
        "node_id": "node0-test",
        "genesis_hash": "ab" * 32,
        "genesis_hash_valid": True,
        "network": {
            "mode": "loopback",
            "bind_addr": "127.0.0.1:0",
            "bootstrap_ok": True,
            "peer_count": 0,
        },
        "consensus": {
            "proof_type": "local_self_validation",
            "proposal_ok": True,
            "self_validation_ok": True,
            "proof_id": "mvsa-proof-001",
        },
        "status": status,
        "reason_code": "OK",
    }
    (state_dir / "node0_mvsa_proof.json").write_text(
        json.dumps(proof, indent=2), encoding="utf-8"
    )
    return proof


def test_resolve_workspace_path_blocks_outside_workspace(tmp_path: Path) -> None:
    manager = Node0StandaloneManager(project_root=tmp_path)

    outside = Path("/tmp") / "outside.txt"
    with pytest.raises(ValueError, match="outside workspace"):
        manager._resolve_workspace_path(str(outside))


def test_filesystem_action_write_read_list(tmp_path: Path) -> None:
    manager = Node0StandaloneManager(project_root=tmp_path)

    write = manager._maybe_execute_filesystem_action(
        "write file missions/demo.txt :: hello from node0"
    )
    assert write is not None
    assert write["action"] == "write"

    target = tmp_path / "missions" / "demo.txt"
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "hello from node0"

    read = manager._maybe_execute_filesystem_action("read file missions/demo.txt")
    assert read is not None
    assert read["action"] == "read"
    assert "hello from node0" in read["preview"]

    listed = manager._maybe_execute_filesystem_action("list dir missions")
    assert listed is not None
    assert listed["action"] == "list"
    assert "demo.txt" in listed["entries"]


def test_health_is_degraded_without_activation(tmp_path: Path) -> None:
    manager = Node0StandaloneManager(project_root=tmp_path)

    report = manager.health()
    assert report["status"] == "degraded"
    assert report["gates"]["identity_credentials"] is False
    assert report["gates"]["assets_file"] is False


def test_health_is_read_only_for_lifecycle_v2(tmp_path: Path) -> None:
    manager = Node0StandaloneManager(project_root=tmp_path)
    lifecycle = {
        "schema_version": "2.0.0",
        "updated_at": "2026-03-10T12:00:00Z",
        "status": "degraded",
        "ok": True,
        "ready": False,
        "node_id": "node0-test",
        "origin": {"authority_source": "canonical_genesis"},
        "identity": {"pat_agents": 7, "sat_agents": 5},
        "artifacts": {},
        "gates": {"genesis_authority_valid": True},
        "mvsa": {},
        "mission": {},
        "restart_recovery": {
            "restart_recovery_ready": False,
            "validated_at": None,
            "required_artifacts_present": False,
        },
        "compat": {},
    }
    manager.lifecycle_path.write_text(json.dumps(lifecycle, indent=2), encoding="utf-8")

    before = manager.lifecycle_path.read_text(encoding="utf-8")
    report = manager.health()
    after = manager.lifecycle_path.read_text(encoding="utf-8")

    assert report["schema_version"] == "2.0.0"
    assert report["status"] == "degraded"
    assert before == after


@pytest.mark.asyncio
async def test_run_task_reports_filesystem_action(tmp_path: Path) -> None:
    manager = Node0StandaloneManager(project_root=tmp_path)

    result = await manager.run_task(
        "write file missions/from_mission.txt :: hello from mission",
        browser_mode="mock",
    )

    fs = result.get("filesystem_action")
    assert fs is not None
    assert fs["action"] == "write"
    target = tmp_path / "missions" / "from_mission.txt"
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "hello from mission"


def test_create_app_activate_accepts_json_body(tmp_path: Path) -> None:
    manager = Node0StandaloneManager(project_root=tmp_path)
    seen: dict[str, object] = {}

    def _fake_activate(
        architect: str = "MoMo", strict: bool = False
    ) -> dict[str, object]:
        seen["architect"] = architect
        seen["strict"] = strict
        return {"ok": True, "architect": architect, "strict": strict}

    manager.activate = _fake_activate  # type: ignore[method-assign]
    app = create_app(manager)
    client = TestClient(app)

    response = client.post("/activate", json={"architect": "tester", "strict": True})

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert seen == {"architect": "tester", "strict": True}


def test_create_app_task_accepts_json_body(tmp_path: Path) -> None:
    manager = Node0StandaloneManager(project_root=tmp_path)
    seen: dict[str, object] = {}

    async def _fake_run_task(
        description: str,
        source: str = "node0_standalone_api",
        browser_mode: str = "mock",
    ) -> dict[str, object]:
        seen["description"] = description
        seen["source"] = source
        seen["browser_mode"] = browser_mode
        return {
            "status": "COMPLETE",
            "filesystem_action": {"action": "write"},
        }

    manager.run_task = _fake_run_task  # type: ignore[method-assign]
    app = create_app(manager)
    client = TestClient(app)

    response = client.post(
        "/task",
        json={"description": "write file missions/api.txt :: hello from api"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "COMPLETE"
    assert body["filesystem_action"]["action"] == "write"
    assert seen == {
        "description": "write file missions/api.txt :: hello from api",
        "source": "node0_standalone_api",
        "browser_mode": "mock",
    }


def test_create_app_mvsa_routes_enforce_api_key(tmp_path: Path) -> None:
    manager = Node0StandaloneManager(project_root=tmp_path)
    manager.mvsa = lambda: {"status": "ready"}  # type: ignore[method-assign]
    manager.prove_mvsa = lambda: {  # type: ignore[method-assign]
        "ok": True,
        "proof_ok": True,
        "lifecycle_status": "degraded",
    }
    app = create_app(manager, api_key="secret")
    client = TestClient(app)

    assert client.get("/health").status_code == 200
    assert client.get("/mvsa").status_code == 401
    assert client.post("/prove-mvsa").status_code == 401
    assert (
        client.get("/mvsa", headers={"x-api-key": "secret"}).json()["status"] == "ready"
    )
    assert (
        client.post("/prove-mvsa", headers={"x-api-key": "secret"}).json()[
            "lifecycle_status"
        ]
        == "degraded"
    )


def test_create_app_lists_agents_and_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = Node0StandaloneManager(project_root=tmp_path)

    class _FakeResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return self._payload

    class _FakeAsyncClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> "_FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
            return False

        async def get(self, url: str) -> _FakeResponse:
            assert url.endswith("/api/tags")
            return _FakeResponse({"models": [{"name": "phi3:mini"}]})

    monkeypatch.setitem(
        sys.modules,
        "httpx",
        types.SimpleNamespace(AsyncClient=_FakeAsyncClient),
    )

    app = create_app(manager)
    client = TestClient(app)

    models = client.get("/v1/models")
    agents = client.get("/v1/agents")

    assert models.status_code == 200
    assert models.json()["ollama_models"] == ["phi3:mini"]
    assert agents.status_code == 200
    assert agents.json()["total"] == 12


def test_create_app_query_accepts_json_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = Node0StandaloneManager(project_root=tmp_path)

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"response": "4", "eval_count": 2}

    class _FakeAsyncClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> "_FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
            return False

        async def post(self, url: str, json: dict[str, object]) -> _FakeResponse:
            assert url.endswith("/api/generate")
            assert json["prompt"] == "What is 2+2?"
            return _FakeResponse()

    monkeypatch.setitem(
        sys.modules,
        "httpx",
        types.SimpleNamespace(AsyncClient=_FakeAsyncClient),
    )

    app = create_app(manager)
    client = TestClient(app)

    response = client.post(
        "/v1/query",
        json={"prompt": "What is 2+2?", "model": "P6-Publisher", "max_tokens": 16},
    )

    assert response.status_code == 200
    body = response.json()
    # P6-Publisher resolves through NODE0_MODEL_FLEET (YAML → Ollama fallback)
    from scripts.node0_standalone import NODE0_MODEL_FLEET as _fleet

    assert body["model"] == _fleet["P6-Publisher"]
    assert body["response"] == "4"


def test_prove_mvsa_blocks_without_authority(tmp_path: Path) -> None:
    manager = Node0StandaloneManager(project_root=tmp_path)

    result = manager.prove_mvsa()

    assert result["ok"] is False
    assert result["status"] == "blocked"
    lifecycle = json.loads(manager.lifecycle_path.read_text(encoding="utf-8"))
    assert lifecycle["schema_version"] == "2.0.0"
    assert lifecycle["status"] == "blocked"
    assert lifecycle["gates"]["genesis_authority_valid"] is False


def test_prove_mvsa_updates_lifecycle_from_authority(tmp_path: Path) -> None:
    manager = Node0StandaloneManager(project_root=tmp_path)
    state_dir = tmp_path / "sovereign_state"
    _make_genesis_json(state_dir)
    manager.assets_path.write_text(
        json.dumps({"integrations": {}}, indent=2), encoding="utf-8"
    )
    manager.awareness_path.write_text(
        json.dumps({"node_id": "node0-test"}, indent=2), encoding="utf-8"
    )
    manager.urp_path.write_text(
        json.dumps({"signed": True, "signature_verified": True}, indent=2),
        encoding="utf-8",
    )
    proof = _make_mvsa_proof(state_dir)

    import core.sovereign.node0_mvsa as node0_mvsa

    original = node0_mvsa.run_mvsa_proof
    node0_mvsa.run_mvsa_proof = lambda _sd, _pr: proof  # type: ignore[assignment]
    try:
        result = manager.prove_mvsa()
    finally:
        node0_mvsa.run_mvsa_proof = original  # type: ignore[assignment]

    assert result["ok"] is True
    assert result["proof_ok"] is True
    assert result["lifecycle_status"] == "degraded"

    lifecycle = json.loads(manager.lifecycle_path.read_text(encoding="utf-8"))
    assert lifecycle["schema_version"] == "2.0.0"
    assert lifecycle["node_id"] == "node0-test"
    assert lifecycle["origin"]["authority_source"] == "canonical_genesis"
    assert lifecycle["gates"]["genesis_authority_valid"] is True
    assert lifecycle["gates"]["mvsa_network_bootstrap_ok"] is True
    assert lifecycle["gates"]["mvsa_self_validation_ok"] is True
    assert lifecycle["status"] == "degraded"


def test_update_lifecycle_mission_marks_ready_when_recovery_passes(
    tmp_path: Path,
) -> None:
    manager = Node0StandaloneManager(project_root=tmp_path)
    state_dir = tmp_path / "sovereign_state"
    _make_genesis_json(state_dir)
    _make_mvsa_proof(state_dir)
    manager.assets_path.write_text(
        json.dumps({"integrations": {}}, indent=2), encoding="utf-8"
    )
    manager.awareness_path.write_text(
        json.dumps({"node_id": "node0-test"}, indent=2), encoding="utf-8"
    )
    manager.urp_path.write_text(
        json.dumps({"signed": True, "signature_verified": True}, indent=2),
        encoding="utf-8",
    )
    manager.lifecycle_path.write_text(
        json.dumps(
            {
                "schema_version": "2.0.0",
                "updated_at": "2026-03-10T12:00:00Z",
                "status": "degraded",
                "ok": True,
                "ready": False,
                "node_id": "node0-test",
                "origin": {"authority_source": "canonical_genesis"},
                "identity": {"pat_agents": 7, "sat_agents": 5},
                "artifacts": {},
                "gates": {
                    "genesis_authority_valid": True,
                    "identity_ready": True,
                    "pat_sat_ready": True,
                    "urp_signed": True,
                    "urp_verified": True,
                    "assets_written": True,
                    "awareness_written": True,
                    "mvsa_network_bootstrap_ok": True,
                    "mvsa_self_validation_ok": True,
                    "mission_path_receipted": False,
                    "restart_recovery_ready": False,
                },
                "mvsa": {"status": "ready"},
                "mission": {},
                "restart_recovery": {
                    "restart_recovery_ready": False,
                    "validated_at": None,
                    "required_artifacts_present": False,
                },
                "compat": {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    manager._update_lifecycle_mission("receipt-123", "COMPLETE", 0.99, 0.98)

    lifecycle = json.loads(manager.lifecycle_path.read_text(encoding="utf-8"))
    assert lifecycle["mission"]["last_evidence_receipt_id"] == "receipt-123"
    assert lifecycle["gates"]["mission_path_receipted"] is True
    assert lifecycle["gates"]["restart_recovery_ready"] is True
    assert lifecycle["restart_recovery"]["restart_recovery_ready"] is True
    assert lifecycle["status"] == "ready"
    assert lifecycle["ready"] is True
