from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest
from starlette.testclient import TestClient

from scripts.node0_standalone import Node0StandaloneManager, create_app


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

    def _fake_activate(architect: str = "MoMo", strict: bool = False) -> dict[str, object]:
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

        async def __aexit__(
            self, exc_type: object, exc: object, tb: object
        ) -> bool:
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

        async def __aexit__(
            self, exc_type: object, exc: object, tb: object
        ) -> bool:
            return False

        async def post(
            self, url: str, json: dict[str, object]
        ) -> _FakeResponse:
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
