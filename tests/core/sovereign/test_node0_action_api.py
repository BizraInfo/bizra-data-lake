"""Node0 desktop/browser action-intent API tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient

from core.sovereign.api import create_fastapi_app


def _runtime(state_dir: Path) -> MagicMock:
    runtime = MagicMock()
    runtime.config = SimpleNamespace(state_dir=state_dir)
    runtime.status.return_value = {
        "health": {"status": "healthy", "critical_subsystems": {}},
        "identity": {"version": "test"},
        "state": {"running": True},
        "autonomous": {"running": False},
    }
    runtime._constitutional_wallets = []
    runtime._last_tick_result = None
    runtime._node0 = None
    runtime._agent_db = None
    return runtime


def test_node0_action_intent_accepts_bounded_browser_handoff(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("BIZRA_AUTH_ALLOW_ANONYMOUS", "true")
    app = create_fastapi_app(_runtime(tmp_path))
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/v1/node0/action-intent",
        json={
            "action_type": "open_url",
            "target": "https://example.com/demo",
            "label": "Open demo",
            "consent": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["accepted"] is True
    assert payload["status"] == "ready_for_user_handoff"
    assert payload["action_type"] == "open_url"
    assert payload["handoff_method"] == "window_open"
    assert payload["server_executed"] is False
    assert payload["truth_label"] == "[ENFORCEMENT: WIRED]"


def test_node0_action_intent_rejects_missing_confirmation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("BIZRA_AUTH_ALLOW_ANONYMOUS", "true")
    app = create_fastapi_app(_runtime(tmp_path))
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/v1/node0/action-intent",
        json={
            "action_type": "copy_text",
            "target": "copy this only after explicit approval",
            "consent": False,
        },
    )

    assert response.status_code == 400
    assert response.json()["error"] == "Explicit user confirmation required"


@pytest.mark.parametrize(
    "target",
    [
        "file:///etc/passwd",
        "javascript:alert(1)",
        "data:text/html,<script>alert(1)</script>",
    ],
)
def test_node0_action_intent_rejects_non_http_browser_targets(
    tmp_path: Path,
    monkeypatch,
    target: str,
) -> None:
    monkeypatch.setenv("BIZRA_AUTH_ALLOW_ANONYMOUS", "true")
    app = create_fastapi_app(_runtime(tmp_path))
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/v1/node0/action-intent",
        json={
            "action_type": "open_url",
            "target": target,
            "consent": True,
        },
    )

    assert response.status_code == 400
    assert response.json()["error"] == "Only http(s) URLs are allowed"


def test_node0_action_intent_requires_authentication(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("BIZRA_AUTH_ALLOW_ANONYMOUS", raising=False)
    app = create_fastapi_app(_runtime(tmp_path))
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/v1/node0/action-intent",
        json={
            "action_type": "copy_text",
            "target": "do not copy without auth",
            "consent": True,
        },
    )

    assert response.status_code in {401, 503}
