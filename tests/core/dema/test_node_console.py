"""Dema Node Console dependency contract tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from core.dema.node0_status import (
    NODE_CONSOLE_FORBIDDEN_ACTIONS,
    NodeConsoleDependencyId,
    NodeConsoleDependencyStatus,
    build_node_console_status,
    read_node0_dema_status,
)


def _by_id(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    dependencies = payload["dependencies"]
    assert isinstance(dependencies, list)
    return {str(item["id"]): item for item in dependencies}


def test_node_console_blocks_missing_runtime_dependencies():
    status = build_node_console_status(
        python_venv=False,
        pyo3_bridge=False,
        rust_bus_available=False,
        model_backend_connected=False,
        loaded_model_count=0,
        token_visible=False,
        auth_required=True,
        daemon_running=False,
        evidence_ledger_observable=False,
    ).to_dict()
    dependencies = _by_id(status)

    assert status["ready"] is False
    assert status["activation_gate"] == "EXPLICIT_GO_REQUIRED"
    assert "mission_dispatch" in NODE_CONSOLE_FORBIDDEN_ACTIONS
    assert dependencies[NodeConsoleDependencyId.PYTHON_VENV.value]["status"] == (
        NodeConsoleDependencyStatus.BLOCKED.value
    )
    assert dependencies[NodeConsoleDependencyId.PYO3_BRIDGE.value]["status"] == (
        NodeConsoleDependencyStatus.BLOCKED.value
    )
    assert dependencies[NodeConsoleDependencyId.MODEL_BACKEND.value]["status"] == (
        NodeConsoleDependencyStatus.BLOCKED.value
    )
    assert (
        dependencies[NodeConsoleDependencyId.TOKEN_CURRENT_PROCESS.value]["status"]
        == NodeConsoleDependencyStatus.BLOCKED.value
    )


def test_node_console_warnings_do_not_grant_runtime_activation():
    status = build_node_console_status(
        python_venv=True,
        pyo3_bridge=True,
        rust_bus_available=True,
        model_backend_connected=True,
        loaded_model_count=1,
        token_visible=False,
        auth_required=False,
        daemon_running=True,
        evidence_ledger_observable=False,
    ).to_dict()
    dependencies = _by_id(status)

    assert status["ready"] is True
    assert status["activation_gate"] == "EXPLICIT_GO_REQUIRED"
    assert (
        dependencies[NodeConsoleDependencyId.TOKEN_CURRENT_PROCESS.value]["status"]
        == NodeConsoleDependencyStatus.WARNING.value
    )
    assert dependencies[NodeConsoleDependencyId.DAEMON_STATE.value]["status"] == (
        NodeConsoleDependencyStatus.WARNING.value
    )
    assert dependencies[NodeConsoleDependencyId.EVIDENCE_LEDGER.value]["status"] == (
        NodeConsoleDependencyStatus.WARNING.value
    )


def test_read_node0_dema_status_embeds_node_console_without_creating_root(
    tmp_path: Path,
):
    root = tmp_path / "dema"
    fake_lm = {
        "connected": True,
        "auth_required": False,
        "token_present": True,
        "model_count": 1,
        "loaded_count": 1,
        "model_ids": ["qwen/qwen3.5-9b"],
        "loaded_model_ids": ["qwen/qwen3.5-9b"],
        "load_state_known": True,
        "attempts": [],
    }
    fake_status = {
        "kind": "dema_service_status",
        "running": False,
        "status": "stopped",
        "profile_present": True,
        "mission_truth_label": "MEASURED",
        "last_tick": None,
    }
    fake_doctor = {"healthy": True, "findings": []}

    with (
        patch("core.dema.node0_status.probe_lm_studio", return_value=fake_lm),
        patch("core.dema.node0_status.python_venv_active", return_value=True),
        patch("core.dema.node0_status.pyo3_bridge_importable", return_value=True),
        patch(
            "core.dema.node0_status._evidence_ledger_observable",
            return_value=True,
        ),
        patch(
            "core.dema.node0_status.dema_service.cmd_status", return_value=fake_status
        ),
        patch(
            "core.dema.node0_status.dema_service.cmd_doctor", return_value=fake_doctor
        ),
        patch("core.dema.node0_status.current_gap_status", return_value={}),
    ):
        report = read_node0_dema_status(root)

    assert not root.exists()
    assert report["ready"] is True
    assert report["dema_node_console"]["kind"] == "dema_node_console_status"
    assert report["dema_node_console"]["ready"] is True
    assert report["dema_node_console"]["forbidden_actions"] == (
        NODE_CONSOLE_FORBIDDEN_ACTIONS
    )
