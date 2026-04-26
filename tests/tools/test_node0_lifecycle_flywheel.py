from __future__ import annotations

import json
from pathlib import Path

from tools.node0_lifecycle_flywheel.closed_loop import (
    STATUS_GATES,
    build_receipt,
    decide_next_action,
    load_state,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _lifecycle(status: str, gates: dict[str, bool]) -> dict[str, object]:
    normalized = {gate: bool(gates.get(gate, False)) for gate in STATUS_GATES}
    return {
        "schema_version": "2.0.0",
        "updated_at": "2026-04-25T00:00:00Z",
        "status": status,
        "ok": status != "blocked",
        "ready": status == "ready",
        "node_id": "node0-test",
        "origin": {"authority_source": "canonical_genesis"},
        "identity": {"pat_agents": 7, "sat_agents": 5},
        "artifacts": {},
        "gates": normalized,
        "mvsa": {"status": "ready"},
        "mission": {},
        "restart_recovery": {},
        "compat": {},
    }


def test_missing_lifecycle_recommends_activation_without_mutation(tmp_path: Path) -> None:
    receipt = build_receipt(tmp_path)

    assert receipt["mode"] == "dry_run"
    assert receipt["decision"]["decision_id"] == "NODE0_ACTIVATE"
    assert "genesis_authority_valid" in receipt["state"]["blocked_gates"]
    assert not (tmp_path / "sovereign_state").exists()


def test_mvsa_ready_without_mission_recommends_receipted_probe(
    tmp_path: Path,
) -> None:
    gates = {gate: True for gate in STATUS_GATES}
    gates["mission_path_receipted"] = False
    gates["restart_recovery_ready"] = False
    _write_json(
        tmp_path / "sovereign_state" / "node0_lifecycle.json",
        _lifecycle("degraded", gates),
    )

    state = load_state(tmp_path)
    decision = decide_next_action(state, project_root=tmp_path)

    assert decision.decision_id == "NODE0_RECEIPT_MISSION"
    assert "task" in decision.command
    assert "--browser-mode" in decision.command
    assert "mock" in decision.command


def test_mission_without_restart_recommends_recovery_refresh(tmp_path: Path) -> None:
    gates = {gate: True for gate in STATUS_GATES}
    gates["restart_recovery_ready"] = False
    _write_json(
        tmp_path / "sovereign_state" / "node0_lifecycle.json",
        _lifecycle("degraded", gates),
    )

    receipt = build_receipt(tmp_path)

    assert receipt["decision"]["decision_id"] == "NODE0_REFRESH_RESTART_RECOVERY"
    assert receipt["decision"]["operator_commands"] == [
        "python scripts/node0_standalone.py prove-mvsa"
    ]


def test_ready_lifecycle_monitors_and_reloops(tmp_path: Path) -> None:
    gates = {gate: True for gate in STATUS_GATES}
    _write_json(
        tmp_path / "sovereign_state" / "node0_lifecycle.json",
        _lifecycle("ready", gates),
    )

    receipt = build_receipt(tmp_path)

    assert receipt["decision"]["decision_id"] == "NODE0_MONITOR_AND_RELOOP"
    assert receipt["decision"]["exit_code_if_strict"] == 0
    assert receipt["state"]["blocked_gates"] == []


def test_audit_state_feeds_execution_priority(tmp_path: Path) -> None:
    audit = tmp_path / "audit"
    _write_json(audit / "audit_summary.json", {"counts": {"secrets": 2}})
    _write_json(audit / "secret_findings.json", [{"finding_id": "S1"}])
    _write_json(audit / "claims_register.json", [])
    _write_json(audit / "code_risks.json", [])
    _write_json(audit / "dependencies.json", {"gaps": []})

    receipt = build_receipt(tmp_path, audit_dir=audit)

    assert receipt["state"]["audit"]["priority"]["priority_id"] == "P0_SECRET_TRIAGE"
    assert receipt["state"]["execution_priority"]["priority"] == "SECURITY"
