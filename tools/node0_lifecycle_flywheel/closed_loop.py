"""Node0 Closed-Loop Lifecycle Flywheel.

This harness binds the existing Node0 standalone lifecycle, audit flywheel,
and execution flywheel into one deterministic operator receipt:

    Observe -> Guard -> Prioritize -> Recommend -> Optionally Act -> Recheck
    -> Receipt -> Encode

Default mode is read-only and non-destructive. The only mutating path is the
explicit ``--execute-next`` CLI flag, which runs the single recommended
``scripts/node0_standalone.py`` command.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


STATUS_GATES = (
    "genesis_authority_valid",
    "identity_ready",
    "pat_sat_ready",
    "urp_signed",
    "urp_verified",
    "assets_written",
    "awareness_written",
    "mvsa_network_bootstrap_ok",
    "mvsa_self_validation_ok",
    "mission_path_receipted",
    "restart_recovery_ready",
)

AVAILABILITY_GATES = (
    "desktop_bridge_reachable",
    "mcp_available",
    "a2a_available",
    "telescript_available",
)

DEFAULT_AUDIT_DIR = (
    Path("docs")
    / "audits"
    / "omnidirectional_hyperdimensional_audit_v0_1"
    / "artifacts"
)

DEFAULT_PROBE_TASK = (
    "write file missions/node0_closed_loop_probe.txt :: "
    "node0 closed loop proof"
)


@dataclass(frozen=True)
class LoopDecision:
    decision_id: str
    title: str
    stage: str
    rationale: str
    command: list[str] = field(default_factory=list)
    operator_commands: list[str] = field(default_factory=list)
    guards: list[str] = field(default_factory=list)
    blocked_by: list[str] = field(default_factory=list)
    exit_code_if_strict: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _bool_map(raw: dict[str, Any], keys: Iterable[str]) -> dict[str, bool]:
    return {key: bool(raw.get(key, False)) for key in keys}


def _node0_command(project_root: Path, *args: str) -> list[str]:
    return [
        sys.executable,
        str(project_root / "scripts" / "node0_standalone.py"),
        *args,
    ]


def _display_command(command: list[str], project_root: Path) -> str:
    if not command:
        return ""
    display = list(command)
    if Path(display[0]).name.startswith("python"):
        display[0] = "python"
    script = str(project_root / "scripts" / "node0_standalone.py")
    if len(display) > 1 and display[1] == script:
        display[1] = "scripts/node0_standalone.py"
    return " ".join(shlex.quote(part) for part in display)


def _artifact_presence(project_root: Path) -> dict[str, bool]:
    state_root = project_root / "sovereign_state"
    return {
        "state_root": state_root.exists(),
        "node0_genesis": (state_root / "node0_genesis.json").exists(),
        "genesis_hash": (state_root / "genesis_hash.txt").exists(),
        "mvsa_proof": (state_root / "node0_mvsa_proof.json").exists(),
        "assets": (state_root / "node0_assets.json").exists(),
        "pat_awareness": (state_root / "pat_awareness.json").exists(),
        "urp_pledge": (state_root / "urp_pledge.json").exists(),
        "lifecycle": (state_root / "node0_lifecycle.json").exists(),
    }


def _audit_report(
    project_root: Path,
    audit_dir: Path | None,
    changed_paths: Iterable[str],
) -> dict[str, Any]:
    resolved = project_root / (audit_dir or DEFAULT_AUDIT_DIR)
    if audit_dir is not None and audit_dir.is_absolute():
        resolved = audit_dir

    if resolved.exists():
        try:
            from tools.audit.flywheel_kernel.kernel import build_report

            return build_report(resolved, changed_paths=changed_paths)
        except Exception as exc:  # noqa: BLE001 - receipt should degrade visibly
            return {
                "schema": "bizra.flywheel.kernel_report.v1",
                "error": f"{type(exc).__name__}: {exc}",
                "priority": {
                    "priority_id": "P-BOOTSTRAP-AUDIT",
                    "title": "Audit flywheel unavailable",
                    "rationale": "Audit artifacts exist but could not be loaded.",
                    "next_actions": ["Fix audit artifact schema or kernel import."],
                    "blocked_by": ["AUDIT_LOAD_ERROR"],
                },
                "guards": [
                    {
                        "guard_id": "AUDIT_LOAD_ERROR",
                        "status": "BLOCK",
                        "signal": "Audit flywheel could not load artifacts.",
                    }
                ],
            }

    return {
        "schema": "bizra.flywheel.kernel_report.v1",
        "state": {"audit_dir": str(resolved), "missing_artifacts": ["audit_dir"]},
        "priority": {
            "priority_id": "P-BOOTSTRAP-AUDIT",
            "title": "Generate audit artifacts",
            "rationale": "No current audit artifacts were found for global guard state.",
            "next_actions": [
                "Run the omni audit in no-network mode.",
                "Re-run this lifecycle flywheel receipt.",
            ],
            "blocked_by": ["G-FW-001"],
        },
        "guards": [
            {
                "guard_id": "G-FW-001",
                "status": "BLOCK",
                "signal": "Required audit artifacts are missing.",
                "evidence": ["audit_dir"],
            }
        ],
        "triggered_patterns": [],
        "pattern_count": 0,
    }


def _priority_context(
    lifecycle_gates: dict[str, bool],
    audit_report: dict[str, Any],
) -> dict[str, Any]:
    audit_state = audit_report.get("state", {})
    claim_counts = audit_state.get("claim_counts", {})
    code_risks = audit_state.get("code_risk_counts", {})
    dep_gaps = audit_state.get("dep_gaps", [])
    blocked_rows = sum(1 for gate in STATUS_GATES if not lifecycle_gates.get(gate))

    return {
        "secret_findings": int(audit_state.get("secret_count", 0) or 0),
        "rotation_required": False,
        "runtime_defaults_insecure": bool(
            code_risks.get("PY_SHELL_TRUE", 0)
            or code_risks.get("PY_EVAL_EXEC", 0)
            or code_risks.get("RS_PANIC", 0)
            or code_risks.get("RS_UNWRAP", 0)
        ),
        "main_branch_red": False,
        "ci_failing_count": 0,
        "dependency_vulnerabilities": 0,
        "sbom_stale": bool(dep_gaps),
        "public_claims_risky": bool(
            claim_counts.get("PROHIBITED", 0)
            or claim_counts.get("NEEDS_REWRITE", 0)
            or claim_counts.get("PROOF_REQUIRED", 0)
        ),
        "node0_activation_blocked_rows": blocked_rows,
    }


def _execution_priority(context: dict[str, Any]) -> dict[str, Any]:
    try:
        from tools.execution_flywheel.priority_engine import recommend_priority

        return recommend_priority(context).to_dict()
    except Exception as exc:  # noqa: BLE001 - receipt should degrade visibly
        return {
            "priority": "NODE0_ACTIVATION",
            "reason": f"Execution priority engine unavailable: {type(exc).__name__}: {exc}",
            "confidence": 0.25,
            "evidence": ["fallback_priority_context"],
        }


def load_state(
    project_root: Path | str,
    *,
    audit_dir: Path | None = None,
    changed_paths: Iterable[str] = (),
    mcp_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load observable Node0 state without mutating the workspace."""

    root = Path(project_root).resolve()
    state_root = root / "sovereign_state"
    lifecycle_path = state_root / "node0_lifecycle.json"
    mvsa_path = state_root / "node0_mvsa_proof.json"

    lifecycle = _read_json(lifecycle_path, {})
    gates_raw = lifecycle.get("gates", {}) if isinstance(lifecycle, dict) else {}
    status_gates = _bool_map(gates_raw, STATUS_GATES)
    availability_gates = _bool_map(gates_raw, AVAILABILITY_GATES)
    blocked_gates = [gate for gate in STATUS_GATES if not status_gates.get(gate)]
    mvsa = lifecycle.get("mvsa", {}) if isinstance(lifecycle, dict) else {}
    persisted_mvsa = _read_json(mvsa_path, {})
    if not mvsa and isinstance(persisted_mvsa, dict):
        mvsa = persisted_mvsa

    audit = _audit_report(root, audit_dir, changed_paths)
    priority_context = _priority_context(status_gates, audit)
    execution_priority = _execution_priority(priority_context)

    status = str(lifecycle.get("status", "blocked" if blocked_gates else "ready"))
    ready = bool(lifecycle.get("ready", status == "ready" and not blocked_gates))

    return {
        "project_root": str(root),
        "generated_at": _utc_now(),
        "state_root": str(state_root),
        "lifecycle_path": str(lifecycle_path),
        "lifecycle_exists": lifecycle_path.exists(),
        "lifecycle_status": status,
        "lifecycle_ready": ready,
        "node_id": lifecycle.get("node_id", "unknown"),
        "status_gates": status_gates,
        "availability_gates": availability_gates,
        "blocked_gates": blocked_gates,
        "artifact_presence": _artifact_presence(root),
        "mvsa_status": str(
            mvsa.get("status", "missing") if isinstance(mvsa, dict) else "missing"
        ),
        "mission": lifecycle.get("mission", {}) if isinstance(lifecycle, dict) else {},
        "restart_recovery": (
            lifecycle.get("restart_recovery", {}) if isinstance(lifecycle, dict) else {}
        ),
        "audit": {
            "priority": audit.get("priority", {}),
            "guards": audit.get("guards", []),
            "triggered_patterns": audit.get("triggered_patterns", []),
            "pattern_count": audit.get("pattern_count", 0),
        },
        "priority_context": priority_context,
        "execution_priority": execution_priority,
        "mcp_context": mcp_context or {},
    }


def decide_next_action(
    state: dict[str, Any],
    *,
    project_root: Path | str,
    architect: str = "MoMo",
    probe_task: str = DEFAULT_PROBE_TASK,
) -> LoopDecision:
    """Choose the next Node0 lifecycle action from observable state."""

    root = Path(project_root).resolve()
    gates = state.get("status_gates", {})
    blocked = list(state.get("blocked_gates", []))
    audit_priority = state.get("audit", {}).get("priority", {})
    audit_blocks = [
        str(guard.get("guard_id"))
        for guard in state.get("audit", {}).get("guards", [])
        if guard.get("status") == "BLOCK"
    ]
    common_guards = [
        "read_only_by_default",
        "execute_next_required_for_mutation",
        "status_determined_by_11_lifecycle_v2_gates",
    ]
    if audit_blocks:
        common_guards.append(
            "global_audit_blocks_present:" + ",".join(sorted(audit_blocks))
        )

    if not state.get("lifecycle_exists") or not gates.get("genesis_authority_valid"):
        command = _node0_command(root, "activate", "--architect", architect)
        return LoopDecision(
            decision_id="NODE0_ACTIVATE",
            title="Resolve authority and activate Node0 MVSA base",
            stage="genesis_authority",
            rationale=(
                "Lifecycle v2 is missing or canonical genesis authority has not "
                "been proven. Node0 cannot close the loop until authority, "
                "identity, URP, assets, awareness, and MVSA bootstrap are written."
            ),
            command=command,
            operator_commands=[_display_command(command, root)],
            guards=common_guards + ["canonical_authority_fail_closed"],
            blocked_by=blocked,
            exit_code_if_strict=3,
        )

    if not gates.get("mvsa_network_bootstrap_ok") or not gates.get(
        "mvsa_self_validation_ok"
    ):
        command = _node0_command(root, "prove-mvsa")
        return LoopDecision(
            decision_id="NODE0_PROVE_MVSA",
            title="Run Rust-backed MVSA proof",
            stage="mvsa_proof",
            rationale=(
                "Authority exists, but loopback bootstrap or self-validation "
                "is not proven. The next closure step is a fresh MVSA proof."
            ),
            command=command,
            operator_commands=[_display_command(command, root)],
            guards=common_guards + ["mvsa_must_be_rust_backed"],
            blocked_by=blocked,
            exit_code_if_strict=3,
        )

    if not gates.get("mission_path_receipted"):
        command = _node0_command(
            root,
            "task",
            probe_task,
            "--source",
            "node0_lifecycle_flywheel",
            "--browser-mode",
            "mock",
        )
        return LoopDecision(
            decision_id="NODE0_RECEIPT_MISSION",
            title="Run one receipted mission through the standalone path",
            stage="mission_receipt",
            rationale=(
                "MVSA is established, but the autonomous mission path has not "
                "emitted an evidence receipt. A small filesystem probe closes "
                "the task -> receipt -> lifecycle gate."
            ),
            command=command,
            operator_commands=[_display_command(command, root)],
            guards=common_guards
            + [
                "probe_task_is_workspace_scoped",
                "browser_mode_mock_for_no_network_default",
            ],
            blocked_by=blocked,
            exit_code_if_strict=2,
        )

    if not gates.get("restart_recovery_ready"):
        command = _node0_command(root, "prove-mvsa")
        return LoopDecision(
            decision_id="NODE0_REFRESH_RESTART_RECOVERY",
            title="Refresh persisted restart recovery gate",
            stage="restart_recovery",
            rationale=(
                "Mission receipts exist, but restart recovery is not marked "
                "ready. A mutating proof refresh recomputes artifact presence "
                "from a fresh process and updates lifecycle v2."
            ),
            command=command,
            operator_commands=[_display_command(command, root)],
            guards=common_guards + ["fresh_process_recovery_required"],
            blocked_by=blocked,
            exit_code_if_strict=2,
        )

    if state.get("lifecycle_ready") and not blocked:
        command = _node0_command(root, "health")
        return LoopDecision(
            decision_id="NODE0_MONITOR_AND_RELOOP",
            title="Node0 lifecycle is closed; monitor and re-loop",
            stage="monitor",
            rationale=(
                "All 11 lifecycle gates are satisfied. The next action is "
                "health observation, audit refresh, and pattern encoding for "
                "any new repeated failure."
            ),
            command=command,
            operator_commands=[
                _display_command(command, root),
                "python -m tools.node0_lifecycle_flywheel.closed_loop",
            ],
            guards=common_guards
            + [
                "no_expansion_before_next_health_receipt",
                f"audit_priority={audit_priority.get('priority_id', 'unknown')}",
            ],
            blocked_by=[],
            exit_code_if_strict=0,
        )

    command = _node0_command(root, "health")
    return LoopDecision(
        decision_id="NODE0_RECONCILE_DEGRADED_STATE",
        title="Reconcile degraded lifecycle state",
        stage="reconcile",
        rationale=(
            "Lifecycle state is degraded but no canonical stage-specific gate "
            "explains the blocker. Read health, inspect lifecycle JSON, and "
            "encode the new failure as a flywheel pattern if it repeats."
        ),
        command=command,
        operator_commands=[_display_command(command, root)],
        guards=common_guards + ["manual_reconciliation_required"],
        blocked_by=blocked,
        exit_code_if_strict=2,
    )


def _execute_command(
    command: list[str],
    project_root: Path,
    timeout_s: int,
) -> dict[str, Any]:
    if not command:
        return {"executed": False, "reason": "no_command"}
    t0 = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=str(project_root),
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        return {
            "executed": True,
            "command": _display_command(command, project_root),
            "returncode": completed.returncode,
            "duration_ms": round((time.perf_counter() - t0) * 1000, 2),
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "executed": True,
            "command": _display_command(command, project_root),
            "returncode": 124,
            "duration_ms": round((time.perf_counter() - t0) * 1000, 2),
            "stdout_tail": (
                (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else ""
            ),
            "stderr_tail": (
                (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else ""
            ),
            "error": "timeout",
        }


def build_receipt(
    project_root: Path | str = ".",
    *,
    audit_dir: Path | None = None,
    changed_paths: Iterable[str] = (),
    mcp_context: dict[str, Any] | None = None,
    architect: str = "MoMo",
    probe_task: str = DEFAULT_PROBE_TASK,
    execute_next: bool = False,
    timeout_s: int = 120,
) -> dict[str, Any]:
    """Build a machine-readable closed-loop receipt."""

    root = Path(project_root).resolve()
    before = load_state(
        root,
        audit_dir=audit_dir,
        changed_paths=changed_paths,
        mcp_context=mcp_context,
    )
    decision = decide_next_action(
        before,
        project_root=root,
        architect=architect,
        probe_task=probe_task,
    )

    execution = {"executed": False, "reason": "dry_run_default"}
    after: dict[str, Any] | None = None
    if execute_next:
        execution = _execute_command(decision.command, root, timeout_s)
        after = load_state(
            root,
            audit_dir=audit_dir,
            changed_paths=changed_paths,
            mcp_context=mcp_context,
        )

    return {
        "schema": "bizra.node0.lifecycle_flywheel.receipt.v1",
        "generated_at": _utc_now(),
        "loop": [
            "Observe",
            "Guard",
            "Prioritize",
            "Recommend",
            "Act(optional)",
            "Recheck",
            "Receipt",
            "Encode",
        ],
        "mode": "execute_next" if execute_next else "dry_run",
        "state": before,
        "decision": decision.to_dict(),
        "execution": execution,
        "post_state": after,
        "next_encoding_rule": (
            "If the same blocked gate appears in two consecutive receipts, "
            "add or refine a tools/execution_flywheel pattern with a paired test."
        ),
    }


def _load_mcp_context(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    payload = _read_json(Path(path), {})
    return payload if isinstance(payload, dict) else {"value": payload}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="node0-lifecycle-flywheel",
        description="Build a BIZRA Node0 closed-loop lifecycle receipt.",
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--audit-dir", default=str(DEFAULT_AUDIT_DIR))
    parser.add_argument("--changed-path", action="append", default=[])
    parser.add_argument("--mcp-context")
    parser.add_argument("--architect", default="MoMo")
    parser.add_argument("--probe-task", default=DEFAULT_PROBE_TASK)
    parser.add_argument("--execute-next", action="store_true")
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--out")
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = Path(args.project_root).resolve()
    audit_dir = Path(args.audit_dir) if args.audit_dir else None
    mcp_context = _load_mcp_context(args.mcp_context)

    receipt = build_receipt(
        root,
        audit_dir=audit_dir,
        changed_paths=args.changed_path,
        mcp_context=mcp_context,
        architect=args.architect,
        probe_task=args.probe_task,
        execute_next=args.execute_next,
        timeout_s=args.timeout_s,
    )
    text = json.dumps(receipt, indent=2, ensure_ascii=True)
    if args.out:
        _write_json(Path(args.out), receipt)
    print(text)

    execution = receipt.get("execution", {})
    if args.execute_next and execution.get("returncode", 0) not in (0, 2):
        return int(execution.get("returncode", 1) or 1)
    if args.strict:
        return int(receipt["decision"].get("exit_code_if_strict", 0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
