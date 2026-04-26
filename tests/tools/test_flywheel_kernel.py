import json
from pathlib import Path

from tools.audit.flywheel_kernel.kernel import (
    build_report,
    decide_priority,
    evaluate_guards,
    load_audit_state,
    should_trigger_audit,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _audit_dir(tmp_path: Path, *, secrets: int = 0) -> Path:
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    _write_json(
        audit_dir / "audit_summary.json",
        {
            "counts": {"secrets": secrets, "findings": 17},
            "website_captures": [],
        },
    )
    _write_json(
        audit_dir / "secret_findings.json",
        [{"finding_id": "S0001"}] * secrets,
    )
    _write_json(
        audit_dir / "claims_register.json",
        [
            {"classification": "PROHIBITED"},
            {"classification": "NEEDS_REWRITE"},
            {"classification": "PROOF_REQUIRED"},
        ],
    )
    _write_json(
        audit_dir / "code_risks.json",
        [
            {"rule": "RS_UNWRAP"},
            {"rule": "PY_SHELL_TRUE"},
        ],
    )
    _write_json(
        audit_dir / "dependencies.json",
        {"gaps": ["Workspace without Cargo.lock: filedfs/Cargo.toml"]},
    )
    return audit_dir


def test_priority_shifts_to_truth_integrity_when_secrets_are_clear(
    tmp_path: Path,
) -> None:
    state = load_audit_state(_audit_dir(tmp_path, secrets=0))
    guards = evaluate_guards(state)
    decision = decide_priority(state, guards)

    assert state.secret_count == 0
    assert decision.priority_id == "P1_TRUTH_INTEGRITY"
    assert "G-FW-003" in decision.blocked_by


def test_secret_findings_keep_priority_at_p0(tmp_path: Path) -> None:
    state = load_audit_state(_audit_dir(tmp_path, secrets=2))
    decision = decide_priority(state, evaluate_guards(state))

    assert decision.priority_id == "P0_SECRET_TRIAGE"
    assert "G-FW-002" in decision.blocked_by


def test_missing_artifacts_returns_bootstrap(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    state = load_audit_state(empty)
    decision = decide_priority(state, evaluate_guards(state))
    assert decision.priority_id == "P-BOOTSTRAP-AUDIT"
    assert "G-FW-001" in decision.blocked_by


def test_changed_paths_trigger_pattern_specific_audit() -> None:
    triggers = should_trigger_audit(
        [
            "runtime/core/autoconfig.py",
            "docs/brand/public_launch_readiness/PUBLIC_CLAIMS_REGISTER.md",
        ]
    )

    pattern_ids = {trigger["pattern_id"] for trigger in triggers}

    assert "FW-P002" in pattern_ids
    assert "FW-P004" in pattern_ids


def test_build_report_is_machine_readable(tmp_path: Path) -> None:
    report = build_report(
        _audit_dir(tmp_path, secrets=0),
        changed_paths=["tools/audit/omni_audit/run_audit.py"],
    )

    assert report["schema"] == "bizra.flywheel.kernel_report.v1"
    assert report["priority"]["priority_id"] == "P1_TRUTH_INTEGRITY"
    assert report["pattern_count"] >= 6
    assert report["triggered_patterns"][0]["pattern_id"] == "FW-P003"


def test_strict_mode_returns_block_exit_code(tmp_path: Path) -> None:
    from tools.audit.flywheel_kernel.kernel import main as kernel_main

    audit_dir = _audit_dir(tmp_path, secrets=3)
    out_path = tmp_path / "report.json"
    rc = kernel_main(
        [
            "--audit-dir",
            str(audit_dir),
            "--out",
            str(out_path),
            "--strict",
        ]
    )
    assert rc == 2
    assert out_path.exists()


def test_strict_mode_passes_when_no_block_guards(tmp_path: Path) -> None:
    from tools.audit.flywheel_kernel.kernel import main as kernel_main

    audit_dir = tmp_path / "clean"
    audit_dir.mkdir()
    _write_json(audit_dir / "audit_summary.json", {"counts": {"secrets": 0}})
    _write_json(audit_dir / "secret_findings.json", [])
    _write_json(audit_dir / "claims_register.json", [])
    _write_json(audit_dir / "code_risks.json", [])
    _write_json(audit_dir / "dependencies.json", {"gaps": []})

    rc = kernel_main(["--audit-dir", str(audit_dir), "--strict"])
    assert rc == 0
