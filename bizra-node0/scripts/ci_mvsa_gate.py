#!/usr/bin/env python3
"""
BIZRA CI MVSA Gate
==================

Validates Node0 MVSA readiness as a CI pipeline gate. Reports structured JSON
with pass/fail per gate, overall decision, and $GITHUB_OUTPUT integration.

Standing on Giants:
- Deming (1950): PDCA — this IS the Check gate
- Juran (1951): Quality ratcheting — no regression past MVSA compliance
- PMBOK 7th Ed (2021): Quality Management — gate-based release control
- Boyd (1976): OODA loop — observe (scan) → orient (score) → decide → act

Architecture:
┌──────────────────────────────────────────────────────────────────────────┐
│ SCHEMA CHECK → AUTHORITY CHECK → LIFECYCLE CHECK → GATE REPORT          │
└──────────────────────────────────────────────────────────────────────────┘

Exit Codes:
    0 - All MVSA gates pass
    1 - One or more gates failed
    3 - Configuration/import error

Usage:
    # Basic gate check (CI default)
    python scripts/ci_mvsa_gate.py --project-root .

    # With JSON report
    python scripts/ci_mvsa_gate.py --project-root . --report /tmp/mvsa_gate.json

    # With GitHub Actions output
    python scripts/ci_mvsa_gate.py --project-root . --github-output "$GITHUB_OUTPUT"
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Ensure repo root is importable
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ═══════════════════════════════════════════════════════════════════════════════
# Gate Model
# ═══════════════════════════════════════════════════════════════════════════════

LIFECYCLE_V2_GATES = [
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
]

LIFECYCLE_V2_REQUIRED_SECTIONS = [
    "schema_version", "updated_at", "status", "ok", "ready",
    "node_id", "origin", "identity", "artifacts", "gates",
    "mvsa", "mission", "restart_recovery", "compat",
]

MVSA_PROOF_REQUIRED_FIELDS = [
    "schema_version", "generated_at", "node_id", "genesis_hash",
    "genesis_hash_valid", "network", "consensus", "status", "reason_code",
]

AUTHORITY_MIGRATION_REQUIRED_FIELDS = [
    "schema_version", "migrated_at", "source_path", "source_kind",
    "result", "reason_code", "genesis_hash",
]


@dataclass
class GateCheck:
    """Single gate check result."""
    name: str
    passed: bool
    detail: str = ""
    severity: str = "error"


@dataclass
class MvsaGateReport:
    """Full MVSA gate evaluation report."""
    timestamp: str = ""
    gate_passed: bool = False
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    checks: List[Dict[str, Any]] = field(default_factory=list)
    authority_status: str = "unknown"
    lifecycle_status: str = "unknown"
    proof_status: str = "unknown"
    schema_compliance: bool = False

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


# ═══════════════════════════════════════════════════════════════════════════════
# Gate Checks
# ═══════════════════════════════════════════════════════════════════════════════

def _check_module_importable(module_name: str, check_name: str) -> GateCheck:
    """Verify a Python module is importable."""
    try:
        importlib.import_module(module_name)
        return GateCheck(name=check_name, passed=True, detail=f"{module_name} importable")
    except ImportError as exc:
        return GateCheck(
            name=check_name, passed=False,
            detail=f"{module_name} import failed: {exc}",
        )


def _check_authority_module() -> List[GateCheck]:
    """Validate authority resolution module exists and is well-formed."""
    checks: List[GateCheck] = []

    checks.append(_check_module_importable(
        "core.sovereign.node0_authority", "authority_module_importable"
    ))
    checks.append(_check_module_importable(
        "core.sovereign.atomic_io", "atomic_io_module_importable"
    ))
    checks.append(_check_module_importable(
        "core.sovereign.node0_mvsa", "mvsa_module_importable"
    ))

    # Check exported symbols
    try:
        from core.sovereign.node0_authority import (  # noqa: F401
            AuthorityResult,
            RESULT_BLOCKED,
            RESULT_CANONICAL,
            RESULT_MIGRATED,
            require_authority,
            resolve_authority,
        )
        checks.append(GateCheck(
            name="authority_api_complete", passed=True,
            detail="All required exports present",
        ))
    except ImportError as exc:
        checks.append(GateCheck(
            name="authority_api_complete", passed=False,
            detail=f"Missing exports: {exc}",
        ))

    return checks


def _check_lifecycle_v2_schema(project_root: Path) -> List[GateCheck]:
    """Validate lifecycle v2 schema structure in the standalone manager."""
    checks: List[GateCheck] = []

    try:
        from scripts.node0_standalone import Node0StandaloneManager
        checks.append(GateCheck(
            name="standalone_manager_importable", passed=True,
            detail="Node0StandaloneManager importable",
        ))
    except ImportError as exc:
        checks.append(GateCheck(
            name="standalone_manager_importable", passed=False,
            detail=f"Import failed: {exc}",
        ))
        return checks

    # Verify _compute_status exists and handles the 3-tier model
    manager = Node0StandaloneManager(project_root=project_root)

    # Test blocked
    blocked_gates = {g: False for g in LIFECYCLE_V2_GATES}
    status_blocked = manager._compute_status(blocked_gates)
    checks.append(GateCheck(
        name="status_blocked_semantics", passed=(status_blocked == "blocked"),
        detail=f"All-false gates → {status_blocked} (expected blocked)",
    ))

    # Test degraded
    degraded_gates = {g: True for g in LIFECYCLE_V2_GATES[:9]}
    degraded_gates["mission_path_receipted"] = False
    degraded_gates["restart_recovery_ready"] = False
    status_degraded = manager._compute_status(degraded_gates)
    checks.append(GateCheck(
        name="status_degraded_semantics", passed=(status_degraded == "degraded"),
        detail=f"First 9 true, last 2 false → {status_degraded} (expected degraded)",
    ))

    # Test ready
    ready_gates = {g: True for g in LIFECYCLE_V2_GATES}
    status_ready = manager._compute_status(ready_gates)
    checks.append(GateCheck(
        name="status_ready_semantics", passed=(status_ready == "ready"),
        detail=f"All-true gates → {status_ready} (expected ready)",
    ))

    return checks


def _check_rust_binary_source(project_root: Path) -> List[GateCheck]:
    """Validate the Rust MVSA binary source exists and is registered."""
    checks: List[GateCheck] = []

    # Source file exists
    rs_source = project_root / "bizra-omega" / "bizra-resourcepool" / "src" / "bin" / "node0_mvsa.rs"
    checks.append(GateCheck(
        name="rust_mvsa_source_exists",
        passed=rs_source.exists(),
        detail=str(rs_source),
    ))

    # Registered in Cargo.toml
    cargo_toml = project_root / "bizra-omega" / "bizra-resourcepool" / "Cargo.toml"
    if cargo_toml.exists():
        content = cargo_toml.read_text(encoding="utf-8")
        has_bin = 'name = "node0-mvsa"' in content
        checks.append(GateCheck(
            name="rust_mvsa_cargo_registered",
            passed=has_bin,
            detail="[[bin]] node0-mvsa in Cargo.toml" if has_bin else "NOT registered",
        ))
    else:
        checks.append(GateCheck(
            name="rust_mvsa_cargo_registered", passed=False,
            detail="Cargo.toml not found",
        ))

    return checks


def _check_mvsa_proof_schema() -> List[GateCheck]:
    """Validate the proof schema constants are correct."""
    checks: List[GateCheck] = []

    try:
        from core.sovereign.node0_mvsa import PROOF_FILE, run_mvsa_proof  # noqa: F401
        checks.append(GateCheck(
            name="mvsa_proof_module_api", passed=True,
            detail="run_mvsa_proof and PROOF_FILE exported",
        ))
    except ImportError as exc:
        checks.append(GateCheck(
            name="mvsa_proof_module_api", passed=False,
            detail=f"Missing: {exc}",
        ))

    return checks


def _check_cli_surface(project_root: Path) -> List[GateCheck]:
    """Validate the CLI has the prove-mvsa subcommand."""
    checks: List[GateCheck] = []

    try:
        from scripts.node0_standalone import build_parser
        parser = build_parser()
        # Attempt to parse prove-mvsa
        try:
            parser.parse_args(["prove-mvsa"])
            checks.append(GateCheck(
                name="cli_prove_mvsa_subcommand", passed=True,
                detail="prove-mvsa subcommand registered",
            ))
        except SystemExit:
            checks.append(GateCheck(
                name="cli_prove_mvsa_subcommand", passed=False,
                detail="prove-mvsa not recognized by parser",
            ))
    except (ImportError, AttributeError) as exc:
        checks.append(GateCheck(
            name="cli_prove_mvsa_subcommand", passed=False,
            detail=f"Parser not available: {exc}",
        ))

    return checks


def _check_api_routes() -> List[GateCheck]:
    """Validate /mvsa and /prove-mvsa API routes exist."""
    checks: List[GateCheck] = []

    try:
        from scripts.node0_standalone import Node0StandaloneManager, create_app
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            (p / "sovereign_state").mkdir()
            manager = Node0StandaloneManager(project_root=p)
            app = create_app(manager)

            routes = {r.path for r in app.routes}  # type: ignore[union-attr]
            has_mvsa = "/mvsa" in routes
            has_prove = "/prove-mvsa" in routes
            checks.append(GateCheck(
                name="api_mvsa_route", passed=has_mvsa,
                detail="/mvsa route registered" if has_mvsa else "/mvsa MISSING",
            ))
            checks.append(GateCheck(
                name="api_prove_mvsa_route", passed=has_prove,
                detail="/prove-mvsa route registered" if has_prove else "/prove-mvsa MISSING",
            ))
    except Exception as exc:  # noqa: BLE001
        checks.append(GateCheck(
            name="api_mvsa_route", passed=False,
            detail=f"Route check failed: {exc}", severity="warning",
        ))

    return checks


def _check_test_coverage(project_root: Path) -> List[GateCheck]:
    """Validate MVSA test files exist."""
    checks: List[GateCheck] = []

    test_files = [
        ("tests/core/sovereign/test_node0_authority.py", "authority_tests_exist"),
        ("tests/core/sovereign/test_node0_mvsa.py", "mvsa_tests_exist"),
        ("tests/integration/test_mvsa_acceptance.py", "acceptance_tests_exist"),
    ]
    for rel_path, name in test_files:
        exists = (project_root / rel_path).exists()
        checks.append(GateCheck(
            name=name, passed=exists,
            detail=rel_path if exists else f"{rel_path} NOT FOUND",
        ))

    return checks


# ═══════════════════════════════════════════════════════════════════════════════
# Gate Runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_mvsa_gate(project_root: Path) -> MvsaGateReport:
    """Run all MVSA gate checks and produce a structured report."""
    all_checks: List[GateCheck] = []

    # Module checks
    all_checks.extend(_check_authority_module())

    # Schema checks
    all_checks.extend(_check_lifecycle_v2_schema(project_root))

    # Rust source
    all_checks.extend(_check_rust_binary_source(project_root))

    # Proof schema
    all_checks.extend(_check_mvsa_proof_schema())

    # CLI surface
    all_checks.extend(_check_cli_surface(project_root))

    # API routes
    all_checks.extend(_check_api_routes())

    # Test coverage
    all_checks.extend(_check_test_coverage(project_root))

    passed = [c for c in all_checks if c.passed]
    failed = [c for c in all_checks if not c.passed]

    report = MvsaGateReport(
        gate_passed=len(failed) == 0,
        total_checks=len(all_checks),
        passed_checks=len(passed),
        failed_checks=len(failed),
        checks=[asdict(c) for c in all_checks],
        schema_compliance=all(
            c.passed for c in all_checks
            if c.name.endswith("_semantics") or c.name.endswith("_schema")
        ),
    )

    # Determine sub-statuses
    authority_checks = [c for c in all_checks if "authority" in c.name]
    report.authority_status = "pass" if all(c.passed for c in authority_checks) else "fail"

    lifecycle_checks = [c for c in all_checks if "status_" in c.name or "lifecycle" in c.name]
    report.lifecycle_status = "pass" if all(c.passed for c in lifecycle_checks) else "fail"

    proof_checks = [c for c in all_checks if "mvsa" in c.name or "rust" in c.name or "proof" in c.name]
    report.proof_status = "pass" if all(c.passed for c in proof_checks) else "fail"

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description="BIZRA CI MVSA Gate — validates MVSA readiness",
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path("."),
        help="Project root directory (default: cwd)",
    )
    parser.add_argument(
        "--report", type=Path, default=None,
        help="Write JSON report to this path",
    )
    parser.add_argument(
        "--github-output", type=str, default=None,
        help="Path to $GITHUB_OUTPUT file for CI integration",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output JSON to stdout",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    report = run_mvsa_gate(project_root)

    # Write report
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(asdict(report), indent=2), encoding="utf-8",
        )

    # GitHub Actions output
    if args.github_output:
        gh_path = Path(args.github_output)
        with gh_path.open("a", encoding="utf-8") as fh:
            fh.write(f"mvsa_gate_passed={str(report.gate_passed).lower()}\n")
            fh.write(f"mvsa_total_checks={report.total_checks}\n")
            fh.write(f"mvsa_passed_checks={report.passed_checks}\n")
            fh.write(f"mvsa_failed_checks={report.failed_checks}\n")
            fh.write(f"mvsa_authority_status={report.authority_status}\n")
            fh.write(f"mvsa_lifecycle_status={report.lifecycle_status}\n")
            fh.write(f"mvsa_proof_status={report.proof_status}\n")
            fh.write(f"mvsa_schema_compliance={str(report.schema_compliance).lower()}\n")

    # Console output
    if args.json:
        print(json.dumps(asdict(report), indent=2))
    else:
        status = "✅ PASS" if report.gate_passed else "❌ FAIL"
        print(f"MVSA Gate: {status} ({report.passed_checks}/{report.total_checks})")
        print(f"  Authority: {report.authority_status}")
        print(f"  Lifecycle: {report.lifecycle_status}")
        print(f"  Proof:     {report.proof_status}")
        if report.failed_checks > 0:
            print(f"\nFailed checks ({report.failed_checks}):")
            for c in report.checks:
                if not c["passed"]:
                    print(f"  ✗ {c['name']}: {c['detail']}")

    return 0 if report.gate_passed else 1


if __name__ == "__main__":
    sys.exit(main())
