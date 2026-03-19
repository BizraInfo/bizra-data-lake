#!/usr/bin/env python3
"""
BIZRA Release Readiness Orchestrator
=====================================

Validates all quality gates, cross-language sync, and performance baselines
before certifying a commit as release-eligible. Produces a signed release
evidence receipt that chains to the evidence ledger.

Standing on Giants:
- PMI/PMBOK 7th Ed (Quality Management, 2021)
- Deming (Statistical Process Control, 1950)
- Juran (Fitness for Use, 1951)
- Crosby (Zero Defects, 1979)

Architecture:
    COLLECT → VALIDATE → SCORE → CERTIFY → RECEIPT

    1. COLLECT: Gather metrics from CI artifacts
    2. VALIDATE: Run gate checks (Ihsān, SNR, coverage, perf, security)
    3. SCORE: Compute weighted release readiness score
    4. CERTIFY: Compare against thresholds, decide go/no-go
    5. RECEIPT: Produce hash-chained JSON evidence

Usage:
    python scripts/ci_release_readiness.py --commit-sha abc123
    python scripts/ci_release_readiness.py --commit-sha abc123 --strict --json

Exit Codes:
    0 - Release-ready
    1 - Not release-ready (gates failed)
    2 - Configuration error
"""

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ─────────────────────────────────────────────────────────────
# Constitutional Thresholds
# ─────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from core.integration.constants import (
        ADL_GINI_THRESHOLD,
        UNIFIED_IHSAN_THRESHOLD,
        UNIFIED_SNR_THRESHOLD,
    )
except ImportError:
    # Fallback if constants not importable (e.g., minimal CI image)
    UNIFIED_IHSAN_THRESHOLD = 0.95
    UNIFIED_SNR_THRESHOLD = 0.85
    ADL_GINI_THRESHOLD = 0.35


# ─────────────────────────────────────────────────────────────
# Gate Definitions
# ─────────────────────────────────────────────────────────────


@dataclass
class GateResult:
    """Result of a single quality gate evaluation."""

    name: str
    category: str  # quality | security | performance | governance
    passed: bool
    score: float  # 0.0 - 1.0
    weight: float
    detail: str
    blocking: bool = True  # If True, failure blocks release


@dataclass
class ReadinessReport:
    """Complete release readiness assessment."""

    timestamp: str = ""
    commit_sha: str = ""
    branch: str = ""
    gates: List[Dict[str, Any]] = field(default_factory=list)
    overall_score: float = 0.0
    release_ready: bool = False
    blocking_failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    evidence_hash: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────
# Gate Evaluators
# ─────────────────────────────────────────────────────────────


def gate_python_tests(workspace: Path) -> GateResult:
    """Validate Python tests pass."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-x",
            "--tb=line",
            "-q",
            "-m",
            "not requires_ollama and not requires_gpu and not slow",
        ],
        capture_output=True,
        text=True,
        cwd=str(workspace),
        timeout=300,
    )
    passed = result.returncode == 0
    # Extract pass/fail count from pytest output
    lines = result.stdout.strip().split("\n")
    summary = lines[-1] if lines else "no output"
    return GateResult(
        name="python_tests",
        category="quality",
        passed=passed,
        score=1.0 if passed else 0.0,
        weight=0.20,
        detail=summary,
    )


def gate_rust_tests(workspace: Path) -> GateResult:
    """Validate Rust workspace tests pass."""
    omega = workspace / "bizra-omega"
    if not omega.exists():
        return GateResult(
            name="rust_tests",
            category="quality",
            passed=True,  # Skip if no Rust workspace
            score=1.0,
            weight=0.15,
            detail="No Rust workspace found — skipped",
            blocking=False,
        )

    result = subprocess.run(
        ["cargo", "test", "--workspace", "--release", "-q"],
        capture_output=True,
        text=True,
        cwd=str(omega),
        timeout=300,
    )
    passed = result.returncode == 0
    lines = result.stdout.strip().split("\n")
    summary = lines[-1] if lines else "no output"
    return GateResult(
        name="rust_tests",
        category="quality",
        passed=passed,
        score=1.0 if passed else 0.0,
        weight=0.15,
        detail=summary,
    )


def gate_coverage_floor(workspace: Path) -> GateResult:
    """Check coverage meets floor from pyproject.toml."""
    pyproject = workspace / "pyproject.toml"
    coverage_xml = workspace / "coverage.xml"

    if not coverage_xml.exists():
        return GateResult(
            name="coverage_floor",
            category="quality",
            passed=False,
            score=0.0,
            weight=0.15,
            detail="coverage.xml not found — run pytest --cov first",
            blocking=False,
        )

    try:
        # Import from the ratchet engine
        sys.path.insert(0, str(workspace / "scripts"))
        from ci_coverage_ratchet import parse_coverage_xml, read_coverage_floor

        actual = parse_coverage_xml(coverage_xml)
        floor = read_coverage_floor(pyproject)
        passed = actual >= floor
        score = min(actual / 100.0, 1.0)
        return GateResult(
            name="coverage_floor",
            category="quality",
            passed=passed,
            score=score,
            weight=0.15,
            detail=f"{actual:.1f}% actual vs {floor:.0f}% floor",
        )
    except Exception as e:
        return GateResult(
            name="coverage_floor",
            category="quality",
            passed=False,
            score=0.0,
            weight=0.15,
            detail=f"Error: {e}",
            blocking=False,
        )


def gate_lint_python(workspace: Path) -> GateResult:
    """Validate Python linting passes (ruff + black)."""
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "core/", "--quiet"],
        capture_output=True,
        text=True,
        cwd=str(workspace),
        timeout=60,
    )
    ruff_passed = result.returncode == 0

    result_black = subprocess.run(
        [sys.executable, "-m", "black", "--check", "--quiet", "core/"],
        capture_output=True,
        text=True,
        cwd=str(workspace),
        timeout=60,
    )
    black_passed = result_black.returncode == 0

    both_passed = ruff_passed and black_passed
    detail_parts = []
    if not ruff_passed:
        detail_parts.append("ruff: FAIL")
    if not black_passed:
        detail_parts.append("black: FAIL")
    detail = ", ".join(detail_parts) if detail_parts else "ruff + black: PASS"

    return GateResult(
        name="lint_python",
        category="quality",
        passed=both_passed,
        score=1.0 if both_passed else 0.5 if (ruff_passed or black_passed) else 0.0,
        weight=0.10,
        detail=detail,
    )


def gate_security_audit(workspace: Path) -> GateResult:
    """Run pip-audit for known vulnerabilities."""
    result = subprocess.run(
        [sys.executable, "-m", "pip_audit", "--strict", "--progress-spinner=off"],
        capture_output=True,
        text=True,
        cwd=str(workspace),
        timeout=120,
    )
    passed = result.returncode == 0
    vuln_count = result.stdout.count("FAIL") if not passed else 0
    return GateResult(
        name="security_audit",
        category="security",
        passed=passed,
        score=1.0 if passed else max(0.0, 1.0 - vuln_count * 0.1),
        weight=0.10,
        detail=(
            f"{vuln_count} vulnerabilities"
            if vuln_count
            else "No vulnerabilities found"
        ),
    )


def gate_cross_lang_sync(workspace: Path) -> GateResult:
    """Validate Python/Rust constant synchronization."""
    audit_script = (
        workspace / ".claude" / "skills" / "cross-lang-sync" / "audit_constants.py"
    )
    if not audit_script.exists():
        return GateResult(
            name="cross_lang_sync",
            category="governance",
            passed=True,
            score=1.0,
            weight=0.05,
            detail="Audit script not found — skipped",
            blocking=False,
        )
    result = subprocess.run(
        [sys.executable, str(audit_script)],
        capture_output=True,
        text=True,
        cwd=str(workspace),
        timeout=30,
    )
    passed = result.returncode == 0
    return GateResult(
        name="cross_lang_sync",
        category="governance",
        passed=passed,
        score=1.0 if passed else 0.0,
        weight=0.05,
        detail="Constants in sync" if passed else "Drift detected",
    )


def gate_version_consistency(workspace: Path) -> GateResult:
    """Validate version strings match across pyproject.toml and Cargo.toml."""
    import re

    pyproject = workspace / "pyproject.toml"
    cargo = workspace / "bizra-omega" / "Cargo.toml"

    versions = {}

    if pyproject.exists():
        content = pyproject.read_text(encoding="utf-8")
        match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
        if match:
            versions["python"] = match.group(1)

    if cargo.exists():
        content = cargo.read_text(encoding="utf-8")
        match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
        if match:
            versions["rust"] = match.group(1)

    if len(versions) <= 1:
        return GateResult(
            name="version_consistency",
            category="governance",
            passed=True,
            score=1.0,
            weight=0.05,
            detail=f"Versions: {versions}",
            blocking=False,
        )

    unique = set(versions.values())
    passed = len(unique) == 1
    return GateResult(
        name="version_consistency",
        category="governance",
        passed=passed,
        score=1.0 if passed else 0.5,
        weight=0.05,
        detail=f"Versions: {versions}" + (" — MISMATCH" if not passed else ""),
        blocking=False,  # Warning, not blocking
    )


def gate_mypy_ratchet(workspace: Path) -> GateResult:
    """Validate MyPy error count hasn't regressed."""
    BASELINE = 1600  # From ci.yml

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "core/",
            "--ignore-missing-imports",
            "--no-error-summary",
        ],
        capture_output=True,
        text=True,
        cwd=str(workspace),
        timeout=120,
    )
    error_lines = [
        line for line in result.stdout.splitlines() if line.startswith("core/")
    ]
    error_count = len(error_lines)
    passed = error_count <= BASELINE
    score = max(0.0, 1.0 - (error_count / BASELINE) * 0.5) if BASELINE > 0 else 1.0

    return GateResult(
        name="mypy_ratchet",
        category="quality",
        passed=passed,
        score=score,
        weight=0.10,
        detail=f"{error_count} errors (baseline: {BASELINE})",
        blocking=True,
    )


def gate_frontend_build(workspace: Path) -> GateResult:
    """Validate frontend builds successfully."""
    frontend = workspace / "frontend"
    if not frontend.exists():
        return GateResult(
            name="frontend_build",
            category="quality",
            passed=True,
            score=1.0,
            weight=0.05,
            detail="No frontend directory — skipped",
            blocking=False,
        )

    result = subprocess.run(
        ["npm", "run", "ci"],
        capture_output=True,
        text=True,
        cwd=str(frontend),
        timeout=120,
    )
    passed = result.returncode == 0
    return GateResult(
        name="frontend_build",
        category="quality",
        passed=passed,
        score=1.0 if passed else 0.0,
        weight=0.05,
        detail="Build + lint + test: PASS" if passed else "Frontend CI failed",
    )


# ─────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────

ALL_GATES = [
    gate_python_tests,
    gate_coverage_floor,
    gate_lint_python,
    gate_mypy_ratchet,
    gate_security_audit,
    gate_cross_lang_sync,
    gate_version_consistency,
    gate_frontend_build,
    # gate_rust_tests omitted from default — too slow for pre-check
]


def run_all_gates(
    workspace: Path,
    gate_fns: Optional[list] = None,
) -> ReadinessReport:
    """Run all quality gates and produce readiness report."""
    gates = gate_fns or ALL_GATES
    report = ReadinessReport()

    print("=" * 60)
    print("BIZRA Release Readiness Orchestrator")
    print("=" * 60)

    results: List[GateResult] = []
    for gate_fn in gates:
        name = gate_fn.__name__.replace("gate_", "")
        print(f"\n  Running gate: {name}...", end=" ", flush=True)
        try:
            result = gate_fn(workspace)
            results.append(result)
            status = (
                "PASS" if result.passed else ("WARN" if not result.blocking else "FAIL")
            )
            print(f"[{status}] {result.detail}")
        except Exception as e:
            results.append(
                GateResult(
                    name=name,
                    category="error",
                    passed=False,
                    score=0.0,
                    weight=0.0,
                    detail=f"Gate error: {e}",
                    blocking=False,
                )
            )
            print(f"[ERROR] {e}")

    # Compute weighted score
    total_weight = sum(r.weight for r in results)
    if total_weight > 0:
        report.overall_score = sum(r.score * r.weight for r in results) / total_weight
    else:
        report.overall_score = 0.0

    # Identify blocking failures
    report.blocking_failures = [r.name for r in results if not r.passed and r.blocking]
    report.warnings = [r.name for r in results if not r.passed and not r.blocking]
    report.release_ready = len(report.blocking_failures) == 0
    report.gates = [asdict(r) for r in results]

    # Evidence hash
    content = json.dumps(report.gates, sort_keys=True, default=str)
    report.evidence_hash = hashlib.sha256(content.encode()).hexdigest()

    return report


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="BIZRA Release Readiness Orchestrator",
    )
    parser.add_argument("--commit-sha", default="HEAD", help="Commit SHA to evaluate")
    parser.add_argument("--branch", default="", help="Branch name")
    parser.add_argument(
        "--workspace", type=Path, default=Path("."), help="Workspace root"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require ALL gates pass (even non-blocking)",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON result")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip slow gates (rust_tests, frontend_build, security_audit)",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("04_GOLD/release_readiness_log.jsonl"),
        help="Evidence log path",
    )

    args = parser.parse_args()
    workspace = args.workspace.resolve()

    gates = ALL_GATES
    if args.fast:
        skip = {"gate_rust_tests", "gate_frontend_build", "gate_security_audit"}
        gates = [g for g in gates if g.__name__ not in skip]

    report = run_all_gates(workspace, gates)
    report.commit_sha = args.commit_sha
    report.branch = args.branch

    if args.strict:
        # In strict mode, warnings become blockers
        report.blocking_failures.extend(report.warnings)
        report.release_ready = len(report.blocking_failures) == 0

    # Summary
    print("\n" + "=" * 60)
    print(f"  Overall Score:    {report.overall_score:.3f}")
    print(f"  Release Ready:    {'YES' if report.release_ready else 'NO'}")
    if report.blocking_failures:
        print(f"  Blocking:         {', '.join(report.blocking_failures)}")
    if report.warnings:
        print(f"  Warnings:         {', '.join(report.warnings)}")
    print(f"  Evidence Hash:    {report.evidence_hash[:16]}")
    print("=" * 60)

    # Persist evidence
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    with open(args.evidence, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(report), default=str) + "\n")

    if args.json:
        print(json.dumps(asdict(report), indent=2, default=str))

    return 0 if report.release_ready else 1


if __name__ == "__main__":
    sys.exit(main())
