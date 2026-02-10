#!/usr/bin/env python3
"""
Verify Artifacts — Auto-discovers and runs all available verification checks.

Detects the project ecosystem, finds test suites, benchmarks, linters, and
schema validators, then runs everything and produces a structured verification
report for the evidence receipt.

Usage:
    verify_artifacts.py --project-dir ./my-project
    verify_artifacts.py --project-dir ./my-project --output ./verification.json
    verify_artifacts.py --project-dir ./my-project --manual "Manually tested login flow end-to-end"
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


# ─── Ecosystem Detection ─────────────────────────────────────────────────────

def detect_ecosystem(project_dir: Path) -> dict:
    """
    Detect what ecosystem(s) the project uses and what checks are available.

    Returns a dict of discovered capabilities.
    """
    capabilities = {
        "ecosystems": [],
        "test_commands": [],
        "benchmark_commands": [],
        "lint_commands": [],
        "build_commands": [],
        "schema_files": [],
    }

    files = set()
    for item in project_dir.iterdir():
        if item.is_file():
            files.add(item.name.lower())

    # Rust
    if "cargo.toml" in files:
        capabilities["ecosystems"].append("rust")
        capabilities["test_commands"].append({"cmd": "cargo test", "type": "test_suite", "ecosystem": "rust"})
        capabilities["lint_commands"].append({"cmd": "cargo clippy -- -D warnings", "type": "static_analysis", "ecosystem": "rust"})
        capabilities["build_commands"].append({"cmd": "cargo check", "type": "build_check", "ecosystem": "rust"})
        if any(p.name.lower().endswith("bench.rs") or "bench" in p.name.lower()
               for p in project_dir.rglob("*.rs")):
            capabilities["benchmark_commands"].append({"cmd": "cargo bench", "type": "benchmark", "ecosystem": "rust"})

    # Node.js / TypeScript
    if "package.json" in files:
        capabilities["ecosystems"].append("node")
        pkg_path = project_dir / "package.json"
        try:
            with open(pkg_path) as f:
                pkg = json.load(f)
            scripts = pkg.get("scripts", {})
            if "test" in scripts:
                capabilities["test_commands"].append({"cmd": "npm test", "type": "test_suite", "ecosystem": "node"})
            if "lint" in scripts:
                capabilities["lint_commands"].append({"cmd": "npm run lint", "type": "static_analysis", "ecosystem": "node"})
            if "bench" in scripts or "benchmark" in scripts:
                capabilities["benchmark_commands"].append({"cmd": "npm run bench", "type": "benchmark", "ecosystem": "node"})
            if "build" in scripts:
                capabilities["build_commands"].append({"cmd": "npm run build", "type": "build_check", "ecosystem": "node"})
        except (json.JSONDecodeError, OSError):
            capabilities["test_commands"].append({"cmd": "npm test", "type": "test_suite", "ecosystem": "node"})

        # TypeScript
        if "tsconfig.json" in files:
            capabilities["ecosystems"].append("typescript")
            capabilities["lint_commands"].append({"cmd": "npx tsc --noEmit", "type": "static_analysis", "ecosystem": "typescript"})

    # Python
    if "pyproject.toml" in files or "setup.py" in files or "setup.cfg" in files:
        capabilities["ecosystems"].append("python")
        capabilities["test_commands"].append({"cmd": "python -m pytest", "type": "test_suite", "ecosystem": "python"})
        capabilities["lint_commands"].append({"cmd": "python -m ruff check .", "type": "static_analysis", "ecosystem": "python"})

    # Also detect Python if there are .py files with tests
    elif any(project_dir.rglob("test_*.py")) or any(project_dir.rglob("*_test.py")):
        capabilities["ecosystems"].append("python")
        capabilities["test_commands"].append({"cmd": "python -m pytest", "type": "test_suite", "ecosystem": "python"})

    # Go
    if "go.mod" in files:
        capabilities["ecosystems"].append("go")
        capabilities["test_commands"].append({"cmd": "go test ./...", "type": "test_suite", "ecosystem": "go"})
        capabilities["lint_commands"].append({"cmd": "go vet ./...", "type": "static_analysis", "ecosystem": "go"})
        capabilities["build_commands"].append({"cmd": "go build ./...", "type": "build_check", "ecosystem": "go"})

    # Makefile
    if "makefile" in files:
        capabilities["ecosystems"].append("make")
        makefile_path = project_dir / "Makefile"
        if not makefile_path.exists():
            makefile_path = project_dir / "makefile"
        try:
            content = makefile_path.read_text()
            if "test:" in content or "test :" in content:
                capabilities["test_commands"].append({"cmd": "make test", "type": "test_suite", "ecosystem": "make"})
            if "lint:" in content or "lint :" in content:
                capabilities["lint_commands"].append({"cmd": "make lint", "type": "static_analysis", "ecosystem": "make"})
            if "bench:" in content or "benchmark:" in content:
                capabilities["benchmark_commands"].append({"cmd": "make bench", "type": "benchmark", "ecosystem": "make"})
        except OSError:
            pass

    # Schema files
    for schema in project_dir.rglob("*.schema.json"):
        capabilities["schema_files"].append(str(schema))
    for schema in project_dir.rglob("*.xsd"):
        capabilities["schema_files"].append(str(schema))

    return capabilities


# ─── Command Execution ───────────────────────────────────────────────────────

def run_check(cmd: str, project_dir: Path, check_type: str, timeout: int = 120) -> dict:
    """
    Execute a verification check and capture results.

    Returns a structured check result.
    """
    print(f"    ▶ Running: {cmd}")
    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        duration_ms = int((time.time() - start) * 1000)
        passed = result.returncode == 0

        # Combine and truncate output
        output = (result.stdout + "\n" + result.stderr).strip()
        output_excerpt = output[:2000] if len(output) > 2000 else output

        # Try to parse test counts from output
        summary = _parse_test_summary(output, check_type)

        status = "✅" if passed else "❌"
        print(f"    {status} {check_type}: {'PASS' if passed else 'FAIL'} ({duration_ms}ms)")

        return {
            "type": check_type,
            "command": cmd,
            "exit_code": result.returncode,
            "passed": passed,
            "duration_ms": duration_ms,
            "summary": summary,
            "output_excerpt": output_excerpt,
        }

    except subprocess.TimeoutExpired:
        duration_ms = int((time.time() - start) * 1000)
        print(f"    ⏰ {check_type}: TIMEOUT after {timeout}s")
        return {
            "type": check_type,
            "command": cmd,
            "exit_code": -1,
            "passed": False,
            "duration_ms": duration_ms,
            "summary": f"Timed out after {timeout} seconds",
            "output_excerpt": "",
        }

    except Exception as e:
        duration_ms = int((time.time() - start) * 1000)
        print(f"    ⚠️  {check_type}: ERROR — {e}")
        return {
            "type": check_type,
            "command": cmd,
            "exit_code": -1,
            "passed": False,
            "duration_ms": duration_ms,
            "summary": f"Execution error: {e}",
            "output_excerpt": "",
        }


def _parse_test_summary(output: str, check_type: str) -> str:
    """Try to extract a human-readable summary from check output."""
    lines = output.strip().split("\n")

    # Look for common summary patterns
    for line in reversed(lines[-20:]):
        line_lower = line.lower().strip()
        # pytest
        if "passed" in line_lower and ("failed" in line_lower or "error" in line_lower or line_lower.startswith("=")):
            return line.strip()
        # cargo test
        if line_lower.startswith("test result:"):
            return line.strip()
        # jest / vitest
        if "tests:" in line_lower or "test suites:" in line_lower:
            return line.strip()
        # go test
        if line_lower.startswith("ok") or line_lower.startswith("fail"):
            return line.strip()

    # Fallback: last non-empty line
    for line in reversed(lines):
        if line.strip():
            return line.strip()[:200]

    return f"{check_type} completed"


# ─── Main Verification Flow ─────────────────────────────────────────────────

def run_verification(project_dir: Path, manual_attestation: str = None) -> dict:
    """
    Discover and run all available verification checks.

    Returns a structured verification report.
    """
    print(f"\n🔍 VERIFICATION — Scanning {project_dir}")

    capabilities = detect_ecosystem(project_dir)

    print(f"   Ecosystems detected: {', '.join(capabilities['ecosystems']) or 'none'}")
    print(f"   Test commands: {len(capabilities['test_commands'])}")
    print(f"   Lint commands: {len(capabilities['lint_commands'])}")
    print(f"   Benchmark commands: {len(capabilities['benchmark_commands'])}")
    print(f"   Build commands: {len(capabilities['build_commands'])}")
    print(f"   Schema files: {len(capabilities['schema_files'])}")

    checks = []
    all_commands = (
        capabilities["test_commands"]
        + capabilities["lint_commands"]
        + capabilities["benchmark_commands"]
        + capabilities["build_commands"]
    )

    if all_commands:
        print(f"\n   Running {len(all_commands)} check(s)...\n")
        for cmd_info in all_commands:
            result = run_check(cmd_info["cmd"], project_dir, cmd_info["type"])
            checks.append(result)
    else:
        print(f"\n   No automated checks available.")

    # Build report
    checks_run = len(checks)
    checks_passed = sum(1 for c in checks if c["passed"])
    checks_failed = checks_run - checks_passed
    overall_pass = checks_passed == checks_run if checks_run > 0 else None

    report = {
        "verification_id": f"verify-{int(time.time())}",
        "timestamp": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "ecosystems": capabilities["ecosystems"],
        "checks": checks,
        "checks_run": checks_run,
        "checks_passed": checks_passed,
        "checks_failed": checks_failed,
        "overall_pass": overall_pass,
    }

    if manual_attestation:
        report["manual_attestation"] = manual_attestation

    # Summary
    print(f"\n   📊 Results: {checks_passed}/{checks_run} passed", end="")
    if checks_failed > 0:
        print(f", {checks_failed} failed", end="")
    if manual_attestation:
        print(f" + manual attestation", end="")
    print()

    return report


def main():
    parser = argparse.ArgumentParser(description="Verify Artifacts — Auto-discovery & execution")
    parser.add_argument("--project-dir", required=True, help="Project directory")
    parser.add_argument("--output", help="Output path for verification JSON (default: stdout)")
    parser.add_argument("--manual", help="Manual attestation text (when no automated checks exist)")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout per check in seconds")

    args = parser.parse_args()
    project_dir = Path(args.project_dir).resolve()

    if not project_dir.is_dir():
        print(f"❌ Directory not found: {project_dir}")
        sys.exit(1)

    report = run_verification(project_dir, args.manual)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n✅ Verification report: {output_path}")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
